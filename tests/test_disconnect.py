"""Client-disconnect behaviour: detach-on-disconnect and SSE heartbeats."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import yaml

from mom.api.sse import with_heartbeat
from mom.config.resolve import ResolvedCatalog, resolve_catalog
from mom.config.schema import Config
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.results import Usage
from mom.engine.pipeline import PipelineDeps, _fan_out
from mom.engine.plan import resolve_plan
from mom.testing import ManualClock


class _GatedClient:
    """A member's ``complete`` blocks until ``release`` is set; records finish vs. cancel."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.finished: list[str] = []
        self.cancelled: list[str] = []

    async def complete(self, spec: CallSpec) -> Completion:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(spec.llm_name)
            raise
        self.finished.append(spec.llm_name)
        return Completion(content="r", reasoning=None, finish_reason="stop", usage=Usage())

    async def stream(self, spec: CallSpec) -> Any:
        yield CompletionChunk(finish_reason="stop", usage=Usage())


def _plan(detach: bool):
    catalog: ResolvedCatalog = resolve_catalog(
        Config.model_validate(
            yaml.safe_load(
                f"""
                version: 2
                defaults: {{ fanout: {{ detach_on_disconnect: {str(detach).lower()} }} }}
                llms:
                  a: {{ model: openai/a }}
                  b: {{ model: openai/b }}
                  s: {{ model: openai/s }}
                prompts: {{ p: synth }}
                ensembles:
                  e:
                    members: [{{ llm: a }}, {{ llm: b }}]
                    synthesizer: {{ llm: s, prompt: p }}
                """
            )
        )
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    return resolve_plan(catalog, ir)


class _Recorder:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def record(self, metric: Any) -> None:
        self.records.append(metric)


async def _fanout_then_disconnect(detach: bool) -> tuple[_GatedClient, _Recorder]:
    """Launch the fan-out, block members mid-flight, then simulate a client disconnect."""
    plan = _plan(detach)
    client = _GatedClient()
    recorder = _Recorder()
    deps = PipelineDeps(client=client, clock=ManualClock(), recorder=recorder)
    gen = _fan_out(deps, plan)
    for _ in plan.members:  # drain the FanoutStarted events (tasks are created on the next pull)
        await gen.__anext__()
    pull = asyncio.ensure_future(gen.__anext__())  # creates the member tasks; blocks on the members
    await asyncio.sleep(0.05)
    assert not pull.done()  # members are in-flight, blocked on `release`
    pull.cancel()  # <- what with_heartbeat/Starlette do on disconnect: cancel the outstanding pull
    with contextlib.suppress(asyncio.CancelledError):
        await pull  # the fan-out's `finally` runs here (detach or cancel)
    return client, recorder


async def test_detach_lets_members_finish_and_cache() -> None:
    client, recorder = await _fanout_then_disconnect(detach=True)
    assert client.finished == []  # nothing finished yet — they're detached, still blocked
    client.release.set()  # the detached members now run to completion (and would cache)
    await asyncio.sleep(0.05)
    assert set(client.finished) == {"a", "b"}
    assert client.cancelled == []
    # the spend really happened, so the detached members are still metered (not silently free)
    assert {m.llm for m in recorder.records} == {"a", "b"}


async def test_default_cancels_members_on_disconnect() -> None:
    client, recorder = await _fanout_then_disconnect(detach=False)
    await asyncio.sleep(0.05)
    assert set(client.cancelled) == {"a", "b"}  # cancelled on disconnect (zero orphaned spend)
    assert client.finished == []
    assert recorder.records == []  # cancelled before completing -> nothing to record


async def _collect(stream, limit: int) -> list[bytes]:
    out: list[bytes] = []
    async for chunk in stream:
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


async def test_heartbeat_injected_on_idle_gap() -> None:
    async def slow():
        await asyncio.sleep(0.06)  # longer than the heartbeat interval
        yield b"data: a\n\n"

    out = await _collect(with_heartbeat(slow(), interval=0.02, comment=b": hb\n\n"), limit=3)
    assert b": hb\n\n" in out  # keepalive emitted during the idle gap
    assert out[-1] == b"data: a\n\n"  # the real frame still arrives, after the heartbeats


async def test_heartbeat_absent_when_stream_is_prompt() -> None:
    async def fast():
        yield b"data: a\n\n"
        yield b"data: b\n\n"

    out = [c async for c in with_heartbeat(fast(), interval=1.0)]
    assert out == [b"data: a\n\n", b"data: b\n\n"]  # no idle gap -> no heartbeat
