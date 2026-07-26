"""Pipeline robustness: error hygiene, status preservation, timeouts, tool ids, metrics survival."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import yaml

from mom.config.resolve import ResolvedCatalog, resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import MomError
from mom.domain.events import PipelineFailed, StreamEvent
from mom.domain.metrics import CallMetric
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.results import Usage
from mom.engine.pipeline import PipelineDeps, _run_member, collect, run_ensemble
from mom.engine.plan import PlannedMember, resolve_plan
from mom.store.metrics import MetricsRecorder
from mom.testing import ManualClock


def _catalog(text: str) -> ResolvedCatalog:
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


async def _events(items: list[StreamEvent]) -> Any:
    for item in items:
        yield item


class _RawFailClient:
    async def complete(self, spec: CallSpec) -> Completion:
        raise ValueError("SECRET provider detail: key=sk-abc123")

    async def stream(self, spec: CallSpec) -> Any:
        yield CompletionChunk(finish_reason="stop", usage=Usage())


async def test_member_error_does_not_leak_raw_exception():
    member = PlannedMember(identity="m", spec=CallSpec(llm_name="m", model="openai/x", messages=[]))
    outcome = await _run_member(PipelineDeps(client=_RawFailClient(), clock=ManualClock()), member)
    assert outcome.status == "error"
    assert outcome.error == "call failed"  # generic, not the raw provider text
    assert "SECRET" not in (outcome.error or "")


async def test_collect_preserves_pipeline_failed_status():
    failed = PipelineFailed(code="timeout", message="synthesis timed out", http_status=504)
    with pytest.raises(MomError) as exc_info:
        await collect(_events([failed]))
    assert exc_info.value.http_status == 504
    assert exc_info.value.code == "timeout"


class _SlowSynthClient:
    async def complete(self, spec: CallSpec) -> Completion:
        return Completion(content="member", reasoning=None, finish_reason="stop", usage=Usage())

    async def stream(self, spec: CallSpec) -> Any:
        await asyncio.sleep(0.5)  # stall much longer than the synth timeout
        yield CompletionChunk(content="late", finish_reason="stop", usage=Usage())


async def test_synth_stream_timeout_is_504():
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
          s: { model: openai/s, timeout: 20ms }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: s }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    deps = PipelineDeps(client=_SlowSynthClient(), clock=ManualClock())
    with pytest.raises(MomError) as exc_info:
        await collect(run_ensemble(plan, deps))
    assert exc_info.value.http_status == 504


class _ArgsFirstToolClient:
    async def complete(self, spec: CallSpec) -> Completion:
        return Completion(content="member", reasoning=None, finish_reason="stop", usage=Usage())

    async def stream(self, spec: CallSpec) -> Any:
        # A provider that streams arguments before the id/name are known.
        yield CompletionChunk(tool_call={"index": 0, "id": None, "name": None, "arguments": "{}"})
        yield CompletionChunk(finish_reason="tool_calls", usage=Usage())


async def test_tool_call_id_never_literal_none():
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    deps = PipelineDeps(client=_ArgsFirstToolClient(), clock=ManualClock())
    result = await collect(run_ensemble(plan, deps))
    assert result.tool_calls
    # Even when the provider streams no id, a client id is always minted — never "" or the literal
    # "None" (the args-first race that produced a bare "None" in v1).
    minted = result.tool_calls[0]["id"]
    assert minted not in ("", "None")
    assert minted.startswith("call")


class _FlakyStore:
    def __init__(self) -> None:
        self.calls = 0
        self.written: list[CallMetric] = []

    async def insert_many(self, metrics: list[CallMetric]) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("disk full")
        self.written.extend(metrics)


def _metric() -> CallMetric:
    return CallMetric(
        request_id="r", ts=1.0, ensemble="e", llm="a", model="m", role="fanout", status="ok"
    )


async def test_metrics_recorder_survives_insert_error():
    store = _FlakyStore()
    recorder = MetricsRecorder(store, maxsize=100, batch=1)  # type: ignore[arg-type]
    await recorder.start()
    recorder.record(_metric())  # first drain -> insert raises -> the worker must NOT die
    await asyncio.sleep(0.05)
    recorder.record(_metric())  # second drain -> succeeds after the backoff
    await asyncio.sleep(0.7)  # allow the ~0.5s post-error backoff, then the retry
    await recorder.stop()
    assert store.calls >= 2  # the worker kept running after the first failure
    assert len(store.written) >= 1
