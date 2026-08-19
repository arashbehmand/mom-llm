"""Pipeline robustness: error hygiene, status preservation, timeouts, tool ids, metrics survival."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import structlog
import yaml

from mom.adapters.eventbus import InMemoryEventBus
from mom.config.resolve import ResolvedCatalog, resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import MomError, UpstreamError
from mom.domain.events import MemberCompleted, PipelineFailed, StreamEvent
from mom.domain.metrics import CallMetric
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.results import Usage
from mom.engine.pipeline import (
    _FREE_MODELS_REPORTED,
    PipelineDeps,
    _cause_text,
    _fan_out,
    _record_member,
    _run_member,
    _warn_once_if_free,
    collect,
    run_ensemble,
)
from mom.engine.plan import PlannedMember, resolve_plan
from mom.store.metrics import MetricsRecorder
from mom.testing import FakeLLM, ManualClock


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


class _MomErrorClient:
    async def complete(self, spec: CallSpec) -> Completion:
        cause = ValueError("SECRET provider detail: key=sk-abc123")
        raise UpstreamError("upstream error") from cause

    async def stream(self, spec: CallSpec) -> Any:
        yield CompletionChunk(finish_reason="stop", usage=Usage())


async def test_mom_error_from_member_call_is_logged_for_the_operator():
    """Regression guard: this branch used to `return` silently (Phase 0 of the v2 remediation) —
    UpstreamError IS a MomError, so the generic `except Exception` handler (which does log) never
    ran for the failures that actually mattered, and operators had zero visibility into why a
    member failed."""
    member = PlannedMember(identity="m", spec=CallSpec(llm_name="m", model="openai/x", messages=[]))
    with structlog.testing.capture_logs() as logs:
        outcome = await _run_member(
            PipelineDeps(client=_MomErrorClient(), clock=ManualClock(), request_id="req-1"), member
        )
    assert outcome.status == "error"
    assert outcome.error == "upstream error"  # client-facing message stays generic
    warnings = [log for log in logs if log["log_level"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["event"] == "member call failed"
    assert warnings[0]["llm"] == "m"
    assert warnings[0]["request_id"] == "req-1"
    assert "SECRET" in warnings[0]["error"]  # the real cause, for logs/metrics only


def test_cause_text_walks_the_cause_chain_and_truncates():
    root = ValueError("root cause")
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = root
    text = _cause_text(wrapped)
    assert "wrapper" in text
    assert "root cause" in text
    assert len(_cause_text(ValueError("x" * 1000), max_chars=50)) == 50


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


class _OneMemberHangsClient:
    async def complete(self, spec: CallSpec) -> Completion:
        if spec.llm_name == "slow":
            await asyncio.sleep(2.0)  # far longer than the fanout deadline below
        return Completion(
            content=f"reply from {spec.llm_name}",
            reasoning=None,
            finish_reason="stop",
            usage=Usage(),
        )

    async def stream(self, spec: CallSpec) -> Any:
        yield CompletionChunk(content="synthesized", finish_reason="stop", usage=Usage())


async def test_fanout_deadline_stops_waiting_on_a_stuck_member():
    # Live bug 2026-07-28: emom's 11 other members finished in 6s-151s, but `ink` hung for 12+
    # minutes with nothing forcing the panel to move on (no fanout.deadline was configured, so
    # the loop waits for literally every member, however long that takes, up to its own 20m call
    # timeout). A deadline lets a healthy quorum synthesize without waiting out a straggler.
    catalog = _catalog(
        """
        version: 2
        defaults:
          fanout: { deadline: 30ms }
        llms:
          fast: { model: openai/fast }
          slow: { model: openai/slow }
        ensembles:
          e:
            members: [fast, slow]
            synthesizer: { llm: fast }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    deps = PipelineDeps(client=_OneMemberHangsClient(), clock=ManualClock())

    result = await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)
    assert result.text == "synthesized"  # synthesis proceeded without the stuck member


def _deadline_catalog(*, detach: bool, min_results: int = 1) -> ResolvedCatalog:
    return _catalog(
        f"""
        version: 2
        defaults:
          fanout: {{ deadline: 30ms, detach_on_disconnect: {str(detach).lower()},
                     min_results: {min_results} }}
        llms:
          fast: {{ model: openai/fast }}
          slow: {{ model: openai/slow }}
        ensembles:
          e:
            members: [fast, slow]
            synthesizer: {{ llm: fast }}
        """
    )


async def test_fanout_deadline_publishes_detached_member_progress():
    """Regression guard for the stuck-dashboard-card bug (fixed alongside the deadline feature
    itself in this session): every announced member must get exactly one resolving
    `member_completed` event — previously a straggler abandoned at the deadline got NONE, so its
    dashboard card stayed "pending" forever even though the response completed normally."""
    catalog = _deadline_catalog(detach=True)
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    bus = InMemoryEventBus()
    client = FakeLLM(delays={"slow": 0.15})  # far longer than the 30ms deadline
    deps = PipelineDeps(client=client, clock=ManualClock(), bus=bus, request_id="req-1")

    result = await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)
    assert result.text == "synthesized answer"

    events = [event async for event in bus.subscribe("req-1")]
    assert [e.kind for e in events] == [
        "fanout_started",
        "member_completed",
        "member_completed",
        "synthesis_started",
        "completed",
    ]
    by_member = {e.member: e for e in events if e.kind == "member_completed"}
    assert by_member["fast"].status == "ok"
    assert by_member["slow"].status == "detached"  # not silently absent
    assert by_member["slow"].preview  # a human-readable note, not empty
    assert by_member["slow"].duration_ms == 30.0  # the deadline, not the eventual real duration

    await asyncio.sleep(0.2)  # let the detached background call finish before the test ends


async def test_fanout_deadline_publishes_aborted_status_when_not_detaching():
    catalog = _deadline_catalog(detach=False)
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    bus = InMemoryEventBus()
    client = FakeLLM(delays={"slow": 0.15})
    deps = PipelineDeps(client=client, clock=ManualClock(), bus=bus, request_id="req-2")

    await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    events = [event async for event in bus.subscribe("req-2")]
    by_member = {e.member: e for e in events if e.kind == "member_completed"}
    assert by_member["slow"].status == "aborted"  # cancelled outright, not left running


async def test_quorum_failure_after_deadline_still_resolves_abandoned_members_then_fails():
    """Both members hang past the deadline -> 0 successes -> QuorumNotMet. The abandoned-member
    events must still land BEFORE the terminal `failed`, so a dashboard watching this request
    resolves every card instead of freezing mid-panel right as the error banner appears."""
    catalog = _catalog(
        """
        version: 2
        defaults:
          fanout: { deadline: 30ms, detach_on_disconnect: true, min_results: 1 }
        llms:
          a: { model: openai/a }
          b: { model: openai/b }
        ensembles:
          e:
            members: [a, b]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    bus = InMemoryEventBus()
    client = FakeLLM(delays={"a": 0.15, "b": 0.15})
    deps = PipelineDeps(client=client, clock=ManualClock(), bus=bus, request_id="req-3")

    with pytest.raises(MomError):
        await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    events = [event async for event in bus.subscribe("req-3")]
    kinds = [e.kind for e in events]
    assert kinds == ["fanout_started", "member_completed", "member_completed", "failed"]
    assert {e.member for e in events if e.kind == "member_completed"} == {"a", "b"}
    assert all(e.status == "detached" for e in events if e.kind == "member_completed")

    await asyncio.sleep(0.2)  # let both detached calls finish before the test ends


async def test_multiple_stragglers_past_deadline_abandoned_in_plan_order():
    catalog = _catalog(
        """
        version: 2
        defaults:
          fanout: { deadline: 30ms, detach_on_disconnect: true }
        llms:
          fast: { model: openai/fast }
          slow1: { model: openai/slow1 }
          slow2: { model: openai/slow2 }
        ensembles:
          e:
            members: [fast, slow1, slow2]
            synthesizer: { llm: fast }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    bus = InMemoryEventBus()
    client = FakeLLM(delays={"slow1": 0.15, "slow2": 0.15})
    deps = PipelineDeps(client=client, clock=ManualClock(), bus=bus, request_id="req-4")

    await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    events = [event async for event in bus.subscribe("req-4")]
    abandoned = [
        e.member for e in events if e.kind == "member_completed" and e.status == "detached"
    ]
    assert abandoned == ["slow1", "slow2"]  # plan (config) order, not set-iteration order

    await asyncio.sleep(0.2)


async def test_vote_first_short_circuit_publishes_a_terminal_completed_event():
    """Regression: this path used to `return` with no terminal progress event at all, so a
    dashboard watching a vote/first turn hung open forever even after the client got its answer."""
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            members: [a]
            synthesizer: { llm: a }
            tools: { strategy: first }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    bus = InMemoryEventBus()
    client = FakeLLM(member_tool_calls={"a": ({"id": "c1", "name": "fn", "arguments": "{}"},)})
    deps = PipelineDeps(client=client, clock=ManualClock(), bus=bus, request_id="req-5")

    await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    events = [event async for event in bus.subscribe("req-5")]
    assert events[-1].kind == "completed"
    assert events[-1].status == "tool_calls"


async def test_skip_fanout_turn_publishes_fanout_started_with_zero_members():
    """Regression for Issue 2: a passthrough/relay turn used to publish NOTHING, so a dashboard
    opened on such a turn showed a blank page indistinguishable from a stuck/broken request."""
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            strategy: passthrough
            members: [a]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    assert plan.skip_fanout
    bus = InMemoryEventBus()
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), bus=bus, request_id="req-6")

    await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    events = [event async for event in bus.subscribe("req-6")]
    assert events[0].kind == "fanout_started"
    assert events[0].members_total == 0
    assert events[-1].kind == "completed"


async def test_skip_fanout_turn_still_carries_the_system_instruction():
    """Regression: the <<SYSTEM>>/<<CONCLUDING-INSTRUCTION>> instruction is stripped from the
    client message during plan resolution on EVERY path, but a passthrough/relay turn used to
    never re-attach it anywhere — it was silently discarded instead of reaching the synthesizer."""
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            strategy: passthrough
            members: [a]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(
        model="e",
        messages=(MessageIR(role="user", content="hi <<SYSTEM>>Reply in Farsi.<</SYSTEM>>"),),
    )
    plan = resolve_plan(catalog, ir)
    assert plan.skip_fanout
    assert plan.instruction == "Reply in Farsi."
    client = FakeLLM()
    deps = PipelineDeps(client=client, clock=ManualClock())

    await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=1.0)

    (spec,) = client.streams
    assert spec.messages[-1] == {"role": "user", "content": "Reply in Farsi."}


class _BothInstantClient:
    """Both members complete with no internal ``await`` at all, so — once ``_fan_out`` creates
    their tasks and suspends on ``asyncio.wait`` — the event loop runs each task to completion in
    its very first scheduling step. Both therefore land in the SAME ``asyncio.wait`` ``done`` set,
    deterministically (not dependent on manually racing a shared gate)."""

    async def complete(self, spec: CallSpec) -> Completion:
        return Completion(
            content=f"reply from {spec.llm_name}",
            reasoning=None,
            finish_reason="stop",
            usage=Usage(),
        )

    async def stream(self, spec: CallSpec) -> Any:
        yield CompletionChunk(finish_reason="stop", usage=Usage())


class _Recorder:
    def __init__(self) -> None:
        self.records: list[CallMetric] = []

    def record(self, metric: CallMetric) -> None:
        self.records.append(metric)


async def test_member_landing_after_teardown_is_still_metered():
    """The core correctness fix in ``_fan_out``: ``pending`` means "not yet accounted for", not
    "not yet returned by this asyncio.wait round". Two members finish in the SAME round; the
    generator is torn down (``.aclose()``, what a client disconnect ultimately triggers) right
    after the FIRST of them is yielded. The SECOND, though already done, must still be detached
    in `finally` — previously it was silently dropped (no event, no metric) because
    ``task.done()`` was already True for it and the old ``if not task.done()`` guard skipped it."""
    catalog = _catalog(
        """
        version: 2
        defaults:
          fanout: { detach_on_disconnect: true }
        llms:
          a: { model: openai/a }
          b: { model: openai/b }
        ensembles:
          e:
            members: [a, b]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    recorder = _Recorder()
    deps = PipelineDeps(client=_BothInstantClient(), clock=ManualClock(), recorder=recorder)

    gen = _fan_out(deps, plan)
    for _ in plan.members:  # drain the two FanoutStarted events (tasks not created yet)
        await gen.__anext__()
    first_event = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert isinstance(first_event, MemberCompleted)  # the first of the batch, already delivered
    # Mirror what run_ensemble's consumption loop does for every MemberCompleted it receives — the
    # real caller, not exercised here since we're driving `_fan_out` directly to control teardown
    # timing precisely.
    _record_member(deps, plan, first_event.outcome, "ensemble")

    await gen.aclose()  # teardown right here — before the second event is ever pulled
    await asyncio.sleep(0.05)  # let the detach done-callback run

    assert {m.llm for m in recorder.records} == {"a", "b"}  # BOTH metered, not just the first


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


# --------------------------------------------------------------------------------------------
# Unpriced-model detection. A model missing from litellm's catalog has no cost-per-token, so its
# spend records as $0 — silently, unless the provider self-reports one. Observed live: Gemini and
# xAI members had billed to $0.00 across 230 calls, and Opus 5 joined them while its catalog entry
# was missing. Whether a provider self-reports is a runtime fact, so this check runs on the result.
# --------------------------------------------------------------------------------------------


def _clear_free_model_reports() -> None:
    _FREE_MODELS_REPORTED.clear()


def test_zero_cost_call_that_burned_tokens_is_reported() -> None:
    _clear_free_model_reports()
    with structlog.testing.capture_logs() as logs:
        _warn_once_if_free(
            "gemini/gemini-3.7-flash", Usage(prompt_tokens=500, completion_tokens=9), 0.0
        )
    assert [entry for entry in logs if entry["log_level"] == "warning"]
    assert logs[0]["model"] == "gemini/gemini-3.7-flash"


def test_zero_cost_is_reported_once_per_model_not_once_per_call() -> None:
    """Spend is a per-model property; repeating the line every call would bury it."""
    _clear_free_model_reports()
    usage = Usage(prompt_tokens=10, completion_tokens=1)
    with structlog.testing.capture_logs() as logs:
        for _ in range(5):
            _warn_once_if_free("xai/grok-4.6", usage, 0.0)
    assert len(logs) == 1


def test_priced_call_is_not_reported() -> None:
    _clear_free_model_reports()
    with structlog.testing.capture_logs() as logs:
        _warn_once_if_free(
            "anthropic/claude-opus-5", Usage(prompt_tokens=10, completion_tokens=1), 0.42
        )
    assert logs == []


def test_token_free_call_is_not_reported() -> None:
    """A call that used no tokens costing nothing is arithmetic, not a missing catalog entry."""
    _clear_free_model_reports()
    with structlog.testing.capture_logs() as logs:
        _warn_once_if_free("openrouter/some/model", Usage(), 0.0)
    assert logs == []
