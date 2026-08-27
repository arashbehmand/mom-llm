"""Engine: plan resolution + the fan-out -> synthesize pipeline (with a fake provider)."""

from __future__ import annotations

import asyncio
from textwrap import dedent

import pytest
from structlog.testing import capture_logs
import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import MomError, UnknownModelError, UpstreamError
from mom.domain.events import (
    AnswerDelta,
    Completed,
    FanoutSkipped,
    FanoutStarted,
    MemberCompleted,
)
from mom.domain.metrics import CallMetric
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.coalesce import CoalesceRegistry
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.testing import FakeLLM, ManualClock, RecordingTracer


def _catalog(text: str):
    return resolve_catalog(Config.model_validate(yaml.safe_load(dedent(text))))


CONFIG = """
version: 2
llms:
  a: { model: openai/a }
  b: { model: openai/b }
ensembles:
  e:
    members: [{ llm: a }, { llm: b }]
    synthesizer: { llm: a, prompt: p }
  solo:
    strategy: passthrough
    members: [{ llm: a }]
    synthesizer: { llm: a }
prompts:
  p: "synthesize the responses"
"""


def _ir(model: str = "e") -> ChatRequestIR:
    return ChatRequestIR(model=model, messages=(MessageIR(role="user", content="hi"),))


async def _events(model: str, client: FakeLLM):
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir(model))
    deps = PipelineDeps(client=client, clock=ManualClock())
    return [event async for event in run_ensemble(plan, deps)]


async def test_collect_synthesizes_from_members():
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock())
    result = await collect(run_ensemble(plan, deps))
    assert result.text == "synthesized answer"
    assert len(result.outcomes) == 2
    assert all(o.ok for o in result.outcomes)
    assert result.finish_reason == "stop"


async def test_event_order():
    events = await _events("e", FakeLLM())
    kinds = [type(e).__name__ for e in events]
    assert kinds[:2] == ["FanoutStarted", "FanoutStarted"]
    assert kinds.count("MemberCompleted") == 2
    # work events precede the first answer delta
    assert kinds.index("SynthesisStarted") < kinds.index("AnswerDelta")
    assert isinstance(events[-1], Completed)


async def test_member_failure_still_synthesizes():
    client = FakeLLM(fail=frozenset({"b"}))
    events = await _events("e", client)
    outcomes = [e.outcome for e in events if isinstance(e, MemberCompleted)]
    by_id = {o.identity: o for o in outcomes}
    assert by_id["b"].status == "error"
    assert by_id["a"].ok
    # synthesis still runs (a succeeded)
    assert any(isinstance(e, AnswerDelta) for e in events)


async def test_passthrough_skips_fanout():
    client = FakeLLM()
    events = await _events("solo", client)
    assert any(isinstance(e, FanoutSkipped) for e in events)
    assert not any(isinstance(e, FanoutStarted) for e in events)
    assert client.completions == []  # no member calls
    assert len(client.streams) == 1  # only the synthesizer streamed


async def test_unknown_model_raises_at_plan_time():
    catalog = _catalog(CONFIG)
    with pytest.raises(UnknownModelError):
        resolve_plan(catalog, _ir("ghost"))


QUORUM_CONFIG = """
version: 2
defaults: {{ fanout: {{ min_results: {min_results} }} }}
llms:
  a: {{ model: openai/a }}
  b: {{ model: openai/b }}
ensembles:
  e:
    members: [{{ llm: a }}, {{ llm: b }}]
    synthesizer: {{ llm: a }}
"""


async def test_quorum_not_met_fails_when_too_few_members_succeed():
    catalog = _catalog(QUORUM_CONFIG.format(min_results=2))
    plan = resolve_plan(catalog, _ir("e"))
    assert plan.min_results == 2
    deps = PipelineDeps(client=FakeLLM(fail=frozenset({"b"})), clock=ManualClock())
    with pytest.raises(MomError) as exc_info:  # only 1 of 2 ok -> below quorum
        await collect(run_ensemble(plan, deps))
    assert exc_info.value.code == "quorum_not_met"
    assert exc_info.value.http_status == 502


async def test_quorum_met_synthesizes():
    catalog = _catalog(QUORUM_CONFIG.format(min_results=2))
    plan = resolve_plan(catalog, _ir("e"))
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock())
    result = await collect(run_ensemble(plan, deps))
    assert result.text == "synthesized answer"  # both members ok -> quorum met


async def test_min_results_zero_re_enables_all_failed_fallback():
    # min_results=0 disables the quorum gate, restoring the graceful all-failed synthesis path.
    catalog = _catalog(QUORUM_CONFIG.format(min_results=0))
    plan = resolve_plan(catalog, _ir("e"))
    deps = PipelineDeps(client=FakeLLM(fail=frozenset({"a", "b"})), clock=ManualClock())
    result = await collect(run_ensemble(plan, deps))
    assert result.text == "synthesized answer"  # synthesizer still runs on the all-failed fallback
    assert all(not o.ok for o in result.outcomes)


# --- lifecycle logging (issue #31) -------------------------------------------------------------
# An operator watching `docker logs` must be able to follow a request at the DEFAULT level. These
# pin the sequence and, just as importantly, that no message content ever rides along.


class _TickingClock(ManualClock):
    """A clock that advances on every read, so durations are non-zero without manual advances."""

    def __init__(self, start: float = 1000.0, tick: float = 0.05) -> None:
        super().__init__(start)
        self._tick = tick

    def now(self) -> float:
        value = super().now()
        self.advance(self._tick)
        return value


class _Recorder:
    def __init__(self) -> None:
        self.records: list[CallMetric] = []

    def record(self, metric: CallMetric) -> None:
        self.records.append(metric)


def _info_events(logs: list[dict[str, object]]) -> list[str]:
    return [str(entry["event"]) for entry in logs if entry["log_level"] == "info"]


async def test_happy_path_emits_lifecycle_info_lines():
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), request_id="req-42")
    with capture_logs() as logs:
        await collect(run_ensemble(plan, deps))

    assert _info_events(logs) == [
        "fan-out started",
        "member dispatched",
        "member dispatched",
        "member completed",
        "member completed",
        "synthesis started",
        "run completed",
    ]
    # Every line is greppable by request: that is what makes interleaved fan-out readable.
    assert all(entry["request_id"] == "req-42" for entry in logs if entry["log_level"] == "info")
    assert all(entry["ensemble"] == "e" for entry in logs if entry["log_level"] == "info")

    completed = [e for e in logs if e["event"] == "member completed"]
    assert {str(e["llm"]) for e in completed} == {"a", "b"}
    assert all(e["status"] == "ok" and e["cached"] is False for e in completed)

    done = next(e for e in logs if e["event"] == "run completed")
    assert done["status"] == "stop"
    assert done["members_ok"] == 2
    assert done["members_total"] == 2
    assert {"total_cost_usd", "total_tokens", "elapsed_seconds"} <= set(done)


async def test_lifecycle_lines_carry_no_message_content():
    """The redaction guard for issue #31: statuses, timings and costs are fine to log; model
    output is not. Sentinel text in both fan-out replies and the synthesized answer must not
    appear anywhere in the captured records."""
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    client = FakeLLM(
        replies={"a": "SECRET-A", "b": "SECRET-B"},
        synth_chunks=("SECRET-SYNTH",),
    )
    deps = PipelineDeps(client=client, clock=ManualClock(), request_id="req-1")
    with capture_logs() as logs:
        result = await collect(run_ensemble(plan, deps))

    assert result.text == "SECRET-SYNTH"  # the content really did flow through the pipeline
    assert "SECRET" not in repr(logs)


async def test_passthrough_turn_still_logs_its_lifecycle():
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir("solo"))
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), request_id="req-2")
    with capture_logs() as logs:
        await collect(run_ensemble(plan, deps))

    # No members to dispatch, but the turn explains itself rather than going silent.
    assert _info_events(logs) == ["fan-out skipped", "synthesis started", "run completed"]
    assert next(e for e in logs if e["event"] == "fan-out skipped")["reason"] == "passthrough"


async def test_quorum_failure_logs_a_terminal_run_failed_line():
    catalog = _catalog(QUORUM_CONFIG.format(min_results=2))
    plan = resolve_plan(catalog, _ir("e"))
    deps = PipelineDeps(
        client=FakeLLM(fail=frozenset({"b"})), clock=ManualClock(), request_id="req-3"
    )
    with capture_logs() as logs, pytest.raises(MomError):
        await collect(run_ensemble(plan, deps))

    failed = next(e for e in logs if e["event"] == "run failed")
    assert failed["log_level"] == "warning"
    assert failed["code"] == "quorum_not_met"
    assert failed["request_id"] == "req-3"
    assert "elapsed_seconds" in failed


async def test_failed_member_is_still_counted_in_the_lifecycle():
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    deps = PipelineDeps(
        client=FakeLLM(fail=frozenset({"b"})), clock=ManualClock(), request_id="req-4"
    )
    with capture_logs() as logs:
        await collect(run_ensemble(plan, deps))

    # Members are counted in and out even when one fails, so a stuck member is visible by absence.
    assert _info_events(logs).count("member dispatched") == 2
    completed = [e for e in logs if e["event"] == "member completed"]
    assert len(completed) == 2
    failed = next(e for e in completed if e["llm"] == "b")
    assert failed["status"] == "error"
    assert failed["error_kind"]  # classified, not raw provider text
    assert next(e for e in logs if e["event"] == "run completed")["members_ok"] == 1


async def test_synthesis_duration_is_recorded():
    """Synthesis timing used to be hardcoded to 0.0 in both the metric and the trace."""
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    recorder, tracer = _Recorder(), RecordingTracer()
    deps = PipelineDeps(
        client=FakeLLM(),
        clock=_TickingClock(),
        recorder=recorder,
        tracer=tracer,
        request_id="req-5",
    )
    with capture_logs() as logs:
        await collect(run_ensemble(plan, deps))

    synth_metric = next(m for m in recorder.records if m.role == "synthesis")
    assert synth_metric.duration_ms is not None
    assert synth_metric.duration_ms > 0
    synth_trace = next(o for o in tracer.observations if o["role"] == "synthesis")
    assert isinstance(synth_trace["duration_ms"], float)
    assert synth_trace["duration_ms"] > 0
    assert float(str(next(e for e in logs if e["event"] == "run completed")["synthesis_ms"])) > 0


async def test_timed_out_member_reports_a_real_duration():
    """A member that timed out used to report duration_ms=0 — zero on exactly the outcome an
    operator is trying to time."""
    catalog = _catalog(
        """
        version: 2
        defaults: { fanout: { min_results: 0 } }  # let the run finish so we can read its log
        llms:
          slow: { model: openai/slow, timeout: 20ms }
        ensembles:
          e:
            members: [{ llm: slow }]
            synthesizer: { llm: slow }
        """
    )
    plan = resolve_plan(catalog, _ir("e"))
    recorder = _Recorder()
    deps = PipelineDeps(
        client=FakeLLM(delays={"slow": 0.15}),
        clock=_TickingClock(),
        recorder=recorder,
        request_id="req-6",
    )
    with capture_logs() as logs:
        await collect(run_ensemble(plan, deps))

    entry = next(e for e in logs if e["event"] == "member completed")
    assert entry["status"] == "timeout"
    assert isinstance(entry["duration_ms"], float)
    assert entry["duration_ms"] > 0
    member_metric = next(m for m in recorder.records if m.role == "fanout")
    assert member_metric.duration_ms is not None
    assert member_metric.duration_ms > 0


async def test_internal_error_logs_a_terminal_line():
    """An unexpected exception becomes a PipelineFailed event rather than propagating, so it never
    reaches the API error handler — this log line is the only place such a bug is visible."""

    class _BrokenSynth(FakeLLM):
        async def stream(self, spec):
            raise ValueError("bug in the pipeline")
            yield  # pragma: no cover - makes this an async generator

    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    deps = PipelineDeps(client=_BrokenSynth(), clock=ManualClock(), request_id="req-7")
    with capture_logs() as logs:
        events = [event async for event in run_ensemble(plan, deps)]

    assert any(type(e).__name__ == "PipelineFailed" for e in events)
    failed = next(e for e in logs if e["event"] == "run failed")
    assert failed["log_level"] == "error"
    assert failed["code"] == "internal_error"
    assert failed["request_id"] == "req-7"
    assert "bug in the pipeline" not in repr({k: v for k, v in failed.items() if k != "exc_info"})


async def test_coalesced_followers_do_not_duplicate_lifecycle_lines():
    """Dedupe attaches followers to the leader's event stream, so the run — and its logging —
    happens once. Were it otherwise, a burst of identical requests would multiply every line."""
    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    registry = CoalesceRegistry()
    client = FakeLLM(delays={"a": 0.05, "b": 0.05})  # hold the run open long enough to attach

    with capture_logs() as logs:
        leader_events, leader_id = registry.attach(
            "same-request",
            "req-leader",
            lambda: run_ensemble(
                plan, PipelineDeps(client=client, clock=ManualClock(), request_id="req-leader")
            ),
        )
        follower_events, follower_leader_id = registry.attach(
            "same-request",
            "req-follower",
            lambda: run_ensemble(
                plan, PipelineDeps(client=client, clock=ManualClock(), request_id="req-follower")
            ),
        )
        assert follower_leader_id == leader_id == "req-leader"  # genuinely coalesced
        both = await asyncio.gather(_drain_events(leader_events), _drain_events(follower_events))

    assert all(any(type(e).__name__ == "Completed" for e in stream) for stream in both)
    assert _info_events(logs).count("run completed") == 1
    assert _info_events(logs).count("member dispatched") == 2  # two members, not two per subscriber
    assert all(e["request_id"] == "req-leader" for e in logs if e["log_level"] == "info")


async def _drain_events(events):
    return [event async for event in events]


_DEADLINE_CONFIG = """
version: 2
defaults: {{ fanout: {{ deadline: 30ms, detach_on_disconnect: {detach}, min_results: 0 }} }}
llms:
  fast: {{ model: openai/fast }}
  slow: {{ model: openai/slow }}
ensembles:
  e:
    members: [{{ llm: fast }}, {{ llm: slow }}]
    synthesizer: {{ llm: fast }}
"""


async def test_abandoned_member_is_a_shortfall_not_a_smaller_denominator():
    """`run completed` counts against the DISPATCHED roster. Counting outcomes instead reported a
    2-member run that lost one as a tidy `members_ok=1 members_total=1` — erasing the shortfall
    from the single line whose job is to summarize it."""
    catalog = _catalog(_DEADLINE_CONFIG.format(detach="false"))
    plan = resolve_plan(catalog, _ir("e"))
    deps = PipelineDeps(
        client=FakeLLM(delays={"slow": 0.2}), clock=ManualClock(), request_id="req-8"
    )
    with capture_logs() as logs:
        await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=2.0)

    assert next(e for e in logs if e["event"] == "fan-out started")["members_total"] == 2
    abandoned = next(e for e in logs if e["event"] == "member abandoned")
    assert abandoned["llm"] == "slow"
    assert abandoned["disposition"] == "aborted"
    done = next(e for e in logs if e["event"] == "run completed")
    assert done["members_ok"] == 1
    assert done["members_total"] == 2  # not 1: the roster, so the loss is visible


async def test_detached_member_logs_when_it_lands_after_the_client_left():
    """A detached member keeps running, finishes, and records a metric for its spend. Without a log
    line the spend is real but invisible, so the logs under-report cost versus the metrics DB."""
    catalog = _catalog(_DEADLINE_CONFIG.format(detach="true"))
    plan = resolve_plan(catalog, _ir("e"))
    deps = PipelineDeps(
        client=FakeLLM(delays={"slow": 0.1}), clock=ManualClock(), request_id="req-9"
    )
    with capture_logs() as logs:
        await asyncio.wait_for(collect(run_ensemble(plan, deps)), timeout=2.0)
        await asyncio.sleep(0.25)  # let the detached member land

    landed = next(e for e in logs if e["event"] == "detached member completed")
    assert landed["llm"] == "slow"
    assert landed["request_id"] == "req-9"
    assert "cost_usd" in landed
    assert "SECRET" not in repr(logs)


async def test_failed_synthesis_records_its_duration():
    """The duration fix reached _record_synth but not _record_synth_failure, leaving a synthesizer
    that timed out recording no duration — the same bug on the outcome most worth timing."""

    class _FailingSynth(FakeLLM):
        async def stream(self, spec):
            raise UpstreamError("synth exploded")
            yield  # pragma: no cover - makes this an async generator

    catalog = _catalog(CONFIG)
    plan = resolve_plan(catalog, _ir())
    recorder = _Recorder()
    deps = PipelineDeps(
        client=_FailingSynth(), clock=_TickingClock(), recorder=recorder, request_id="req-10"
    )
    events = [event async for event in run_ensemble(plan, deps)]

    assert any(type(e).__name__ == "PipelineFailed" for e in events)
    synth_metric = next(m for m in recorder.records if m.role == "synthesis")
    assert synth_metric.status == "error"
    assert synth_metric.duration_ms is not None
    assert synth_metric.duration_ms > 0
