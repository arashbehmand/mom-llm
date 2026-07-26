"""Engine: plan resolution + the fan-out -> synthesize pipeline (with a fake provider)."""

from __future__ import annotations

from textwrap import dedent

import pytest
import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import MomError, UnknownModelError
from mom.domain.events import (
    AnswerDelta,
    Completed,
    FanoutSkipped,
    FanoutStarted,
    MemberCompleted,
)
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.testing import FakeLLM, ManualClock


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
