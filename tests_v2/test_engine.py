"""Engine: plan resolution + the fan-out -> synthesize pipeline (with a fake provider)."""

from __future__ import annotations

from textwrap import dedent

import pytest
import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import UnknownModelError
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
