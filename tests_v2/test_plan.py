"""Plan resolution: effort-tier math, passthrough, per-LLM timeout, search merge, tool params.

These drive the public ``resolve_plan`` against small catalogs — the same path the API uses — so
the effort ladder and provider-param assembly are covered as behavior, not via private helpers.
"""

from __future__ import annotations

from textwrap import dedent

import pytest
import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.config.types import EffortLevel
from mom.domain.errors import InvalidRequestError
from mom.domain.request import ChatRequestIR, MessageIR, ToolSpec
from mom.engine.plan import resolve_plan


CONFIG = """
version: 2
llms:
  fast:  { model: openai/fast }
  deep:  { model: openai/deep, timeout: 90s }
  synth: { model: openai/synth }
  seeker:
    model: openai/seeker
    params: { tools: [{ type: web_search_preview }] }
    search: { tools: [{ type: extra }], web_search_options: { search_context_size: high } }
ensembles:
  tiered:
    effort_tiers: [low, high]
    default_tier: low
    members:
      - { llm: fast, effort: { low: "off", high: high } }
      - { llm: deep, effort: pass }
    synthesizer: { llm: synth, prompt: p, effort: { low: low, high: xhigh } }
  search_ens:
    members: [{ llm: seeker }]
    synthesizer: { llm: synth, prompt: p }
prompts:
  p: "synthesize"
"""


def _catalog():
    return resolve_catalog(Config.model_validate(yaml.safe_load(dedent(CONFIG))))


def _ir(model: str = "tiered", **overrides) -> ChatRequestIR:
    base = {"model": model, "messages": (MessageIR(role="user", content="hi"),)}
    base.update(overrides)
    return ChatRequestIR(**base)


def _member(plan, identity: str):
    return next(m for m in plan.members if m.identity == identity)


def test_default_tier_applies_when_client_sends_no_effort():
    plan = resolve_plan(_catalog(), _ir(effort=None))

    assert plan.tier is EffortLevel.LOW  # the ensemble default_tier
    # "off" cell -> no reasoning param; "pass" cell with no client effort -> also nothing.
    assert "reasoning_effort" not in _member(plan, "fast").spec.params
    assert "reasoning_effort" not in _member(plan, "deep").spec.params
    # Synthesizer's low-tier cell is a concrete level.
    assert plan.synth.params["reasoning_effort"] == "low"
    # Per-LLM timeout overrides the global default.
    assert _member(plan, "deep").spec.timeout_seconds == 90.0


def test_client_effort_selects_tier_and_passthrough_relays_it():
    plan = resolve_plan(_catalog(), _ir(effort="high"))

    assert plan.tier is EffortLevel.HIGH
    assert _member(plan, "fast").spec.params["reasoning_effort"] == "high"
    # "pass" relays the client's requested (normalized) effort to this member.
    assert _member(plan, "deep").spec.params["reasoning_effort"] == "high"
    assert plan.synth.params["reasoning_effort"] == "xhigh"


def test_client_effort_snaps_to_nearest_defined_tier():
    # "minimal" is below both tiers -> nearest is the lowest defined tier (low).
    plan = resolve_plan(_catalog(), _ir(effort="minimal"))
    assert plan.tier is EffortLevel.LOW


def test_invalid_client_effort_is_rejected():
    with pytest.raises(InvalidRequestError, match="invalid reasoning effort"):
        resolve_plan(_catalog(), _ir(effort="lightspeed"))


def test_web_search_merges_provider_tools_into_member_params():
    plan = resolve_plan(_catalog(), _ir(model="search_ens", web_search=True))

    params = _member(plan, "seeker").spec.params
    # The provider's search `tools` list is appended to the member's existing `tools`.
    assert params["tools"] == [{"type": "web_search_preview"}, {"type": "extra"}]
    # Non-list search keys are merged in wholesale.
    assert params["web_search_options"] == {"search_context_size": "high"}


def test_web_search_ignored_when_not_requested():
    plan = resolve_plan(_catalog(), _ir(model="search_ens", web_search=False))
    params = _member(plan, "seeker").spec.params
    assert params["tools"] == [{"type": "web_search_preview"}]
    assert "web_search_options" not in params


def test_tools_and_parallel_tool_calls_forwarded_to_synthesizer():
    ir = _ir(
        tools=(ToolSpec(name="lookup", description="d", parameters={"type": "object"}),),
        parallel_tool_calls=True,
    )
    plan = resolve_plan(_catalog(), ir)

    assert plan.synth.params["parallel_tool_calls"] is True
    assert plan.synth.params["tool_choice"] == "auto"
    assert isinstance(plan.synth.params["tools"], list)
    assert plan.synth.params["tools"][0]["function"]["name"] == "lookup"
