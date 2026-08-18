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
  synth2: { model: openai/synth2 }
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
      - { llm: seeker, as: extra, effort: { low: skip, high: high } }
    synthesizer: { llm: synth, prompt: p, effort: { low: low, high: xhigh } }
  search_ens:
    members: [{ llm: seeker }]
    synthesizer: { llm: synth, prompt: p }
  filtered:
    members:
      - { llm: fast }
      - { llm: deep, as: deep2 }
      - { llm: seeker }
    synthesizer: { llm: synth, prompt: p }
  passthru:
    strategy: passthrough
    members: [fast]
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


# -------------------------------------------------------------------------------------------
# <<SYSTEM>> directives, end to end through resolve_plan.
# -------------------------------------------------------------------------------------------


def _ir_with_block(block: str, *, model: str = "filtered", **overrides) -> ChatRequestIR:
    return _ir(model=model, messages=(MessageIR(role="user", content=f"hi {block}"),), **overrides)


def test_exclude_removes_the_member_and_strips_the_block_from_client_messages():
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>exclude: deep2<</SYSTEM>>"))
    identities = {m.identity for m in plan.members}
    assert identities == {"fast", "seeker"}
    assert "SYSTEM" not in plan.client_messages[0]["content"]
    assert plan.client_messages[0]["content"] == "hi"


def test_exclude_by_as_alias_identity_works():
    # "deep2" is the `as:` alias; the underlying llm name "deep" is not itself a valid identity
    # to exclude on this ensemble (two members share the llm — see the next test).
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>exclude: deep2<</SYSTEM>>"))
    assert "deep2" not in {m.identity for m in plan.members}


def test_exclude_unknown_identity_raises_400_listing_the_roster():
    with pytest.raises(InvalidRequestError, match="fast, seeker") as excinfo:
        resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>exclude: nonexistent<</SYSTEM>>"))
    assert "nonexistent" in str(excinfo.value)


def test_only_restricts_the_panel_to_named_members():
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>only: fast<</SYSTEM>>"))
    assert {m.identity for m in plan.members} == {"fast"}


def test_exclude_a_tier_skipped_member_is_a_harmless_no_op():
    # "extra" (effort: skip at the low/default tier) is validated against the FULL roster
    # (ensemble.members), not members_at(tier) — so excluding it must not 400 as "unknown member"
    # even though it's already absent from the low-tier panel for an unrelated (static) reason.
    plan = resolve_plan(
        _catalog(), _ir_with_block("<<SYSTEM>>exclude: extra<</SYSTEM>>", model="tiered")
    )
    assert {m.identity for m in plan.members} == {"fast", "deep"}


def test_quorum_below_min_results_after_exclude_is_a_pre_flight_400():
    text = dedent(CONFIG) + "\ndefaults:\n  fanout: { min_results: 3 }\n"
    config = resolve_catalog(Config.model_validate(yaml.safe_load(text)))
    block = "<<SYSTEM>>exclude: deep2<</SYSTEM>>"
    with pytest.raises(InvalidRequestError, match="min_results"):
        resolve_plan(config, _ir_with_block(block, model="filtered"))


def test_exclude_all_members_is_a_400():
    with pytest.raises(InvalidRequestError):
        resolve_plan(
            _catalog(), _ir_with_block("<<SYSTEM>>exclude: fast, deep2, seeker<</SYSTEM>>")
        )


def test_directives_on_a_tiered_ensemble_dont_affect_normal_requests():
    # No directive present -> zero behavior change (the fast-path in _select_members).
    plan = resolve_plan(_catalog(), _ir(model="tiered"))
    assert {m.identity for m in plan.members} == {"fast", "deep"}


def test_exclude_on_a_passthrough_ensemble_is_a_400():
    block = "<<SYSTEM>>exclude: fast<</SYSTEM>>"
    with pytest.raises(InvalidRequestError, match="passthrough"):
        resolve_plan(_catalog(), _ir_with_block(block, model="passthru"))


def test_show_work_directive_overrides_the_ensemble_default():
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>show_work: inline<</SYSTEM>>"))
    assert plan.show_work == "inline"


def test_invalid_show_work_directive_is_a_400():
    with pytest.raises(InvalidRequestError, match="show_work"):
        resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>show_work: verbose<</SYSTEM>>"))


def test_synth_directive_retargets_the_synthesizer():
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>synth: synth2<</SYSTEM>>"))
    assert plan.synth.llm_name == "synth2"
    assert plan.synth.model == "openai/synth2"


def test_unknown_synth_directive_is_a_400():
    with pytest.raises(InvalidRequestError, match="synth"):
        resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>synth: nonexistent<</SYSTEM>>"))


def test_instruction_alone_threads_through_unaffected_by_directives():
    plan = resolve_plan(_catalog(), _ir_with_block("<<SYSTEM>>Answer tersely.<</SYSTEM>>"))
    assert plan.instruction == "Answer tersely."
    assert {m.identity for m in plan.members} == {"fast", "deep2", "seeker"}


def test_legacy_marker_still_works_end_to_end():
    block = "<<CONCLUDING-INSTRUCTION>>Reply in Farsi.<</CONCLUDING-INSTRUCTION>>"
    plan = resolve_plan(_catalog(), _ir_with_block(block))
    assert plan.instruction == "Reply in Farsi."
