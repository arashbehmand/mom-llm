"""v2 config: schema validation, extends resolution, and the effort matrix."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mom.config.loader import load_config, parse_config
from mom.config.resolve import ConfigError, resolve_catalog
from mom.config.types import EffortLevel


EXAMPLE = Path(__file__).parent.parent / "config.example.yaml"


def _resolve(yaml_text: str):
    return resolve_catalog(parse_config(_yaml(yaml_text)))


def _yaml(text: str):
    import yaml

    return yaml.safe_load(dedent(text))


# ---- happy path: the shipped example ---------------------------------------------------------
def test_example_config_loads_and_resolves():
    catalog = load_config(EXAMPLE)
    assert set(catalog.ensembles) == {"bmom", "mom-code", "mom-debug"}
    assert "gpt" in catalog.llms


def test_mcp_surface_is_off_unless_asked_for():
    """A second network surface on the same port is opt-in, and old configs stay valid."""
    minimal = _resolve(
        """
        version: 2
        llms: { a: { model: x/y } }
        ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }
        """
    )
    assert minimal.config.server.mcp.enabled is False

    enabled = _resolve(
        """
        version: 2
        server: { mcp: { enabled: true } }
        llms: { a: { model: x/y } }
        ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }
        """
    )
    assert enabled.config.server.mcp.enabled is True


def test_example_mom_debug_covers_every_llm():
    catalog = load_config(EXAMPLE)
    debug_members = {m.identity for m in catalog.ensembles["mom-debug"].members}
    assert debug_members == set(catalog.llms)


def test_example_effort_matrix():
    catalog = load_config(EXAMPLE)
    bmom = catalog.ensembles["bmom"]
    assert bmom.effort_tiers == (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH)
    assert bmom.default_tier == EffortLevel.MEDIUM

    by_id = {m.identity: m for m in bmom.members}
    # positional list normalized l/m/h -> canonical labels
    assert by_id["gpt"].effort_by_tier == {
        EffortLevel.LOW: "low",
        EffortLevel.MEDIUM: "medium",
        EffortLevel.HIGH: "high",
    }
    # scalar 'pass' broadcasts to every tier
    assert set(by_id["gemini"].effort_by_tier.values()) == {"pass"}
    # 'skip' excludes flash at the low tier
    assert by_id["flash"].effort_by_tier[EffortLevel.LOW] == "skip"


def test_members_at_drops_skipped_tier():
    catalog = load_config(EXAMPLE)
    bmom = catalog.ensembles["bmom"]
    low_ids = {m.identity for m in bmom.members_at(EffortLevel.LOW)}
    assert "flash" not in low_ids
    assert "gpt" in low_ids
    high_ids = {m.identity for m in bmom.members_at(EffortLevel.HIGH)}
    assert "flash" in high_ids


def test_passthrough_ensemble():
    catalog = load_config(EXAMPLE)
    code = catalog.ensembles["mom-code"]
    assert code.strategy == "passthrough"
    assert len(code.members) == 1


def test_gemini_key_env_inference():
    catalog = load_config(EXAMPLE)
    gemini = catalog.llms["gemini"]
    assert gemini.api_key_env == "GEMINI_API_KEY"
    assert gemini.key_env_candidates == ("GEMINI_API_KEY", "GOOGLE_API_KEY")


# ---- extends resolution ----------------------------------------------------------------------
def test_extends_deep_merges_params_and_inherits_model():
    catalog = _resolve(
        """
        version: 2
        llms:
          base: { model: openai/gpt-x, params: { reasoning_effort: low, temperature: 0.2 } }
          child: { extends: base, params: { reasoning_effort: high } }
        ensembles:
          e:
            members: [{ llm: base }, { llm: child }]
            synthesizer: { llm: base }
        """
    )
    child = catalog.llms["child"]
    assert child.model == "openai/gpt-x"  # inherited
    assert child.params["reasoning_effort"] == "high"  # overridden
    assert child.params["temperature"] == 0.2  # deep-merged from parent


def test_extends_null_deletes_inherited_param():
    catalog = _resolve(
        """
        version: 2
        llms:
          base: { model: openai/gpt-x, params: { reasoning_effort: low } }
          child: { extends: base, params: { reasoning_effort: null } }
        ensembles:
          e: { members: [{ llm: child }], synthesizer: { llm: child } }
        """
    )
    assert "reasoning_effort" not in catalog.llms["child"].params


def test_cyclic_extends_is_rejected():
    with pytest.raises(ConfigError, match="cyclic"):
        _resolve(
            """
            version: 2
            llms:
              a: { extends: b, model: x/y }
              b: { extends: a }
            ensembles:
              e: { members: [{ llm: a }], synthesizer: { llm: a } }
            """
        )


# ---- variants: an effort-family expands into `<parent>-<suffix>` siblings --------------------


def test_variants_expand_into_sibling_llms():
    catalog = _resolve(
        """
        version: 2
        llms:
          base:
            model: openai/gpt-x
            variants:
              m: { params: { reasoning_effort: medium } }
              p: { api: responses, params: { reasoning: { effort: max } } }
        ensembles:
          e:
            members: [base, base-m, base-p]
            synthesizer: { llm: base }
        """
    )
    assert catalog.llms["base-m"].model == "openai/gpt-x"  # inherited
    assert catalog.llms["base-m"].params == {"reasoning_effort": "medium"}
    assert catalog.llms["base-p"].api == "responses"  # per-variant override
    assert catalog.llms["base-p"].params == {"reasoning": {"effort": "max"}}
    assert catalog.llms["base"].params == {}  # the parent itself is untouched


def test_variant_inherits_capability_fields():
    # A variant is the same model at a different effort — capability fields like `search` travel
    # with it (search is request-triggered anyway, so inheriting it just means the variant CAN
    # search when the client asks, exactly like its parent).
    catalog = _resolve(
        """
        version: 2
        llms:
          base:
            model: openrouter/x/y
            search: { extra_body: { plugins: [{ id: web }] } }
            variants:
              l: { params: { reasoning: { effort: low } } }
        ensembles:
          e: { members: [base, base-l], synthesizer: { llm: base } }
        """
    )
    assert catalog.llms["base"].search is not None
    assert catalog.llms["base-l"].search == catalog.llms["base"].search


def test_variant_name_collision_is_rejected():
    with pytest.raises(ConfigError, match="collides"):
        _resolve(
            """
            version: 2
            llms:
              base: { model: x/y, variants: { m: { params: {} } } }
              base-m: { model: x/other }
            ensembles:
              e: { members: [base], synthesizer: { llm: base } }
            """
        )


def test_bare_string_members_are_shorthand_for_llm_mapping():
    catalog = _resolve(
        """
        version: 2
        llms:
          a: { model: x/a }
          b: { model: x/b }
        ensembles:
          e:
            members: [a, { llm: b, effort: pass }]
            synthesizer: { llm: a }
        """
    )
    members = {m.identity: m for m in catalog.ensembles["e"].members}
    assert set(members) == {"a", "b"}


def test_members_all_expands_to_every_llm_including_variants():
    catalog = _resolve(
        """
        version: 2
        llms:
          a: { model: x/a, variants: { l: { params: { reasoning_effort: low } } } }
          b: { model: x/b }
        ensembles:
          e:
            members: all
            synthesizer: { llm: a }
        """
    )
    assert {m.identity for m in catalog.ensembles["e"].members} == {"a", "a-l", "b"}


def test_members_all_exclude_opts_specific_llms_out():
    catalog = _resolve(
        """
        version: 2
        llms:
          a: { model: x/a }
          b: { model: x/b }
          c: { model: x/c }
        ensembles:
          e:
            members: { all: true, exclude: [b] }
            synthesizer: { llm: a }
        """
    )
    assert {m.identity for m in catalog.ensembles["e"].members} == {"a", "c"}


def test_members_all_exclude_of_unknown_llm_is_rejected():
    with pytest.raises(ConfigError, match="unknown llm"):
        _resolve(
            """
            version: 2
            llms: { a: { model: x/y } }
            ensembles:
              e: { members: { all: true, exclude: [ghost] }, synthesizer: { llm: a } }
            """
        )


# ---- roster patches: members_exclude / members_include ---------------------------------------
# The list-in-a-layered-config problem: `members:` REPLACES wholesale when one layer merges over
# another, so before these an override could only drop a model from a panel by restating the whole
# roster — which then silently stops tracking the roster it was copied from.
ROSTER = """
version: 2
llms:
  a: { model: x/a }
  b: { model: x/b }
  c: { model: x/c }
ensembles:
  e:
    effort_tiers: [high, max]
    default_tier: max
    members:
      - { llm: a, effort: [high, max] }
      - { llm: b, effort: high }
      - { llm: a, as: a2, effort: high }
    synthesizer: { llm: a }
"""


def _roster(patch: str = "") -> list[str]:
    catalog = _resolve(ROSTER + patch)
    return [m.identity for m in catalog.ensembles["e"].members]


def test_members_exclude_drops_a_seat_by_identity():
    assert _roster("    members_exclude: [b, a2]\n") == ["a"]


def test_members_exclude_takes_a_bare_name_for_the_one_model_case():
    assert _roster("    members_exclude: b\n") == ["a", "a2"]


def test_members_exclude_of_a_name_that_is_not_seated_is_a_no_op():
    """Deliberately not an error, unlike the same typo in `members: {all, exclude}`: an exclusion
    lives in a different file from the roster it patches (usually an untracked override) and has
    to survive the base config dropping that model on its own."""
    assert _roster("    members_exclude: [ghost]\n") == ["a", "b", "a2"]


def test_members_include_appends_a_model_that_is_not_on_the_roster():
    catalog = _resolve(ROSTER + "    members_include: [{ llm: c, effort: high }]\n")
    members = catalog.ensembles["e"].members
    assert [m.identity for m in members] == ["a", "b", "a2", "c"]
    assert members[-1].effort_by_tier[EffortLevel.MAX] == "high"


def test_members_include_redeclares_a_seated_member_in_place():
    """Not a second seat, and not appended: keeping its position is what makes this the way a
    layer retunes one member's effort without restating the roster."""
    catalog = _resolve(ROSTER + "    members_include: [{ llm: b, effort: max }]\n")
    members = catalog.ensembles["e"].members
    assert [m.identity for m in members] == ["a", "b", "a2"]
    assert members[1].effort_by_tier[EffortLevel.HIGH] == "max"


def test_members_include_wins_over_members_exclude_on_the_same_name():
    patch = "    members_exclude: [b]\n    members_include: [b]\n"
    assert _roster(patch) == ["a", "a2", "b"]


def test_members_exclude_composes_with_the_all_shorthand():
    catalog = _resolve(
        """
        version: 2
        llms:
          a: { model: x/a }
          b: { model: x/b }
        ensembles:
          e:
            members: all
            members_exclude: [b]
            synthesizer: { llm: a }
        """
    )
    assert [m.identity for m in catalog.ensembles["e"].members] == ["a"]


def test_members_include_of_an_unknown_llm_is_rejected():
    """The other half of the asymmetry: an exclusion that matches nothing is already satisfied,
    but a member that cannot be built is a broken panel — the same error `members:` would give."""
    with pytest.raises(ConfigError, match="unknown llm"):
        _resolve(ROSTER + "    members_include: [ghost]\n")


def test_excluding_every_member_is_rejected():
    with pytest.raises(ConfigError, match="no members"):
        _resolve(ROSTER + "    members_exclude: [a, b, a2]\n")


def test_members_include_cannot_grow_a_passthrough_ensemble():
    with pytest.raises(ConfigError, match="at most one"):
        _resolve(
            """
            version: 2
            llms:
              a: { model: x/a }
              b: { model: x/b }
            ensembles:
              e:
                strategy: passthrough
                members: [a]
                members_include: [b]
                synthesizer: { llm: a }
            """
        )


def test_members_all_rejected_for_passthrough():
    with pytest.raises(ConfigError, match="at most one member"):
        _resolve(
            """
            version: 2
            llms: { a: { model: x/y } }
            ensembles:
              e: { strategy: passthrough, members: all, synthesizer: { llm: a } }
            """
        )


# ---- validation errors -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        (
            """
            version: 2
            server: { cors: { origins: ['*'], allow_credentials: true } }
            llms: { a: { model: x/y } }
            ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }
            """,
            "wildcard",
        ),
        (
            """
            version: 2
            llms: { a: { model: x/y, params: { model: sneaky } } }
            ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }
            """,
            "reserved",
        ),
        (
            """
            version: 2
            llms: { 'a:b': { model: x/y } }
            ensembles: { e: { members: [{ llm: 'a:b' }], synthesizer: { llm: 'a:b' } } }
            """,
            "reserved characters",
        ),
    ],
)
def test_schema_validation_errors(yaml_text: str, match: str):
    with pytest.raises(ConfigError, match=match):
        parse_config(_yaml(yaml_text))


def test_effort_list_length_must_match_tiers():
    with pytest.raises(ConfigError, match="effort list"):
        _resolve(
            """
            version: 2
            llms: { a: { model: x/y } }
            ensembles:
              e:
                effort_tiers: [low, high]
                default_tier: low
                members: [{ llm: a, effort: [l, m, h] }]
                synthesizer: { llm: a }
            """
        )


def test_unknown_member_llm_is_rejected():
    with pytest.raises(ConfigError, match="unknown llm"):
        _resolve(
            """
            version: 2
            llms: { a: { model: x/y } }
            ensembles: { e: { members: [{ llm: ghost }], synthesizer: { llm: a } } }
            """
        )


def test_default_tier_requires_effort_tiers():
    with pytest.raises(ConfigError, match="default_tier"):
        parse_config(
            _yaml(
                """
                version: 2
                llms: { a: { model: x/y } }
                ensembles:
                  e: { default_tier: low, members: [{ llm: a }], synthesizer: { llm: a } }
                """
            )
        )


def test_same_llm_twice_needs_distinct_as():
    # Two members of the same llm need an `as` to be distinct.
    with pytest.raises(ConfigError, match="duplicate member"):
        parse_config(
            _yaml(
                """
                version: 2
                llms: { a: { model: x/y } }
                ensembles:
                  e: { members: [{ llm: a }, { llm: a }], synthesizer: { llm: a } }
                """
            )
        )
    # ...but with an `as` alias they coexist.
    catalog = _resolve(
        """
        version: 2
        llms: { a: { model: x/y } }
        ensembles:
          e: { members: [{ llm: a }, { llm: a, as: second }], synthesizer: { llm: a } }
        """
    )
    assert {m.identity for m in catalog.ensembles["e"].members} == {"a", "second"}
