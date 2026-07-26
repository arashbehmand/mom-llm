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
    assert set(catalog.ensembles) == {"bmom", "mom-code"}
    assert "gpt" in catalog.llms


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
