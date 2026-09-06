"""Capability cards: safe-minimum context, `advertise:` overrides, and the described panel."""

from __future__ import annotations

import yaml

from mom.config.capabilities import describe, ensemble_card
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config


def _catalog(text: str):
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def test_context_length_is_min_across_panel():
    catalog = _catalog(
        """
        version: 2
        llms:
          big: { model: openai/big, capabilities: { context_length: 200000 } }
          small: { model: openai/small, capabilities: { context_length: 8000 } }
        ensembles:
          e:
            members: [{ llm: big }, { llm: small }]
            synthesizer: { llm: big }
        """
    )
    card = ensemble_card("e", catalog.ensembles["e"], catalog)
    assert card.context_length == 8000  # the weakest member caps the safe advertised window


def test_advertise_overrides_the_computed_card():
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: a }
            advertise: { vision: false, context_length: 12345 }
        """
    )
    card = ensemble_card("e", catalog.ensembles["e"], catalog)
    assert card.supports_vision is False
    assert card.context_length == 12345


# ---- describe(): what a model picker shows -----------------------------------------------------
def _describe(text: str, name: str = "e") -> str:
    catalog = _catalog(text)
    return describe(ensemble_card(name, catalog.ensembles[name], catalog))


def test_description_carries_the_configured_text_and_then_the_panel():
    """An ensemble name says nothing about what answers it — the models do."""
    described = _describe(
        """
        version: 2
        llms:
          a: { model: openai/gpt-x }
          b: { model: anthropic/claude-y }
          syn: { model: openai/gpt-syn }
        ensembles:
          e:
            description: "A panel."
            members: [a, b]
            synthesizer: { llm: syn }
        """
    )
    assert described == (
        "A panel.\n\nFans out to 2 models — gpt-x, claude-y — then synthesizes with gpt-syn."
    )


def test_an_ensemble_with_no_description_still_gets_the_panel():
    described = _describe(
        """
        version: 2
        llms:
          a: { model: openai/gpt-x }
          syn: { model: openai/gpt-syn }
        ensembles:
          e: { members: [a], synthesizer: { llm: syn } }
        """
    )
    assert described == "Fans out to 1 model — gpt-x — then synthesizes with gpt-syn."


def test_a_passthrough_ensemble_describes_the_one_model_that_answers():
    described = _describe(
        """
        version: 2
        llms:
          a: { model: openai/gpt-x }
        ensembles:
          e: { strategy: passthrough, members: [a], synthesizer: { llm: a } }
        """
    )
    assert described == "Answers directly with gpt-x — no panel."


def test_one_model_seated_twice_is_named_once():
    """Two seats, one model (the same llm at two efforts, or an `as:` alias) — a list that repeats
    a name reads like a bug."""
    described = _describe(
        """
        version: 2
        llms:
          a: { model: openai/gpt-x, variants: { l: { params: { reasoning_effort: low } } } }
          syn: { model: openai/gpt-syn }
        ensembles:
          e:
            members: [a, a-l]
            synthesizer: { llm: syn }
        """
    )
    assert described.startswith("Fans out to 1 model — gpt-x — ")


def test_a_kitchen_sink_panel_counts_the_names_it_does_not_spell_out():
    """`members: all` can hold the whole catalog; a description nobody reads is worse than a
    shorter one."""
    llms = "\n".join(f"          m{i}: {{ model: openai/model-{i} }}" for i in range(20))
    described = _describe(
        "version: 2\nllms:\n"
        + llms
        + "\nensembles:\n  e: { members: all, synthesizer: { llm: m0 } }\n"
    )
    assert "Fans out to 20 models" in described
    assert "model-11, +8 more" in described  # 12 spelled out, the rest counted
    assert "model-12" not in described
