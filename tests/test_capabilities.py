"""Capability cards: safe-minimum context and `advertise:` overrides."""

from __future__ import annotations

import yaml

from mom.config.capabilities import ensemble_card
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
