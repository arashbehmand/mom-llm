"""Compute an ensemble's advertised capability card from its resolved members.

Aggregation follows what actually happens at runtime, and is configurable per LLM via the
``capabilities`` override:

- vision  = ANY member can see (image requests filter to those; others drop out)
- tools / structured output = the SYNTHESIZER (it owns the final tool calls / structured answer)
- reasoning = the ensemble exposes effort tiers OR shows work
- web_search = ANY member (or the synthesizer) declares a search block
- context_length = MIN across the panel (the weakest member caps the safe window)
- max_output_tokens = the synthesizer's declared limit

An ensemble's ``advertise:`` block overrides any computed field with an explicit value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mom.config.resolve import ResolvedCatalog, ResolvedEnsemble, ResolvedLlm


@dataclass(frozen=True, slots=True)
class ModelCard:
    id: str
    description: str | None
    supports_tools: bool
    supports_vision: bool
    supports_reasoning: bool
    supports_web_search: bool
    supports_streaming: bool
    effort_levels: tuple[str, ...]
    context_length: int | None
    max_output_tokens: int | None
    members: tuple[str, ...]
    synthesizer: str


def _vision(llm: ResolvedLlm) -> bool:
    caps = llm.capabilities
    return caps is None or caps.vision is not False


def _flag(llm: ResolvedLlm, attr: str) -> bool | None:
    caps = llm.capabilities
    return getattr(caps, attr) if caps is not None else None


def ensemble_card(name: str, ensemble: ResolvedEnsemble, catalog: ResolvedCatalog) -> ModelCard:
    member_llms = [catalog.llms[m.llm] for m in ensemble.members]
    synth_llm = catalog.llms[ensemble.synthesizer.llm]

    supports_vision = (
        any(_vision(llm) for llm in member_llms) if member_llms else _vision(synth_llm)
    )
    supports_tools = _flag(synth_llm, "tools") is not False
    supports_reasoning = ensemble.effort_tiers is not None or ensemble.show_work != "off"
    supports_web_search = any(llm.search is not None for llm in [*member_llms, synth_llm])
    effort_levels = tuple(t.label for t in ensemble.effort_tiers) if ensemble.effort_tiers else ()

    synth_caps = synth_llm.capabilities
    # Safe aggregate: the panel can only accept as much context as its smallest member.
    context_length = _min_context([*member_llms, synth_llm])
    max_output_tokens = synth_caps.max_output_tokens if synth_caps is not None else None

    card = ModelCard(
        id=name,
        description=ensemble.description,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        supports_web_search=supports_web_search,
        supports_streaming=True,
        effort_levels=effort_levels,
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        members=tuple(m.identity for m in ensemble.members),
        synthesizer=ensemble.synthesizer.llm,
    )
    return _apply_advertise(card, ensemble.advertise)


def _min_context(llms: list[ResolvedLlm]) -> int | None:
    """The smallest declared context window across the panel (the safe advertised limit)."""
    values = [
        llm.capabilities.context_length
        for llm in llms
        if llm.capabilities is not None and llm.capabilities.context_length is not None
    ]
    return min(values) if values else None


# Ensemble ``advertise:`` keys -> ModelCard fields. An explicit value overrides the computed
# default (the mode strings any/all/min/sum reproduce the defaults, so they are left as no-ops).
_ADVERTISE_FIELDS: dict[str, str] = {
    "vision": "supports_vision",
    "tools": "supports_tools",
    "reasoning": "supports_reasoning",
    "web_search": "supports_web_search",
    "context_length": "context_length",
    "max_output_tokens": "max_output_tokens",
}


def _apply_advertise(card: ModelCard, advertise: Mapping[str, Any]) -> ModelCard:
    overrides: dict[str, Any] = {}
    for key, value in advertise.items():
        field = _ADVERTISE_FIELDS.get(key)
        if field is None:
            continue
        if (field.startswith("supports_") and isinstance(value, bool)) or (
            field in ("context_length", "max_output_tokens")
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            overrides[field] = value
    return replace(card, **overrides) if overrides else card
