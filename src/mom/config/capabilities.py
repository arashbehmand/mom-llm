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
    #: The ensemble's own ``description:``, verbatim. ``describe()`` is what a wire surface shows.
    description: str | None
    supports_tools: bool
    supports_vision: bool
    supports_reasoning: bool
    supports_web_search: bool
    supports_streaming: bool
    effort_levels: tuple[str, ...]
    context_length: int | None
    max_output_tokens: int | None
    #: Member *identities* (the `as:`/llm names a `<<SYSTEM>>` directive addresses).
    members: tuple[str, ...]
    synthesizer: str
    strategy: str = "synthesize"
    #: The provider model each member runs, aligned with ``members``; the same for the synthesizer.
    #: A name in a model list means nothing on its own — these are what the panel actually is.
    member_models: tuple[str, ...] = ()
    synthesizer_model: str = ""


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
        strategy=ensemble.strategy,
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
        member_models=tuple(llm.model for llm in member_llms),
        synthesizer_model=synth_llm.model,
    )
    return _apply_advertise(card, ensemble.advertise)


#: How many model names a panel line spells out before it starts counting. A `members: all` panel
#: can hold the whole catalog, and a description nobody can read is worse than a shorter one.
_PANEL_NAMES_SHOWN = 12


def _model_name(model: str) -> str:
    """``openrouter/z-ai/glm-5.3`` -> ``glm-5.3`` — the part a human reads as the model."""
    return model.rsplit("/", 1)[-1]


def _panel_line(card: ModelCard) -> str:
    """One sentence naming the models behind the ensemble name."""
    synth = _model_name(card.synthesizer_model)
    if card.strategy == "passthrough" or not card.member_models:
        return f"Answers directly with {synth} — no panel."
    # Deduplicated: the same model can hold several seats (one llm at two efforts, or an `as:`
    # alias), and a list that repeats a name four times reads like a bug.
    names = list(dict.fromkeys(_model_name(model) for model in card.member_models))
    # One name over the cap is shorter spelled out than counted, and reads better besides.
    hidden = 0 if len(names) <= _PANEL_NAMES_SHOWN + 1 else len(names) - _PANEL_NAMES_SHOWN
    shown = ", ".join(names if not hidden else names[:_PANEL_NAMES_SHOWN])
    if hidden:
        shown = f"{shown}, +{hidden} more"
    plural = "model" if len(names) == 1 else "models"
    return f"Fans out to {len(names)} {plural} — {shown} — then synthesizes with {synth}."


def describe(card: ModelCard) -> str:
    """The description a client shows in a model picker: what the ensemble is, and what is in it.

    A model list is where a human chooses, and an ensemble name (`emom`, `bmom`) tells them nothing
    about the panel behind it — so the configured ``description:`` is followed by the models that
    actually answer. An ensemble with no ``description:`` still gets the panel line, which is the
    half a client cannot reconstruct from the id alone.
    """
    configured = (card.description or "").strip()
    panel = _panel_line(card)
    return f"{configured}\n\n{panel}" if configured else panel


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
