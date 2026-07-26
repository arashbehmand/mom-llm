"""Compute an ensemble's advertised capability card from its resolved members.

Aggregation follows what actually happens at runtime, and is configurable per LLM via the
``capabilities`` override:

- vision  = ANY member can see (image requests filter to those; others drop out)
- tools / structured output = the SYNTHESIZER (it owns the final tool calls / structured answer)
- reasoning = the ensemble exposes effort tiers OR shows work
- web_search = ANY member (or the synthesizer) declares a search block
- context_length / max_output_tokens = the synthesizer's declared limits (config override)
"""

from __future__ import annotations

from dataclasses import dataclass

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
    context_length = synth_caps.context_length if synth_caps is not None else None
    max_output_tokens = synth_caps.max_output_tokens if synth_caps is not None else None

    return ModelCard(
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
