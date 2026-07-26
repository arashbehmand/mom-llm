"""Resolve a request + catalog into an execution plan (before any streaming starts).

Unknown models/LLMs and unusable effort values fail here as typed ``MomError``s, so they become
clean HTTP errors instead of mid-stream surprises after fan-out money is spent.
"""

from __future__ import annotations

from dataclasses import dataclass

from mom.config.resolve import ResolvedCatalog, ResolvedEnsemble, ResolvedLlm
from mom.config.types import (
    EFFORT_OFF,
    EFFORT_PASSTHROUGH,
    EffortLevel,
    nearest_tier,
    normalize_effort_cell,
    parse_effort_level,
)
from mom.domain.errors import InvalidRequestError, UnknownModelError
from mom.domain.ports import CallSpec
from mom.domain.request import ChatRequestIR
from mom.domain.synthesis import extract_concluding_instruction, messages_to_dicts


@dataclass(frozen=True, slots=True)
class PlannedMember:
    identity: str
    spec: CallSpec


@dataclass(frozen=True, slots=True)
class SynthPlan:
    llm_name: str
    model: str
    api: str
    proxy_url_env: str | None
    params: dict[str, object]
    prompt: str | None
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    ensemble: str
    strategy: str
    show_work: str
    tier: EffortLevel | None
    client_messages: list[dict[str, object]]
    members: tuple[PlannedMember, ...]
    synth: SynthPlan
    instruction: str | None


def _effort_param(token: str, client_effort: str | None) -> dict[str, object]:
    """Map one resolved effort cell to provider params (provider-naive; refined by effort_map)."""
    if token == EFFORT_OFF:
        return {}
    if token == EFFORT_PASSTHROUGH:
        if not client_effort:
            return {}
        return {"reasoning_effort": normalize_effort_cell(client_effort)}
    return {"reasoning_effort": token}


def _resolve_tier(ensemble: ResolvedEnsemble, client_effort: str | None) -> EffortLevel | None:
    if ensemble.effort_tiers is None:
        return None
    if client_effort is None:
        return ensemble.default_tier
    try:
        requested = parse_effort_level(client_effort)
    except ValueError as exc:
        raise InvalidRequestError(f"invalid reasoning effort {client_effort!r}") from exc
    return nearest_tier(requested, list(ensemble.effort_tiers))


def _timeout_seconds(catalog: ResolvedCatalog, llm: ResolvedLlm) -> float:
    if llm.timeout is not None:
        return float(llm.timeout.total_seconds())
    return float(catalog.config.defaults.call.timeout.total_seconds())


def _member_params(
    llm: ResolvedLlm,
    effort_by_tier: dict[EffortLevel, str],
    tier: EffortLevel | None,
    client_effort: str | None,
) -> dict[str, object]:
    params = dict(llm.params)
    if tier is not None:
        token = effort_by_tier.get(tier, EFFORT_OFF)
        params.update(_effort_param(token, client_effort))
    return params


def resolve_plan(catalog: ResolvedCatalog, ir: ChatRequestIR) -> ExecutionPlan:
    """Resolve a chat request against the catalog into a ready-to-run plan."""
    ensemble = catalog.ensembles.get(ir.model)
    if ensemble is None:
        raise UnknownModelError(f"unknown model {ir.model!r}")

    messages, instruction = extract_concluding_instruction(ir.messages)
    client_messages = messages_to_dicts(messages)
    tier = _resolve_tier(ensemble, ir.effort)

    members: list[PlannedMember] = []
    for member in ensemble.members_at(tier):
        llm = catalog.llms[member.llm]
        members.append(
            PlannedMember(
                identity=member.identity,
                spec=CallSpec(
                    llm_name=member.identity,
                    model=llm.model,
                    messages=client_messages,
                    params=_member_params(llm, dict(member.effort_by_tier), tier, ir.effort),
                    api=llm.api,
                    proxy_url_env=llm.proxy_url_env,
                    timeout_seconds=_timeout_seconds(catalog, llm),
                ),
            )
        )

    syn = ensemble.synthesizer
    syn_llm = catalog.llms[syn.llm]
    synth = SynthPlan(
        llm_name=syn.llm,
        model=syn_llm.model,
        api=syn_llm.api,
        proxy_url_env=syn_llm.proxy_url_env,
        params=_member_params(syn_llm, dict(syn.effort_by_tier), tier, ir.effort),
        prompt=catalog.config.prompts.get(syn.prompt) if syn.prompt else None,
        timeout_seconds=_timeout_seconds(catalog, syn_llm),
    )

    return ExecutionPlan(
        ensemble=ir.model,
        strategy=ensemble.strategy,
        show_work=ensemble.show_work,
        tier=tier,
        client_messages=client_messages,
        members=tuple(members),
        synth=synth,
        instruction=instruction,
    )
