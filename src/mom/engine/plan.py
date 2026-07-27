"""Resolve a request + catalog into an execution plan (before any streaming starts).

Unknown models/LLMs and unusable effort values fail here as typed ``MomError``s, so they become
clean HTTP errors instead of mid-stream surprises after fan-out money is spent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mom.config.resolve import ResolvedCatalog, ResolvedEnsemble, ResolvedLlm
from mom.config.types import (
    EFFORT_OFF,
    EFFORT_PASSTHROUGH,
    EffortLevel,
    nearest_tier,
    normalize_effort_cell,
    parse_effort_level,
)
from mom.domain.cost import Pricing
from mom.domain.errors import InvalidRequestError, UnknownModelError
from mom.domain.ports import CallSpec
from mom.domain.prompt_caching import is_anthropic_family, openai_prompt_cache_key
from mom.domain.request import ChatRequestIR, Sampling
from mom.domain.synthesis import extract_concluding_instruction, messages_to_dicts
from mom.domain.tooling import (
    classify_turn,
    member_tool_summary,
    provider_supports_remote_mcp,
    tool_choice_to_wire,
    tools_to_wire,
)


SkipReason = Literal["passthrough", "tool_continuation"]


@dataclass(frozen=True, slots=True)
class PlannedMember:
    identity: str
    spec: CallSpec
    pricing: Pricing | None = None


@dataclass(frozen=True, slots=True)
class SynthPlan:
    llm_name: str
    model: str
    api: str
    proxy_url_env: str | None
    key_env_candidates: tuple[str, ...]
    retries: int
    params: dict[str, object]
    prompt: str | None
    timeout_seconds: float
    pricing: Pricing | None
    anthropic_cache_ttl: str | None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    ensemble: str
    strategy: str
    # Tool-call arbitration axis (orthogonal to `strategy`): arbitrate | vote | first.
    tool_strategy: str
    vote_threshold: int
    stream_profile: str
    skip_fanout: bool
    skip_reason: SkipReason | None
    show_work: str
    tier: EffortLevel | None
    client_messages: list[dict[str, object]]
    members: tuple[PlannedMember, ...]
    synth: SynthPlan
    instruction: str | None
    # Bounded fan-out: at most `max_concurrency` members call upstream at once (default cap 16,
    # so an unset config can't open an unbounded number of connections). `fanout_deadline` is an
    # optional overall wall-clock budget after which stragglers are abandoned.
    max_concurrency: int = 16
    fanout_deadline: float | None = None
    # Quorum: at least this many members must succeed (`ok`) before the pipeline synthesizes;
    # fewer fails with QuorumNotMet (502). 0 disables the check (the all-failed fallback runs).
    min_results: int = 1
    # When true, in-flight members are NOT cancelled if the request is torn down (client
    # disconnect); they finish and cache in the background so a retry of the turn hits cache.
    detach_on_disconnect: bool = False


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


def _pricing_of(llm: ResolvedLlm) -> Pricing | None:
    """Convert an LLM's config pricing block into the pure domain ``Pricing`` (or None)."""
    p = llm.pricing
    if p is None:
        return None
    return Pricing(
        input_per_1m=p.input_per_1m or 0.0,
        output_per_1m=p.output_per_1m or 0.0,
        reasoning_per_1m=p.reasoning_per_1m or 0.0,
        cache_read_per_1m=p.cache_read_per_1m or 0.0,
        cache_write_per_1m=p.cache_write_per_1m or 0.0,
    )


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


def _vision_ok(llm: ResolvedLlm) -> bool:
    """A member is vision-usable unless its config explicitly marks vision unsupported."""
    caps = llm.capabilities
    return caps is None or caps.vision is not False


def _estimate_input_tokens(messages: list[dict[str, object]]) -> int:
    """A cheap ~chars/4 heuristic over a member's wire messages (for the input-overflow guard).

    Deliberately provider-naive: the guard only needs a rough magnitude to compare against an
    llm's ``max_input_tokens``, and staying pure keeps plan resolution free of any tokenizer.
    """
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(
                len(part["text"])
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    return chars // 4


def _apply_search(
    params: dict[str, object], llm: ResolvedLlm, web_search: bool
) -> dict[str, object]:
    """Merge the LLM's provider search params when the client requested web search."""
    if not web_search or llm.search is None:
        return params
    search: Mapping[str, Any] = llm.search
    out = dict(params)
    for key, value in search.items():
        existing = out.get("tools")
        if key == "tools" and isinstance(value, list) and isinstance(existing, list):
            out["tools"] = [*existing, *value]
        else:
            out[key] = value
    return out


def _apply_sampling(params: dict[str, object], sampling: Sampling) -> None:
    """Apply the client's generation controls to the synthesizer (its client-visible output).

    Client-sent values win over the synthesizer's configured defaults — honoring them is the
    whole point (v1 dropped them silently). Advisory members keep their own tuned params, so a
    client ``max_tokens`` never truncates a member's internal reasoning.
    """
    if sampling.temperature is not None:
        params["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        params["top_p"] = sampling.top_p
    if sampling.max_output_tokens is not None:
        params["max_tokens"] = sampling.max_output_tokens
    if sampling.stop:
        params["stop"] = list(sampling.stop)
    if sampling.seed is not None:
        params["seed"] = sampling.seed


def resolve_plan(catalog: ResolvedCatalog, ir: ChatRequestIR) -> ExecutionPlan:
    """Resolve a chat request against the catalog into a ready-to-run plan."""
    ensemble = catalog.ensembles.get(ir.model)
    if ensemble is None:
        raise UnknownModelError(f"unknown model {ir.model!r}")

    messages, instruction = extract_concluding_instruction(ir.messages)
    client_messages = messages_to_dicts(messages)
    tier = _resolve_tier(ensemble, ir.effort)
    retries = catalog.config.defaults.call.retries
    fanout = catalog.config.defaults.fanout

    is_relay = ensemble.tools_continuation == "relay" and classify_turn(messages) == "relay"
    skip_fanout = ensemble.strategy == "passthrough" or is_relay
    skip_reason: SkipReason | None = None
    if ensemble.strategy == "passthrough":
        skip_reason = "passthrough"
    elif is_relay:
        skip_reason = "tool_continuation"

    # vote/first make members the deciders: they get the real tool schemas so they can propose
    # structured calls. In arbitrate mode members stay advisory — a schema-free summary only (they
    # cannot invoke tools; the synthesizer owns the client-visible call).
    members_propose = ensemble.tool_strategy in ("vote", "first")
    member_messages = client_messages
    if (
        ir.tools
        and ensemble.member_tool_context == "summary"
        and not skip_fanout
        and not members_propose
    ):
        member_messages = [
            *client_messages,
            {"role": "system", "content": member_tool_summary(ir.tools)},
        ]

    members: list[PlannedMember] = []
    if not skip_fanout:
        estimated_input_tokens = _estimate_input_tokens(member_messages)
        for member in ensemble.members_at(tier):
            llm = catalog.llms[member.llm]
            # Image requests propagate only to vision-capable members (others drop out).
            if ir.has_images and not _vision_ok(llm):
                continue
            # Per-llm input budget: a member whose estimated input exceeds its max_input_tokens
            # either drops out of the panel (on_input_overflow: skip, the default) or fails the
            # whole request (reject) — resolved pre-flight so 'reject' is a clean 400 before spend.
            if llm.max_input_tokens is not None and estimated_input_tokens > llm.max_input_tokens:
                if ensemble.on_input_overflow == "reject":
                    raise InvalidRequestError(
                        f"input (~{estimated_input_tokens} tokens) exceeds "
                        f"max_input_tokens={llm.max_input_tokens} for llm {member.llm!r}"
                    )
                continue
            params = _member_params(llm, dict(member.effort_by_tier), tier, ir.effort)
            params = _apply_search(params, llm, ir.web_search)
            if members_propose and ir.tools:
                params["tools"] = tools_to_wire(ir.tools)
                params["tool_choice"] = tool_choice_to_wire(ir.tool_choice)
                if ir.parallel_tool_calls is not None:
                    params["parallel_tool_calls"] = ir.parallel_tool_calls
            members.append(
                PlannedMember(
                    identity=member.identity,
                    spec=CallSpec(
                        llm_name=member.identity,
                        model=llm.model,
                        messages=member_messages,
                        params=params,
                        api=llm.api,
                        proxy_url_env=llm.proxy_url_env,
                        key_env_candidates=llm.key_env_candidates,
                        retries=retries,
                        timeout_seconds=_timeout_seconds(catalog, llm),
                    ),
                    pricing=_pricing_of(llm),
                )
            )

    syn = ensemble.synthesizer
    syn_llm = catalog.llms[syn.llm]
    synth_params = _member_params(syn_llm, dict(syn.effort_by_tier), tier, ir.effort)
    synth_params = _apply_search(synth_params, syn_llm, ir.web_search)
    _apply_sampling(synth_params, ir.sampling)
    if ir.response_format is not None:
        synth_params["response_format"] = ir.response_format
    wire_tools = tools_to_wire(ir.tools) if ir.tools else []
    if ir.mcp_tools:
        # Forward opaque `type: mcp` blocks only to a synthesizer whose provider accepts them;
        # otherwise keep the clean 400 (before any fan-out spend) — MCP runs client-side instead.
        if provider_supports_remote_mcp(syn_llm.model, syn_llm.api):
            wire_tools = [*wire_tools, *ir.mcp_tools]
        else:
            raise InvalidRequestError(
                "MCP tools are not supported by this ensemble's synthesizer; execute MCP "
                "client-side and pass plain function tools instead"
            )
    if wire_tools:
        synth_params["tools"] = wire_tools
    if ir.tools:
        synth_params["tool_choice"] = tool_choice_to_wire(ir.tool_choice)
        if ir.parallel_tool_calls is not None:
            synth_params["parallel_tool_calls"] = ir.parallel_tool_calls
    # Provider prompt caching: Anthropic reads cache_control breakpoints (injected at stream time
    # in the pipeline); OpenAI/xAI just need a stable prompt_cache_key for prefix-cache affinity.
    pcache = catalog.config.defaults.provider_cache
    provider = syn_llm.model.split("/", 1)[0].lower() if "/" in syn_llm.model else ""
    anthropic_cache_ttl: str | None = None
    if is_anthropic_family(syn_llm.model) and pcache.anthropic.enabled:
        anthropic_cache_ttl = "1h" if pcache.anthropic.ttl.total_seconds() > 300 else "5m"
    elif provider in {"openai", "azure", "xai"} and pcache.openai.prompt_cache_key == "auto":
        synth_params["prompt_cache_key"] = openai_prompt_cache_key(client_messages)
    prompt_name = syn.search_prompt if (ir.web_search and syn.search_prompt) else syn.prompt
    synth = SynthPlan(
        llm_name=syn.llm,
        model=syn_llm.model,
        api=syn_llm.api,
        proxy_url_env=syn_llm.proxy_url_env,
        key_env_candidates=syn_llm.key_env_candidates,
        retries=retries,
        params=synth_params,
        prompt=catalog.config.prompts.get(prompt_name) if prompt_name else None,
        timeout_seconds=_timeout_seconds(catalog, syn_llm),
        pricing=_pricing_of(syn_llm),
        anthropic_cache_ttl=anthropic_cache_ttl,
    )

    return ExecutionPlan(
        ensemble=ir.model,
        strategy=ensemble.strategy,
        tool_strategy=ensemble.tool_strategy,
        vote_threshold=ensemble.vote_threshold,
        stream_profile=ensemble.stream_profile,
        skip_fanout=skip_fanout,
        skip_reason=skip_reason,
        show_work=ensemble.show_work,
        tier=tier,
        client_messages=client_messages,
        members=tuple(members),
        synth=synth,
        instruction=instruction,
        max_concurrency=fanout.max_concurrency or 16,
        fanout_deadline=fanout.deadline.total_seconds() if fanout.deadline else None,
        min_results=fanout.min_results,
        detach_on_disconnect=fanout.detach_on_disconnect,
    )
