"""Pure projections from mom's resolved types onto the MCP result schemas.

No I/O and no container: everything here is a value in, a value out, so the tool bodies in
``server.py`` stay thin enough to read as a list of what each tool does.
"""

from __future__ import annotations

from typing import Any, Literal

from mom.api.mcp.schemas import (
    CacheStats,
    ConsultResult,
    EnsembleInfo,
    ErrorInfo,
    InFlightRun,
    LlmInfo,
    MemberInfo,
    MemberReport,
    PricingInfo,
    RecentRun,
    RunCall,
    RunMemberReport,
    UsageGroup,
    UsageInfo,
    UsageReport,
)
from mom.config.capabilities import ModelCard
from mom.config.resolve import ResolvedCatalog, ResolvedEnsemble, ResolvedLlm
from mom.domain.errors import MomError
from mom.domain.ports import RunSummary
from mom.domain.results import EnsembleResult, ModelOutcome, Usage


def llm_info(llm: ResolvedLlm, *, catalogue_pricing: dict[str, float] | None) -> LlmInfo:
    """One catalog llm, with config pricing preferred over litellm's map (that is billing order:
    ``compute_cost`` uses a declared ``pricing:`` block and falls back to litellm)."""
    caps = llm.capabilities
    pricing = _pricing_from_config(llm.pricing)
    source: Literal["config", "litellm", "unknown"] = "config"
    if pricing is None and catalogue_pricing is not None:
        pricing = PricingInfo(**catalogue_pricing)
        source = "litellm"
    elif pricing is None:
        source = "unknown"
    return LlmInfo(
        name=llm.name,
        model=llm.model,
        api=llm.api,
        vision=getattr(caps, "vision", None),
        tools=getattr(caps, "tools", None),
        reasoning=getattr(caps, "reasoning", None),
        web_search=llm.search is not None,
        context_length=getattr(caps, "context_length", None),
        max_output_tokens=getattr(caps, "max_output_tokens", None),
        max_input_tokens=llm.max_input_tokens,
        timeout_seconds=llm.timeout.total_seconds() if llm.timeout is not None else None,
        pricing=pricing,
        pricing_source=source,
    )


def _pricing_from_config(pricing: Any | None) -> PricingInfo | None:
    if pricing is None:
        return None
    fields = PricingInfo.model_fields
    declared = {name: getattr(pricing, name, None) for name in fields}
    return PricingInfo(**declared) if any(v is not None for v in declared.values()) else None


def ensemble_info(
    ensemble: ResolvedEnsemble, card: ModelCard, catalog: ResolvedCatalog
) -> EnsembleInfo:
    """The advertised card (shared with ``/v1/models``) plus the panel composition an agent needs
    to decide whether to call this ensemble or assemble its own."""

    def member(identity: str, llm_name: str, effort: Any) -> MemberInfo:
        return MemberInfo(
            identity=identity,
            llm=llm_name,
            model=catalog.llms[llm_name].model,
            effort_by_tier={tier.label: value for tier, value in effort.items()},
        )

    return EnsembleInfo(
        name=ensemble.name,
        description=ensemble.description,
        strategy=ensemble.strategy,
        members=[member(m.identity, m.llm, m.effort_by_tier) for m in ensemble.members],
        synthesizer=member(
            ensemble.synthesizer.llm, ensemble.synthesizer.llm, ensemble.synthesizer.effort_by_tier
        ),
        effort_tiers=[t.label for t in ensemble.effort_tiers or ()],
        default_tier=ensemble.default_tier.label if ensemble.default_tier else None,
        show_work=ensemble.show_work,
        tool_strategy=ensemble.tool_strategy,
        supports_tools=card.supports_tools,
        supports_vision=card.supports_vision,
        supports_reasoning=card.supports_reasoning,
        supports_web_search=card.supports_web_search,
        context_length=card.context_length,
        max_output_tokens=card.max_output_tokens,
    )


def member_report(outcome: ModelOutcome, *, include_answers: bool) -> MemberReport:
    """One member's outcome. ``error`` is the client-safe half only — ``error_kind`` and
    ``error_detail`` are operator-facing and stay in metrics/logs, as on every wire surface."""
    return MemberReport(
        identity=outcome.identity,
        llm=outcome.llm,
        model=outcome.model,
        status=outcome.status,
        cost_usd=outcome.cost_usd,
        duration_ms=outcome.duration_ms,
        cached=outcome.cached,
        finish_reason=outcome.finish_reason,
        error=outcome.error,
        answer=outcome.content if include_answers else None,
        reasoning=outcome.reasoning if include_answers else None,
    )


def abandoned_report(identity: str, model: str) -> MemberReport:
    """A member the fan-out deadline passed by. It produced no outcome, so it would otherwise
    vanish from the panel entirely — leaving a consult silently reporting fewer members than it
    asked for.

    ``llm`` repeats the identity because that is what the engine reports for every member
    (``ModelOutcome.llm`` is the identity too, which is also what the metrics ``llm`` column
    holds) — matching ``member_report`` matters more than the field's name.
    """
    return MemberReport(
        identity=identity,
        llm=identity,
        model=model,
        status="abandoned",
        error="no result before the fan-out deadline",
    )


def usage_info(usage: Usage) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )


def consult_success(
    result: EnsembleResult,
    *,
    ensemble: str,
    request_id: str,
    coalesced: bool,
    progress_url: str | None,
    members: list[MemberReport],
    include_answers: bool,
) -> ConsultResult:
    """A completed run: text, or a tool call the caller is expected to execute."""
    tool_calls = [dict(call) for call in result.tool_calls]
    ended_in_tools = bool(tool_calls) or result.finish_reason == "tool_calls"
    return ConsultResult(
        status="tool_calls" if ended_in_tools else "ok",
        answer="" if ended_in_tools else result.text,
        reasoning=result.reasoning if include_answers else "",
        finish_reason=result.finish_reason,
        tool_calls=tool_calls,
        ensemble=ensemble,
        request_id=request_id,
        coalesced=coalesced,
        progress_url=progress_url,
        total_cost_usd=result.total_cost_usd,
        usage=usage_info(result.usage),
        members=members,
    )


def consult_failure(
    exc: MomError,
    *,
    ensemble: str,
    request_id: str,
    coalesced: bool,
    progress_url: str | None,
    members: list[MemberReport],
) -> ConsultResult:
    """A run that died upstream. Members and cost are whatever completed first — spend that
    really happened is reported even though there is no answer to show for it."""
    return ConsultResult(
        status="failed",
        finish_reason="error",
        error=ErrorInfo(code=exc.code, message=exc.safe_message, http_status=exc.http_status),
        ensemble=ensemble,
        request_id=request_id,
        coalesced=coalesced,
        progress_url=progress_url,
        total_cost_usd=sum(m.cost_usd for m in members),
        members=members,
    )


def in_flight_run(summary: RunSummary) -> InFlightRun:
    return InFlightRun(
        request_id=summary.request_id,
        ensemble=summary.ensemble,
        state=summary.state,
        started_at=summary.started_at,
        updated_at=summary.updated_at,
        members_total=summary.members_total,
        members=[
            RunMemberReport(
                identity=m.identity,
                model=m.model,
                status=m.status,
                duration_ms=m.duration_ms,
                cost_usd=m.cost_usd,
            )
            for m in summary.members
        ],
        cost_usd=summary.cost_usd,
        finish_reason=summary.finish_reason,
        detail=summary.detail,
    )


def recent_run(row: dict[str, Any]) -> RecentRun:
    return RecentRun(
        request_id=str(row.get("request_id", "")),
        ensemble=str(row.get("ensemble") or ""),
        started_ts=float(row.get("started_ts") or 0.0),
        last_ts=float(row.get("last_ts") or 0.0),
        calls=int(row.get("calls") or 0),
        cost_usd=float(row.get("cost_usd") or 0.0),
        failures=int(row.get("failures") or 0),
        cache_hits=int(row.get("cache_hits") or 0),
    )


def run_call(row: dict[str, Any]) -> RunCall:
    return RunCall(
        ts=float(row.get("ts") or 0.0),
        ensemble=str(row.get("ensemble") or ""),
        llm=str(row.get("llm") or ""),
        model=row.get("model"),
        role=str(row.get("role") or ""),
        turn_type=str(row.get("turn_type") or ""),
        status=str(row.get("status") or ""),
        cache_hit=bool(row.get("cache_hit")),
        cost_usd=row.get("cost_usd"),
        duration_ms=row.get("duration_ms"),
        prompt_tokens=row.get("prompt_tokens"),
        completion_tokens=row.get("completion_tokens"),
        total_tokens=row.get("total_tokens"),
        finish_reason=row.get("finish_reason"),
        error=row.get("error"),
        attempts=int(row.get("attempts") or 1),
    )


def usage_report(
    totals: dict[str, Any],
    *,
    window_days: float | None,
    ensemble: str | None,
    by_ensemble: list[dict[str, Any]],
    by_llm: list[dict[str, Any]],
    savings: float,
) -> UsageReport:
    return UsageReport(
        window_days=window_days,
        ensemble=ensemble,
        calls=int(totals.get("calls") or 0),
        cost_usd=float(totals.get("cost_usd") or 0.0),
        prompt_tokens=int(totals.get("prompt_tokens") or 0),
        completion_tokens=int(totals.get("completion_tokens") or 0),
        reasoning_tokens=int(totals.get("reasoning_tokens") or 0),
        cached_prompt_tokens=int(totals.get("cached_prompt_tokens") or 0),
        errors=int(totals.get("errors") or 0),
        timeouts=int(totals.get("timeouts") or 0),
        cache_hits=int(totals.get("cache_hits") or 0),
        billable_calls=int(totals.get("billable_calls") or 0),
        estimated_cache_savings_usd=savings,
        by_ensemble=[_usage_group(row, "ensemble") for row in by_ensemble],
        by_llm=[_usage_group(row, "member") for row in by_llm],
    )


def _usage_group(row: dict[str, Any], key_field: str) -> UsageGroup:
    return UsageGroup(
        key=str(row.get(key_field) or ""),
        calls=int(row.get("calls") or 0),
        cost_usd=float(row.get("cost_usd") or 0.0),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        errors=int(row.get("errors") or 0),
        cache_hits=int(row.get("cache_hits") or 0),
    )


def cache_stats(raw: dict[str, int] | None) -> CacheStats:
    if raw is None:
        return CacheStats(enabled=False)
    return CacheStats(
        enabled=True,
        entries=int(raw.get("entries") or 0),
        bytes=int(raw.get("bytes") or 0),
        hits=int(raw.get("hits") or 0),
    )
