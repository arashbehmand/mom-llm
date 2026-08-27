"""The six tools, and the factory both transports build from.

``build_mcp_server`` takes an accessor rather than a container because over HTTP the app is
constructed before the lifespan builds the container — the mount has to reach ``app.state`` at
call time, while ``mom mcp`` already holds one. Same tool definitions either way, so the two
transports cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from mom.api.mcp import projections
from mom.api.mcp.consult import run_consult
from mom.api.mcp.schemas import (
    CacheStats,
    ConsultResult,
    EnsembleInfo,
    LlmInfo,
    RecentRun,
    RunCall,
    RunsReport,
    UsageReport,
)
from mom.config.capabilities import ensemble_card
from mom.domain.ports import RunIndex, RunSummary
from mom.runtime.container import Container
from mom.runtime.logging import get_logger


logger = get_logger("mom.api.mcp")

INSTRUCTIONS = """\
mom runs a panel of LLMs and synthesizes one answer from their perspectives.

Use `consult` for a question worth more than one model's opinion — a design call, a review, a
judgement where being wrong is expensive. Name a configured `ensemble` (see `list_ensembles`), or
assemble a panel for this question alone by passing `panel` (catalog llm names from `list_llms`)
plus a `synthesizer`. An inline panel exists only for that call.

The remaining tools are read-only views of the gateway: `runs` (what is running and what ran),
`usage` (spend), `cache_stats`. Purging and config changes are deliberately not available here.\
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

# Ceiling on how many recent runs one `runs` call may materialize.
_MAX_RUNS = 200


def build_mcp_server(get_container: Callable[[], Container | None]) -> MCPServer[Any]:
    """Define the MoM tool surface against a container accessor.

    The accessor may return None: over HTTP the app exists before its lifespan builds the
    container, so "not ready yet" is a real state rather than a bug to assert away.
    """
    mcp: MCPServer[Any] = MCPServer(
        name="mom",
        title="MoM — Mixture of Models",
        instructions=INSTRUCTIONS,
    )

    def current_container() -> Container:
        found = get_container()
        if found is None:  # pragma: no cover - defensive; the gate returns 503 before this
            raise ToolError("gateway is not ready")
        return found

    @mcp.tool(
        title="List models",
        description=(
            "Every model in the catalog (bases and variants) with its provider model string, "
            "capabilities and pricing. These names are what an inline `consult` panel accepts."
        ),
        annotations=_READ_ONLY,
    )
    def list_llms() -> list[LlmInfo]:
        catalog = current_container().catalog
        return [
            projections.llm_info(llm, catalogue_pricing=_catalogue_pricing(llm.model))
            for llm in catalog.llms.values()
        ]

    @mcp.tool(
        title="List ensembles",
        description=(
            "The configured panels: members, effort tiers, and the synthesizer that combines "
            "their answers. Pass a name to `consult` as `ensemble`."
        ),
        annotations=_READ_ONLY,
    )
    def list_ensembles() -> list[EnsembleInfo]:
        catalog = current_container().catalog
        return [
            projections.ensemble_info(ensemble, ensemble_card(name, ensemble, catalog), catalog)
            for name, ensemble in catalog.ensembles.items()
        ]

    @mcp.tool(
        title="Consult a panel",
        description=(
            "Ask a panel of models a question and get back one synthesized answer, with the "
            "per-member cost breakdown. Either name a configured `ensemble`, or pass `panel` "
            "(llm names) plus a `synthesizer` to assemble one for this call only. Progress is "
            "reported per member while it runs."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=True
        ),
    )
    async def consult(
        prompt: Annotated[str, "The question to put to the panel."],
        ctx: Context[Any, Any],
        ensemble: Annotated[str | None, "A configured ensemble name."] = None,
        panel: Annotated[list[str] | None, "Catalog llm names for a one-off panel."] = None,
        synthesizer: Annotated[str | None, "Catalog llm that combines an inline panel."] = None,
        effort: Annotated[str | None, "Effort tier, for ensembles that declare tiers."] = None,
        system: Annotated[str | None, "Optional system message for the panel."] = None,
        tools: Annotated[
            list[dict[str, Any]] | None,
            "OpenAI-shaped tool definitions. The panel may answer with a tool call instead of "
            "text; executing it is the caller's job (no continuation over MCP).",
        ] = None,
        include_member_answers: Annotated[
            bool, "Include each member's own answer, not just its status and cost."
        ] = False,
    ) -> Annotated[CallToolResult, ConsultResult]:
        result = await run_consult(
            current_container(),
            ctx,
            ensemble=ensemble,
            panel=panel,
            synthesizer=synthesizer,
            prompt=prompt,
            system=system,
            effort=effort,
            tools=tools,
            include_member_answers=include_member_answers,
            base_url=_base_url(ctx),
        )
        return _consult_tool_result(result)

    @mcp.tool(
        title="Inspect runs",
        description=(
            "What is running now and what ran recently, with per-member status and cost. Pass a "
            "`request_id` for one run's per-call detail."
        ),
        annotations=_READ_ONLY,
    )
    async def runs(
        request_id: Annotated[str | None, "Limit to one run."] = None,
        limit: Annotated[int, "How many recent runs to list (1-200)."] = 20,
    ) -> RunsReport:
        # Clamped, not trusted: SQLite reads a negative LIMIT as "no limit", so an unbounded
        # value would group and serialize every run the ledger has ever recorded.
        limit = max(1, min(limit, _MAX_RUNS))
        current = current_container()
        bus = current.bus
        # A bus that keeps no index (Redis pub/sub) can't answer "what is running" at all; the
        # report says so rather than returning an empty list that reads as "nothing is".
        indexed = isinstance(bus, RunIndex)
        summaries: list[RunSummary] = bus.snapshot(request_id) if isinstance(bus, RunIndex) else []
        in_flight = [projections.in_flight_run(s) for s in summaries if s.in_flight]
        reader = current.metrics_reader
        recent: list[RecentRun] = []
        calls: list[RunCall] | None = None
        if reader is not None:
            if request_id is not None:
                calls = [projections.run_call(row) for row in await reader.run_calls(request_id)]
            else:
                recent = [
                    projections.recent_run(row) for row in await reader.recent_runs(limit=limit)
                ]
        return RunsReport(
            in_flight=in_flight,
            recent=recent,
            calls=calls,
            in_flight_visibility="process" if indexed else "none",
        )

    @mcp.tool(
        title="Usage and spend",
        description=(
            "Aggregate spend and call counts over a window, grouped by ensemble and by llm — "
            "the same figures `mom metrics usage` reports."
        ),
        annotations=_READ_ONLY,
    )
    async def usage(
        days: Annotated[float, "Window in days; 0 or less means all time."] = 7.0,
        ensemble: Annotated[str | None, "Limit to one ensemble."] = None,
    ) -> UsageReport:
        current = current_container()
        reader = current.metrics_reader
        if reader is None:
            return UsageReport(note="metrics are not enabled on this gateway")
        # The container's clock, not wall time: metric rows are stamped from the same port, so a
        # deployment (or test) running on an injected clock would otherwise window against a
        # different timeline than the rows it is filtering.
        start = current.clock.now() - days * 86400 if days > 0 else None
        totals = await reader.aggregate(start=start, ensemble=ensemble)
        by_ensemble = await reader.aggregate_by("ensemble", start=start, ensemble=ensemble)
        by_llm = await reader.aggregate_by("member", start=start, ensemble=ensemble)
        savings = await reader.estimated_cache_savings(start=start, ensemble=ensemble)
        report = projections.usage_report(
            dict(totals),
            window_days=days if days > 0 else None,
            ensemble=ensemble,
            by_ensemble=[dict(row) for row in by_ensemble],
            by_llm=[dict(row) for row in by_llm],
            savings=savings,
        )
        # The recorder drops metrics rather than block a call, so these are a floor. GET /health
        # reports how many were dropped.
        report.note = "a lower bound: metrics are recorded off the hot path and may be dropped"
        return report

    @mcp.tool(
        title="Cache statistics",
        description="Response-cache entry count, size on disk, and cumulative hits.",
        annotations=_READ_ONLY,
    )
    async def cache_stats() -> CacheStats:
        store = current_container().cache_store
        return projections.cache_stats(await store.stats() if store is not None else None)

    return mcp


def _catalogue_pricing(model: str) -> dict[str, float] | None:
    """litellm's list price for a model, or None. Best-effort: a listing must not fail because
    the pricing catalog could not be consulted."""
    try:
        from mom.adapters.litellm_client import pricing_for

        return pricing_for(model)
    except Exception:  # pragma: no cover - defensive around a third-party catalog read
        logger.debug("pricing lookup failed", model=model, exc_info=True)
        return None


def _base_url(ctx: Context[Any, Any]) -> str | None:
    """The gateway's own base URL as this caller reached it, for the progress link.

    None over stdio, where there is no request to read a host from — the link then depends on
    `server.public_url` being configured, and is omitted when it is not.
    """
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError):
        return None
    url = getattr(request, "url", None)
    if url is None:
        return None
    # Built from the URL's own parts rather than by trimming the path off a string: a gateway
    # reached at a host that happens to end in "mcp" would have its hostname eaten instead.
    return f"{url.scheme}://{url.netloc}"


def _consult_tool_result(result: ConsultResult) -> CallToolResult:
    """Wrap the envelope for MCP: structured for the caller, one text block for a text-only one.

    A failed run is ``isError`` *with* the payload attached rather than a protocol error: the
    model should be able to read what failed and what it cost, and decide whether to retry or
    pick a different panel.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=_text_summary(result))],
        structured_content=result.model_dump(mode="json"),
        is_error=result.status == "failed",
    )


def _text_summary(result: ConsultResult) -> str:
    if result.status == "failed":
        error = result.error
        return f"{error.code}: {error.message}" if error else "the panel failed"
    if result.status == "tool_calls":
        names = _tool_call_names(result.tool_calls)
        return f"[panel ended in {len(result.tool_calls)} tool call(s): {', '.join(names)}]"
    return result.answer


def _tool_call_names(tool_calls: Sequence[dict[str, Any]]) -> list[str]:
    names = []
    for call in tool_calls:
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        names.append(str(name or "unnamed"))
    return names
