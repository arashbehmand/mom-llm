"""The tools, and the factory both transports build from.

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

from mom.adapters.litellm_client import capabilities_for, pricing_for
from mom.api.mcp import projections
from mom.api.mcp.consult import PanelRequest, execute_consult, prepare_consult
from mom.api.mcp.jobs import (
    MAX_WAIT_SECONDS,
    JobRecord,
    JobRegistry,
    strip_reasoning,
    without_member_answers,
)
from mom.api.mcp.schemas import (
    AnswersReport,
    CacheStats,
    ConsultResult,
    EnsembleInfo,
    JobResult,
    JobsReport,
    JobStatus,
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

A panel can take minutes. When it may outlast your tool-call timeout, or you want to keep working
while it runs, call `submit` instead (same arguments): it returns a `job_id` at once. Then `status`
shows which members have answered, `result` returns the answer once it is ready (optionally
waiting for it), and `cancel` stops a job you no longer need. `status` with no `job_id` lists jobs.

The remaining tools are read-only views of the gateway: `runs` (what is running and what ran),
`usage` (spend), `cache_stats`. Purging and config changes are deliberately not available here.\
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

# Ceiling on how many recent runs one `runs` call may materialize.
_MAX_RUNS = 200

_SPENDS = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

# The `<<SYSTEM>>` directives, as typed arguments. A block in the prompt still works and merges
# with these (`domain/directives.merged`); an agent reading a tool schema should not have to know
# the header format exists.
Only = Annotated[list[str] | None, "Run ONLY these members of the ensemble (identities)."]
Exclude = Annotated[list[str] | None, "Drop these members from the panel for this run."]
Include = Annotated[
    list[str] | None,
    "Add these to the panel: a member an effort tier dropped, or any catalog llm not on it.",
]
Synth = Annotated[str | None, "Synthesize with this llm instead of the ensemble's own."]
Instruction = Annotated[
    str | None,
    "An instruction for the synthesizer alone, kept out of what the members are asked.",
]
ShowWork = Annotated[str | None, "off | inline | native — whether the answer carries the panel's."]
Dedupe = Annotated[bool | None, "Attach to an identical run already in flight, or refuse to."]
CacheSynth = Annotated[
    bool | None, "Keep this synthesis for the next identical run, or force a fresh one."
]


def build_mcp_server(
    get_container: Callable[[], Container | None], *, jobs: JobRegistry | None = None
) -> MCPServer[Any]:
    """Define the MoM tool surface against a container accessor.

    The accessor may return None: over HTTP the app exists before its lifespan builds the
    container, so "not ready yet" is a real state rather than a bug to assert away.

    ``jobs`` holds the background consults; a transport that owns a lifetime passes its own so it
    can close it (``JobRegistry.aclose``) when it stops.
    """
    jobs = jobs if jobs is not None else JobRegistry()
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
            projections.llm_info(
                llm,
                catalogue_pricing=_from_catalogue(pricing_for, llm.model),
                catalogue_capabilities=_from_catalogue(capabilities_for, llm.model),
            )
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
        annotations=_SPENDS,
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
        only: Only = None,
        exclude: Exclude = None,
        include: Include = None,
        synth: Synth = None,
        instruction: Instruction = None,
        show_work: ShowWork = None,
        dedupe: Dedupe = None,
        cache_synth: CacheSynth = None,
        include_member_answers: Annotated[
            bool,
            "Put each member's own answer in this result. Off by default: they are recorded "
            "whatever you pass, and `answers` fetches them when you want them.",
        ] = False,
    ) -> Annotated[CallToolResult, ConsultResult]:
        container = current_container()
        prepared = prepare_consult(
            container,
            ensemble=ensemble,
            panel=panel,
            synthesizer=synthesizer,
            prompt=prompt,
            system=system,
            effort=effort,
            tools=tools,
            panel_request=PanelRequest(
                only, exclude, include, synth, instruction, show_work, dedupe, cache_synth
            ),
        )
        result = await execute_consult(
            container,
            prepared,
            ctx,
            request_id=container.ids.new_id("req"),
            base_url=_base_url(ctx),
        )
        await jobs.record(
            container, result, members_total=len(prepared.plan.members), prompt=prompt
        )
        return _consult_tool_result(
            result if include_member_answers else without_member_answers(result)
        )

    @mcp.tool(
        title="Submit a consult",
        description=(
            "Start a consult in the background and get its `job_id` back at once; same arguments "
            "as `consult`. Use it when the panel may run longer than your tool-call timeout, or "
            "to keep working meanwhile. Follow it with `status`, `result` and `cancel`. A bad "
            "argument is rejected here, before anything runs."
        ),
        annotations=_SPENDS,
    )
    async def submit(
        prompt: Annotated[str, "The question to put to the panel."],
        ctx: Context[Any, Any],
        ensemble: Annotated[str | None, "A configured ensemble name."] = None,
        panel: Annotated[list[str] | None, "Catalog llm names for a one-off panel."] = None,
        synthesizer: Annotated[str | None, "Catalog llm that combines an inline panel."] = None,
        effort: Annotated[str | None, "Effort tier, for ensembles that declare tiers."] = None,
        system: Annotated[str | None, "Optional system message for the panel."] = None,
        tools: Annotated[
            list[dict[str, Any]] | None,
            "OpenAI-shaped tool definitions, as for `consult`.",
        ] = None,
        only: Only = None,
        exclude: Exclude = None,
        include: Include = None,
        synth: Synth = None,
        instruction: Instruction = None,
        show_work: ShowWork = None,
        dedupe: Dedupe = None,
        cache_synth: CacheSynth = None,
    ) -> JobStatus:
        container = current_container()
        prepared = prepare_consult(
            container,
            ensemble=ensemble,
            panel=panel,
            synthesizer=synthesizer,
            prompt=prompt,
            system=system,
            effort=effort,
            tools=tools,
            panel_request=PanelRequest(
                only, exclude, include, synth, instruction, show_work, dedupe, cache_synth
            ),
        )
        return await jobs.submit(container, prepared, prompt=prompt, base_url=_base_url(ctx))

    @mcp.tool(
        title="Job status",
        description=(
            "Where a submitted consult stands: which members have answered, what they cost so "
            "far, and whether synthesis has started. No answer — `result` returns that. Without "
            "a `job_id`, lists the jobs on this machine, newest first."
        ),
        annotations=_READ_ONLY,
    )
    async def status(
        job_id: Annotated[str | None, "The id `submit` returned."] = None,
        limit: Annotated[int, "How many jobs to list when no job_id is given (1-200)."] = 20,
    ) -> JobsReport:
        if job_id is not None:
            return JobsReport(jobs=[await jobs.status(job_id)])
        return JobsReport(jobs=await jobs.recent(max(1, min(limit, _MAX_RUNS))))

    @mcp.tool(
        title="Job result",
        description=(
            "A submitted consult's answer: the same envelope `consult` returns, once the job has "
            "completed, without the members' own answers (`answers` has those). While it is "
            "still running you get its status instead. `wait_seconds` "
            f"(up to {MAX_WAIT_SECONDS:g}) waits for it to finish first — keep that below your "
            "own tool-call timeout."
        ),
        annotations=_READ_ONLY,
    )
    async def result(
        job_id: Annotated[str, "The id `submit` returned."],
        wait_seconds: Annotated[float, "How long to wait for the job to finish, if running."] = 0,
    ) -> Annotated[CallToolResult, JobResult]:
        return _job_tool_result(await jobs.result(job_id, wait_seconds=wait_seconds))

    @mcp.tool(
        title="What each member said",
        description=(
            "Each panel member's own answer for one run — a job, or a `consult` this gateway ran "
            "(every run is kept for a day). Use it when the synthesis is not enough: to see who "
            "disagreed, or what the members that did answer said while one of them hangs. Name a "
            "`member` for just that one."
        ),
        annotations=_READ_ONLY,
    )
    async def answers(
        job_id: Annotated[str, "A job id, or the `request_id` a consult returned."],
        member: Annotated[str | None, "One member's identity, instead of all of them."] = None,
        reasoning: Annotated[
            bool, "Include each member's reasoning as well as its answer. Often long."
        ] = False,
    ) -> AnswersReport:
        report = await jobs.answers(job_id, member=member)
        if reasoning:
            return report
        return report.model_copy(update={"members": strip_reasoning(report.members)})

    @mcp.tool(
        title="Cancel a job",
        description=(
            "Stop a submitted consult, including its members' calls still in flight, so it "
            "spends nothing more. A job that has already finished is left as it is."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def cancel(job_id: Annotated[str, "The id `submit` returned."]) -> JobStatus:
        return await jobs.cancel(job_id)

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
        # Only a bus without an index (a hand-built container; wiring always wraps one) cannot
        # answer "what is running". Saying so beats an empty list that reads as "nothing is".
        indexed = isinstance(bus, RunIndex)
        summaries: list[RunSummary] = bus.snapshot(request_id) if isinstance(bus, RunIndex) else []
        # Terminal entries are reported too, not just live ones. A run's metrics rows are written
        # by a queue the recorder drains off the hot path, so between a consult returning and
        # that flush the ledger has nothing — and the index is the only thing that can say the
        # run existed, let alone what it cost.
        in_flight = [projections.in_flight_run(s) for s in summaries if s.in_flight]
        just_finished = [projections.in_flight_run(s) for s in summaries if not s.in_flight]

        reader = current.metrics_reader
        recent: list[RecentRun] = []
        calls: list[RunCall] | None = None
        if reader is not None:
            if request_id is not None:
                # For one run the ledger answers in full detail, per call; `recent` is the
                # listing view and has nothing to add that `calls` doesn't say better.
                calls = [projections.run_call(row) for row in await reader.run_calls(request_id)]
            else:
                recent = [
                    projections.recent_run(row) for row in await reader.recent_runs(limit=limit)
                ]
        return RunsReport(
            in_flight=in_flight,
            just_finished=just_finished,
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


def _from_catalogue[T](lookup: Callable[[str], T | None], model: str) -> T | None:
    """What litellm's catalog says about a model, or None. Best-effort: a listing must not fail
    because a third-party catalog read did."""
    try:
        return lookup(model)
    except Exception:  # pragma: no cover - defensive around a third-party catalog read
        logger.debug("model catalogue lookup failed", model=model, exc_info=True)
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


def _job_tool_result(record: JobRecord) -> CallToolResult:
    """A job's status plus, once there is one, its result — worded for a text-only client too.

    Without the members' own answers: a session polling for the synthesis should not be handed
    the whole panel's output because it happened to ask a second time. `answers` has them.
    """
    job = record.status
    result = without_member_answers(record.result) if record.result is not None else None
    if job.state == "completed" and result is not None:
        text, is_error = _text_summary(result), result.status == "failed"
    elif job.state in ("running", "synthesizing"):
        stage = f"synthesizing with {job.synthesizer}" if job.synthesizer else "running"
        text = (
            f"job {job.job_id} is still {stage}: {job.members_done} of {job.members_total} "
            f"members done, ${job.cost_usd:.4f} so far. Call `result` again later."
        )
        is_error = False
    else:
        text = f"job {job.job_id} {job.state}: {job.detail or 'no result'}"
        is_error = job.state != "cancelled"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=JobResult(job=job, result=result).model_dump(mode="json"),
        is_error=is_error,
    )


def _text_summary(result: ConsultResult) -> str:
    """The text block, for a client that reads `content` rather than `structuredContent`.

    On a failure that is the whole story it gets, so the breakdown the envelope preserves — who
    ran, how they ended, what it cost — is spelled out here too rather than left to a field the
    reader may never look at.
    """
    if result.status == "failed":
        error = result.error
        headline = f"{error.code}: {error.message}" if error else "the panel failed"
        if not result.members:
            return headline
        ran = ", ".join(f"{m.identity} {m.status}" for m in result.members)
        return (
            f"{headline} — {len(result.members)} member(s) ran ({ran}), "
            f"${result.total_cost_usd:.4f} spent"
        )
    if result.status == "tool_calls":
        names = ", ".join(_tool_call_names(result.tool_calls))
        called = f"[panel called {len(result.tool_calls)} tool(s): {names}]"
        # A model may write prose *and* call a tool; neither half is dropped.
        return f"{result.answer}\n\n{called}" if result.answer else called
    return result.answer


def _tool_call_names(tool_calls: Sequence[dict[str, Any]]) -> list[str]:
    names = []
    for call in tool_calls:
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        names.append(str(name or "unnamed"))
    return names
