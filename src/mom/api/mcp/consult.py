"""Running a panel for the ``consult`` tool: inline panels, progress, and the outcome envelope.

The run itself is the ordinary path — ``resolve_plan`` → ``resolve_events`` → ``collect``, the
same three calls ``routers/chat.py`` makes. What is specific to MCP is on either side of it:
assembling a call-scoped panel from catalog llms beforehand, and folding the event stream into
progress notifications plus one structured result afterwards.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
import contextlib
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mom.api.mcp.projections import (
    abandoned_report,
    consult_failure,
    consult_success,
    member_report,
)
from mom.api.mcp.schemas import ConsultResult, MemberReport
from mom.api.reqid import progress_url_from_base
from mom.api.runs import resolve_events
from mom.config.resolve import ResolvedCatalog, resolve_ensemble
from mom.config.schema import EnsembleConfig
from mom.domain.errors import ConfigError, MomError
from mom.domain.events import (
    FanoutStarted,
    MemberAbandoned,
    MemberCompleted,
    StreamEvent,
    SynthesisStarted,
)
from mom.domain.request import ChatRequestIR, MessageIR, ToolSpec
from mom.domain.results import ModelOutcome, Usage
from mom.engine.pipeline import collect
from mom.engine.plan import resolve_plan
from mom.runtime.container import Container


# The name an inline panel runs (and bills) under. Fixed rather than per-call, because `ensemble`
# is a metrics grouping dimension: a unique name per consult would shatter it into thousands of
# one-row buckets, where one bucket keeps "what have ad-hoc panels cost me" answerable.
#
# The `:` is deliberate. Config rejects `:` and `+` in llm and ensemble names (reserved
# characters), so this name cannot collide with anything an operator configures — an inline panel
# can never shadow a real ensemble, and a real ensemble can never be mistaken for one.
INLINE_ENSEMBLE = "mcp:adhoc"


def build_ir(
    *,
    ensemble: str,
    prompt: str,
    system: str | None,
    effort: str | None,
    tools: Sequence[dict[str, Any]] | None,
) -> ChatRequestIR:
    """The one-turn request a consult runs. Single-turn deliberately: a conversation belongs on
    the model endpoint, which is what the wire surfaces are for."""
    messages: list[MessageIR] = []
    if system:
        messages.append(MessageIR(role="system", content=system))
    messages.append(MessageIR(role="user", content=prompt))
    return ChatRequestIR(
        model=ensemble,
        messages=tuple(messages),
        tools=tuple(_tool_specs(tools)),
        effort=effort,
    )


def _tool_specs(tools: Sequence[dict[str, Any]] | None) -> list[ToolSpec]:
    """Accept OpenAI-shaped tool definitions, either wrapped (``{type, function}``) or bare."""
    specs: list[ToolSpec] = []
    for tool in tools or ():
        nested = tool.get("function")
        body: dict[str, Any] = nested if isinstance(nested, dict) else tool
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise ToolError("each tool needs a name (OpenAI function-tool shape)")
        specs.append(
            ToolSpec(
                name=name,
                description=body.get("description"),
                parameters=body.get("parameters"),
                # Forwarded like the wire surfaces do (see api/translate.py): dropping it here
                # would quietly weaken schema enforcement for the same tool definition depending
                # on which surface the caller happened to use.
                strict=body.get("strict"),
            )
        )
    return specs


def inline_catalog(
    catalog: ResolvedCatalog, *, panel: Sequence[str], synthesizer: str
) -> ResolvedCatalog:
    """A copy of the catalog with one extra, call-scoped ensemble (see ``INLINE_ENSEMBLE``).

    Built as an ``EnsembleConfig`` and put through the normal resolver, so an inline panel gets
    the same schema defaults and the same unknown-llm validation a configured one does. The
    catalog is a frozen dataclass over read-only mappings, so this is a value — the process
    catalog is untouched and nothing reaches config on disk.
    """
    if not panel:
        raise ToolError("panel needs at least one llm (see list_llms)")
    duplicates = sorted({name for name in panel if list(panel).count(name) > 1})
    if duplicates:
        raise ToolError(
            f"panel lists {', '.join(duplicates)} more than once; each llm may appear once"
        )
    try:
        config = EnsembleConfig.model_validate(
            {"members": list(panel), "synthesizer": {"llm": synthesizer}}
        )
        resolved = resolve_ensemble(INLINE_ENSEMBLE, config, catalog.llms, catalog.config.prompts)
    except (ConfigError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    return replace(
        catalog,
        ensembles=MappingProxyType({**catalog.ensembles, INLINE_ENSEMBLE: resolved}),
    )


@dataclass
class RunObserver:
    """What the event stream said, kept for the result — including for a run that fails.

    ``collect`` raises on a terminal failure, so by the time the tool can build a payload the
    stream is gone. Accumulating outcomes as they pass is what lets a failed consult still report
    the members that answered and the money they cost.
    """

    outcomes: list[ModelOutcome] = field(default_factory=list)
    abandoned: list[tuple[str, str]] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(outcome.cost_usd for outcome in self.outcomes)

    @property
    def usage(self) -> Usage:
        """Tokens spent by the members seen so far — what a failed run has to report, since
        `collect` raised before it could total anything."""
        total = Usage()
        for outcome in self.outcomes:
            total = total + outcome.usage
        return total

    def reports(self, *, include_answers: bool) -> list[MemberReport]:
        """Every member of the panel, whether it answered, failed, or ran out of time."""
        return [
            *(member_report(o, include_answers=include_answers) for o in self.outcomes),
            *(abandoned_report(identity, model) for identity, model in self.abandoned),
        ]


async def with_progress(
    events: AsyncIterator[StreamEvent],
    ctx: Context[Any, Any],
    observer: RunObserver,
    *,
    total: int,
) -> AsyncGenerator[StreamEvent, None]:
    """Pass the stream through untouched, reporting each milestone to the MCP client.

    A fan-out is slow enough that a client with no feedback cannot tell it from a hang, and the
    per-member cost is only knowable while it streams. This is the encoders' pattern — one more
    fold over the same events — not a second consumer: ``collect`` still sees every event.
    """
    step = 0
    async for event in events:
        message: str | None = None
        if isinstance(event, FanoutStarted):
            # Reported, not just counted: until the first member lands there is nothing else to
            # say, and a panel whose slowest seat runs for minutes is otherwise indistinguishable
            # from a hung tool call — which is what these notifications exist to rule out.
            step += 1
            message = f"asking {event.identity} ({event.model})"
        elif isinstance(event, MemberCompleted):
            observer.outcomes.append(event.outcome)
            step += 1
            message = (
                f"{event.outcome.identity}: {event.outcome.status} "
                f"(${observer.cost_usd:.4f} so far)"
            )
        elif isinstance(event, MemberAbandoned):
            observer.abandoned.append((event.identity, event.model))
            step += 1
            message = f"{event.identity}: abandoned at the fan-out deadline"
        elif isinstance(event, SynthesisStarted):
            step += 1
            message = f"synthesizing with {event.llm}"
        if message is not None:
            await _report(ctx, step, total, message)
        yield event


async def _report(ctx: Context[Any, Any], progress: int, total: int, message: str) -> None:
    """Best-effort progress. Suppressed narrowly and only around the notification itself: there
    is no request context when a tool is called directly (tests), and a client that sent no
    progress token gets a documented no-op — neither is a reason to fail the run."""
    with contextlib.suppress(Exception):
        await ctx.report_progress(progress, total, message)


async def run_consult(
    container: Container,
    ctx: Context[Any, Any],
    *,
    ensemble: str | None,
    panel: Sequence[str] | None,
    synthesizer: str | None,
    prompt: str,
    system: str | None,
    effort: str | None,
    tools: Sequence[dict[str, Any]] | None,
    include_member_answers: bool,
    base_url: str | None,
) -> ConsultResult:
    """Run one panel and return its outcome envelope (never raises for an upstream failure)."""
    if (ensemble is None) == (panel is None):
        raise ToolError("pass exactly one of `ensemble` (a configured panel) or `panel` (llms)")
    if ensemble is not None:
        catalog, name = container.catalog, ensemble
    else:
        if not synthesizer:
            raise ToolError("`panel` needs a `synthesizer` llm to combine the answers")
        catalog = inline_catalog(container.catalog, panel=panel or (), synthesizer=synthesizer)
        name = INLINE_ENSEMBLE

    ir = build_ir(ensemble=name, prompt=prompt, system=system, effort=effort, tools=tools)
    try:
        plan = resolve_plan(catalog, ir)
    except MomError as exc:
        # A caller mistake — unknown ensemble, unknown llm, a panel the config rejects. It is
        # reported as a failed *call* (no result payload) rather than a run outcome, because the
        # agent has to change the arguments before anything can run.
        raise ToolError(exc.safe_message) from exc

    if effort and plan.tier is None:
        # `resolve_plan` only validates an effort token against an ensemble that declares tiers;
        # for one that doesn't — every inline panel, since they are tierless — it neither applies
        # nor rejects it. Silently accepting an argument that cannot do anything is the wrong
        # answer for a tool an agent calls believing it bought deeper reasoning.
        raise ToolError(
            f"ensemble {name!r} declares no effort tiers, so `effort` would have no effect"
            if ensemble is not None
            else "an inline `panel` has no effort tiers; configure an ensemble to use `effort`"
        )

    if name == INLINE_ENSEMBLE:
        # Coalescing keys on the ensemble NAME plus the messages, never the roster, so two
        # different inline panels asking the same question would look identical and the second
        # would silently receive the first one's answer. Named ensembles have a real identity
        # and keep the optimization.
        plan = replace(plan, dedupe=False)

    request_id = container.ids.new_id("req")
    events, leader_request_id = resolve_events(container, plan, ir, request_id)
    coalesced = leader_request_id != request_id
    # No token in the link: this one is returned as tool-result data, not as a response header to
    # a caller who already authenticated. A client that reached the HTTP surface has the token
    # already; over stdio it never had one, and must not learn it from here.
    progress_url = progress_url_from_base(base_url, leader_request_id, container, with_token=False)
    observer = RunObserver()
    # Two steps per member (asked, then answered), one for synthesis starting, and one left
    # unclaimed: synthesis is usually the longest wait, so a bar reading 100% the moment it
    # begins is a lie.
    total = len(plan.members) * 2 + 2

    try:
        # `aclosing`, because `collect` raises from inside its own `async for`: without it the
        # generator chain is left suspended at `yield PipelineFailed(...)`, deferring
        # `run_ensemble`'s cleanup — which cancels still-pending member calls — to garbage
        # collection. Abandoned upstream calls would keep running after the tool returned.
        async with aclosing(with_progress(events, ctx, observer, total=total)) as stream:
            result = await collect(stream)
    except MomError as exc:
        return consult_failure(
            exc,
            ensemble=name,
            request_id=leader_request_id,
            coalesced=coalesced,
            progress_url=progress_url,
            members=observer.reports(include_answers=include_member_answers),
            usage=observer.usage,
            notices=list(plan.notices),
        )

    # The observer for both outcomes, not `result.outcomes` here and the observer there: the two
    # are built from the same MemberCompleted events, and one source means the success and
    # failure envelopes cannot drift apart later.
    return consult_success(
        result,
        ensemble=name,
        request_id=leader_request_id,
        coalesced=coalesced,
        progress_url=progress_url,
        members=observer.reports(include_answers=include_member_answers),
        notices=list(plan.notices),
    )
