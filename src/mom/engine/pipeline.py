"""The one orchestration pipeline. Streaming and non-streaming are two consumers of it.

``run_ensemble`` emits a typed event stream; ``collect`` drains it into an ``EnsembleResult``.
There is no second code path, so the two modes cannot drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any
import uuid

from mom.domain.cost import compute_cost
from mom.domain.errors import ErrorKind, MomError, QuorumNotMet, UpstreamError, UpstreamTimeout
from mom.domain.events import (
    AnswerDelta,
    Completed,
    FanoutSkipped,
    FanoutStarted,
    FinishReason,
    MemberAbandoned,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.metrics import CallMetric, MetricsSink, TurnType
from mom.domain.ports import (
    CallSpec,
    Clock,
    CompletionChunk,
    EventBus,
    IdFactory,
    LLMClient,
    ToolCallCustody,
    Tracer,
)
from mom.domain.progress import PREVIEW_CHARS, ProgressEvent
from mom.domain.prompt_caching import inject_anthropic_cache
from mom.domain.results import EnsembleResult, ModelOutcome, OutcomeStatus, Usage
from mom.domain.synthesis import all_failed_message, append_instruction, build_synthesis_messages
from mom.domain.tooling import restore_provider_tool_ids, select_member_tool_call
from mom.engine.plan import ExecutionPlan, PlannedMember
from mom.runtime.logging import get_logger


logger = get_logger("mom.engine")


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    client: LLMClient
    clock: Clock
    recorder: MetricsSink | None = None
    tracer: Tracer | None = None
    bus: EventBus | None = None
    request_id: str = ""
    ids: IdFactory | None = None
    custody: ToolCallCustody | None = None


def _mint_call_id(deps: PipelineDeps) -> str:
    """A stable, client-facing tool-call id that never carries a provider-native signature."""
    if deps.ids is not None:
        return deps.ids.new_id("call")
    return f"call_{uuid.uuid4().hex}"


def _publish(deps: PipelineDeps, event: ProgressEvent) -> None:
    """Publish a progress event, swallowing any failure (progress must never break a request)."""
    if deps.bus is None:
        return
    try:
        deps.bus.publish(deps.request_id, event)
    except Exception:
        logger.debug("progress publish failed", exc_info=True)


def _preview(text: str) -> str | None:
    """A bounded glimpse of an output for the progress feed — see ``PREVIEW_CHARS``."""
    if not text:
        return None
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS] + "…"


def _record_member(
    deps: PipelineDeps, plan: ExecutionPlan, outcome: ModelOutcome, turn_type: TurnType
) -> None:
    if deps.recorder is None:
        return
    usage = outcome.usage
    deps.recorder.record(
        CallMetric(
            request_id=deps.request_id,
            ts=deps.clock.now(),
            ensemble=plan.ensemble,
            llm=outcome.llm,
            model=outcome.model,
            role="fanout",
            # The full status, not the old "ok"/"error" collapse — 'empty'/'timeout'/'aborted'
            # are distinct failure modes and used to be indistinguishable in the metrics DB.
            status=outcome.status,
            cache_hit=outcome.cached,
            turn_type=turn_type,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_prompt_tokens=usage.cached_prompt_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=outcome.cost_usd,
            duration_ms=outcome.duration_ms,
            error=outcome.error,
            finish_reason=outcome.finish_reason,
            error_kind=outcome.error_kind,
            error_detail=outcome.error_detail,
            attempts=outcome.attempts,
        )
    )


def _record_synth(
    deps: PipelineDeps,
    plan: ExecutionPlan,
    usage: Usage,
    cost: float,
    turn_type: TurnType,
    *,
    finish_reason: str | None = None,
    attempts: int = 1,
    duration_ms: float | None = None,
) -> None:
    if deps.recorder is None:
        return
    deps.recorder.record(
        CallMetric(
            request_id=deps.request_id,
            ts=deps.clock.now(),
            ensemble=plan.ensemble,
            llm=plan.synth.llm_name,
            model=plan.synth.model,
            role="synthesis",
            status="ok",
            duration_ms=duration_ms,
            turn_type=turn_type,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_prompt_tokens=usage.cached_prompt_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
            finish_reason=finish_reason,
            attempts=attempts,
        )
    )


def _elapsed_ms(deps: PipelineDeps, started: float | None) -> float | None:
    """Milliseconds since ``started``, or None if the stage never began."""
    return None if started is None else (deps.clock.now() - started) * 1000.0


def _record_synth_failure(
    deps: PipelineDeps,
    plan: ExecutionPlan,
    turn_type: TurnType,
    *,
    error: str,
    error_kind: ErrorKind,
    error_detail: str | None,
    attempts: int,
    duration_ms: float | None = None,
) -> None:
    """A failed synthesis call used to be recorded nowhere — the metrics DB simply had no row for
    it, so a repeatedly-failing (and repeatedly-retried, per this session's own retry-loop fix)
    synthesizer was invisible to any cost/error accounting."""
    if deps.recorder is None:
        return
    deps.recorder.record(
        CallMetric(
            request_id=deps.request_id,
            ts=deps.clock.now(),
            ensemble=plan.ensemble,
            llm=plan.synth.llm_name,
            model=plan.synth.model,
            role="synthesis",
            status="error",
            duration_ms=duration_ms,
            turn_type=turn_type,
            error=error,
            error_kind=error_kind,
            error_detail=error_detail,
            attempts=attempts,
        )
    )


def _exc_site(exc: BaseException) -> str:
    """Where an exception was raised, as ``file:line in function`` — never its message.

    Deliberately not a traceback: a traceback's final line is ``ExceptionType: <message>``, and an
    exception raised while parsing a provider's chunk carries that provider's text in its message
    (``int(call["index"])`` on a malformed tool call renders the raw value verbatim). A file name,
    line number and function name are identifiers from this repo, never provider data — so this
    still says where to look without putting an unbounded string on stdout.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "unknown"
    innermost = frames[-1]
    return f"{Path(innermost.filename).name}:{innermost.lineno} in {innermost.name}"


def _coerce_finish(value: str | None) -> FinishReason:
    match value:
        case "stop" | "length" | "tool_calls" | "content_filter" | "error":
            return value
        case _:
            return "stop"


def _cause_text(exc: BaseException, *, max_chars: int = 500) -> str:
    """Walk ``__cause__``/``__context__`` to the root and return a truncated, operator-facing str.

    litellm/provider SDKs often wrap the real failure (a timeout, an HTTP error body) in a generic
    exception; the outer ``str(exc)`` alone can be uninformative (e.g. just the exception class
    name). This is for logs/metrics/tracing only — never returned to a client.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            parts.append(text)
        current = current.__cause__ or current.__context__
    combined = " | caused by: ".join(parts) if parts else exc.__class__.__name__
    return combined[:max_chars]


async def _run_member(deps: PipelineDeps, member: PlannedMember) -> ModelOutcome:
    start = deps.clock.now()
    common = {"identity": member.identity, "llm": member.identity, "model": member.spec.model}
    try:
        timeout = member.spec.timeout_seconds
        completion = await asyncio.wait_for(deps.client.complete(member.spec), timeout=timeout)
    except TimeoutError:
        # Elapsed on every failure path too, not just success: a member that timed out or errored
        # otherwise reports duration_ms=0 (the ModelOutcome default) in logs, metrics and traces —
        # zero on exactly the outcomes an operator is trying to time.
        return ModelOutcome(
            **common,  # type: ignore[arg-type]
            status="timeout",
            error="timed out",
            error_kind="timeout",
            duration_ms=(deps.clock.now() - start) * 1000.0,
        )
    except MomError as exc:
        # exc.safe_message is what the client sees; log the real cause for the operator — this
        # branch used to return silently, which is why provider failures were invisible in logs
        # and Langfuse (UpstreamError is a MomError, so the generic `except Exception` below,
        # which does log, never ran for the failures that actually mattered). exc.kind/exc.detail
        # are already the classified/scrubbed pair the adapter attached (or the class-level
        # "unknown"/None default for a MomError that isn't an UpstreamError) — never re-derive a
        # detail from raw text here, so `error_detail` is scrubbed-or-absent, never "sometimes".
        logger.warning(
            "member call failed",
            llm=member.identity,
            model=member.spec.model,
            request_id=deps.request_id,
            error=_cause_text(exc),
        )
        return ModelOutcome(
            **common,  # type: ignore[arg-type]
            status="error",
            error=exc.safe_message,
            error_kind=exc.kind,
            error_detail=exc.detail,
            attempts=exc.attempts,
            duration_ms=(deps.clock.now() - start) * 1000.0,
        )
    except Exception as exc:
        # Never surface raw provider/third-party text (it can carry keys, URLs, internal paths);
        # log it for the operator and return a safe, generic message. No adapter has classified or
        # scrubbed this one (it isn't a MomError at all — a client implementation bug, not a normal
        # provider failure), so `error_detail` stays unset rather than persisting unscrubbed text.
        logger.warning(
            "member call failed",
            llm=member.identity,
            model=member.spec.model,
            request_id=deps.request_id,
            error=_cause_text(exc),
        )
        return ModelOutcome(
            **common,  # type: ignore[arg-type]
            status="error",
            error="call failed",
            error_kind="unknown",
            duration_ms=(deps.clock.now() - start) * 1000.0,
        )
    duration_ms = (deps.clock.now() - start) * 1000.0
    # A member that only proposed tool calls (no prose) is still a real answer, not "empty" — its
    # proposal feeds the candidate envelope and the vote/first strategies.
    answered = bool(completion.content.strip() or completion.tool_calls)
    status: OutcomeStatus = "ok" if answered else "empty"
    # Config pricing wins; otherwise fall back to the adapter's litellm cost. Cache hits cost $0.
    if completion.cached:
        cost = 0.0
    elif member.pricing is not None:
        cost = compute_cost(completion.usage, member.pricing)
    else:
        cost = completion.cost_usd or 0.0
        _warn_once_if_free(member.spec.model, completion.usage, cost)
    return ModelOutcome(
        **common,  # type: ignore[arg-type]
        status=status,
        content=completion.content,
        reasoning=completion.reasoning,
        usage=completion.usage,
        cached=completion.cached,
        duration_ms=duration_ms,
        cost_usd=cost,
        tool_calls=completion.tool_calls,
        finish_reason=completion.finish_reason,
        attempts=completion.attempts,
    )


# Models already reported as unpriced. Spend is per-model, so one line per model is the whole
# signal; repeating it every call would bury it.
_FREE_MODELS_REPORTED: set[str] = set()


def _warn_once_if_free(model: str, usage: Usage, cost: float) -> None:
    """Report a real call that burned tokens and priced at $0 — once per model.

    The startup catalog check predicts this; here it is *observed*, which is what makes it exact.
    A model missing from litellm's catalog has no cost-per-token, so its spend silently records
    as zero — unless the provider returns a cost of its own, and whether it does is a runtime
    fact no config can state (OpenRouter does; Gemini and xAI do not). Rather than guess from a
    provider name, price the call and see.

    Live at the time of writing: gemini-3.7-flash and both xAI models had billed to $0.00 across
    230 calls, and Opus 5 joined them while its catalog entry was missing. Every metric, budget,
    and cost report over that window undercounted, with nothing anywhere saying so. Fix by
    declaring ``pricing:`` for the model or raising the litellm floor.
    """
    if cost or not usage.total_tokens or model in _FREE_MODELS_REPORTED:
        return
    _FREE_MODELS_REPORTED.add(model)
    logger.warning(
        "model priced at $0 for a real call: litellm has no cost-per-token for it and the "
        "provider reported none, so its spend is invisible to metrics and budgets — declare "
        "`pricing:` for it in config, or raise the litellm floor so its catalog entry exists",
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


# Members detached on client disconnect keep running here so they finish and cache; the strong
# reference stops the event loop from GC-ing them, and the callback clears it when they're done.
_DETACHED: set[asyncio.Task[ModelOutcome]] = set()


def _detach_member(
    task: asyncio.Task[ModelOutcome],
    deps: PipelineDeps,
    plan: ExecutionPlan,
    turn_type: TurnType,
) -> None:
    """Let an in-flight member finish in the background (and cache) instead of cancelling it.

    The request loop is gone, so record the member's metric from the completion callback — the
    spend really happened, and a later retry that cache-hits it must not double-count as free.
    """
    _DETACHED.add(task)

    def _done(t: asyncio.Task[ModelOutcome]) -> None:
        _DETACHED.discard(t)
        if t.cancelled():
            return
        if t.exception() is not None:
            logger.warning(
                "detached fan-out member errored",
                request_id=deps.request_id,
                ensemble=plan.ensemble,
                error=str(t.exception()),
            )
            return
        outcome = t.result()
        _record_member(deps, plan, outcome, turn_type)
        # The run's own logging ended with the client; without this, a detached member's spend is
        # recorded as a metric but never appears in the log, so the lines under-report real cost.
        # Distinct event name because it lands after this request's "run completed".
        logger.info(
            "detached member completed",
            request_id=deps.request_id,
            ensemble=plan.ensemble,
            llm=outcome.llm,
            model=outcome.model,
            status=outcome.status,
            cached=outcome.cached,
            duration_ms=round(outcome.duration_ms, 1),
            tokens=outcome.usage.total_tokens,
            cost_usd=round(outcome.cost_usd, 6),
        )

    task.add_done_callback(_done)


async def _fan_out(
    deps: PipelineDeps, plan: ExecutionPlan, turn_type: TurnType = "ensemble"
) -> AsyncIterator[StreamEvent]:
    for member in plan.members:
        yield FanoutStarted(member.identity, member.spec.model)
    sem = asyncio.Semaphore(plan.max_concurrency)

    async def run(member: PlannedMember) -> ModelOutcome:
        async with sem:  # cap concurrent upstream calls
            # Inside the semaphore, so this marks when the call actually goes out — not when it
            # was planned. On a panel wider than max_concurrency the stagger is the whole signal.
            logger.info(
                "member dispatched",
                request_id=deps.request_id,
                ensemble=plan.ensemble,
                llm=member.identity,
                model=member.spec.model,
            )
            return await _run_member(deps, member)

    tasks = [asyncio.create_task(run(m)) for m in plan.members]
    # task -> its PlannedMember, so an abandoned task can be named (identity/model) in its event.
    member_of: dict[asyncio.Task[ModelOutcome], PlannedMember] = dict(
        zip(tasks, plan.members, strict=True)
    )
    loop = asyncio.get_running_loop()
    deadline = None if plan.fanout_deadline is None else loop.time() + plan.fanout_deadline
    # `pending` means "not yet accounted for" — its outcome hasn't been reported (a yielded
    # MemberCompleted, which is what triggers run_ensemble's _record_member) and it hasn't been
    # handed to `finally` for detach/cancel. This is DELIBERATELY not the same thing as
    # asyncio.wait's own "didn't complete this round" set: `yield` is a suspension point, so if the
    # consumer tears this generator down (GeneratorExit) between two yields in the same round, a
    # task that finished (is in `done`) but hasn't been individually yielded yet must still fall
    # through to `finally` — otherwise it's silently dropped with no event and no metric, because
    # `task.done()` is already True for it and the old `if not task.done()` check in `finally`
    # skipped it entirely.
    pending = set(tasks)
    try:
        while pending:
            timeout = None if deadline is None else max(0.0, deadline - loop.time())
            done, _ = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:  # deadline elapsed with nothing new — abandon the stragglers
                logger.warning(
                    "fan-out deadline exceeded",
                    request_id=deps.request_id,
                    ensemble=plan.ensemble,
                    pending=len(pending),
                )
                # In plan (config) order, not set-iteration order, for a deterministic sequence.
                # Still genuinely running (not done) — leave them IN `pending` so `finally` below
                # detaches/cancels them; this only announces that they won't make the deadline.
                for task in tasks:
                    if task in pending:
                        member = member_of[task]
                        yield MemberAbandoned(member.identity, member.spec.model)
                break
            for task in done:
                pending.discard(task)  # accounted for — its outcome is about to be reported
                yield MemberCompleted(await task)
    finally:
        for task in tasks:
            if task in pending:
                if plan.detach_on_disconnect:
                    # finish + cache in the background (a retry hits it); its metric is recorded
                    # from the completion callback since the request loop is gone.
                    _detach_member(task, deps, plan, turn_type)
                else:
                    task.cancel()


async def _stream_with_timeout(
    stream: AsyncIterator[CompletionChunk], seconds: float
) -> AsyncIterator[CompletionChunk]:
    """Yield chunks, failing with ``UpstreamTimeout`` if none arrives within ``seconds``.

    The timeout guards only fetching the next chunk (an idle/stalled provider); the ``yield`` is
    outside it, so a healthy stream is never interrupted mid-flight.
    """
    iterator = stream.__aiter__()
    while True:
        try:
            async with asyncio.timeout(seconds):
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise UpstreamTimeout("synthesis timed out") from exc
        yield chunk


def _short_circuit_tool_events(
    deps: PipelineDeps, calls: tuple[dict[str, Any], ...], outcomes: list[ModelOutcome]
) -> list[StreamEvent]:
    """Emit a member-selected (vote/first) tool call directly, minting client ids, no synthesis.

    Provider ids from members are dropped (not stashed): a relay continuation goes to the
    synthesizer, so there is nothing for a member id to restore. The final ``Completed`` carries
    the members' usage/cost (no synthesizer call was made).
    """
    events: list[StreamEvent] = []
    for index, call in enumerate(calls):
        fn = call.get("function", {})
        name = str(fn.get("name") or "") if isinstance(fn, dict) else ""
        events.append(ToolCallStarted(index=index, call_id=_mint_call_id(deps), name=name))
        arguments = fn.get("arguments") if isinstance(fn, dict) else None
        if arguments:
            events.append(ToolCallDelta(index=index, arguments_fragment=str(arguments)))
    total_usage = Usage()
    for outcome in outcomes:
        total_usage = total_usage + outcome.usage
    total_cost = sum(o.cost_usd for o in outcomes)
    events.append(
        Completed(finish_reason="tool_calls", usage=total_usage, total_cost_usd=total_cost)
    )
    return events


def _restore_relay_ids(
    deps: PipelineDeps, plan: ExecutionPlan, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """On a relay turn, swap minted client ids back to the synthesizer's provider-native ids."""
    custody = deps.custody
    if custody is None:
        return messages
    synth_llm = plan.synth.llm_name

    def _lookup(client_id: str) -> str | None:
        return custody.provider_id(client_id, synth_llm)

    return restore_provider_tool_ids(messages, _lookup)


def _synth_spec(plan: ExecutionPlan, messages: list[dict[str, object]]) -> CallSpec:
    return CallSpec(
        llm_name=plan.synth.llm_name,
        model=plan.synth.model,
        messages=messages,
        params=dict(plan.synth.params),
        api=plan.synth.api,
        proxy_url_env=plan.synth.proxy_url_env,
        key_env_candidates=plan.synth.key_env_candidates,
        retries=plan.synth.retries,
        retry_backoff_seconds=plan.synth.retry_backoff_seconds,
        timeout_seconds=plan.synth.timeout_seconds,
    )


async def run_ensemble(plan: ExecutionPlan, deps: PipelineDeps) -> AsyncIterator[StreamEvent]:
    """Run an ensemble, emitting a typed event stream. Never raises — failures are events."""
    turn_type: TurnType = "relay" if plan.skip_reason == "tool_continuation" else "ensemble"
    terminal_published = False
    run_started = deps.clock.now()
    # Bound rather than contextvars: this generator is consumed from varying tasks (the router's,
    # a coalescing leader's, or `aclose()` during registry teardown), so there is no single context
    # to bind to — and structlog's capture_logs drops merge_contextvars, which would make these
    # fields invisible to tests. Every line below carries request_id/ensemble for free.
    log = logger.bind(request_id=deps.request_id, ensemble=plan.ensemble)
    # True only once the REAL synthesizer call is under way (not the vote/first short-circuit's
    # synthetic SynthesisStarted, which never calls an LLM) — gates the failure-recording below so
    # a fan-out-stage error is never mistakenly recorded as a synthesis failure.
    synthesizing = False
    synth_started: float | None = None
    # The DISPATCHED roster size, not len(outcomes): a member abandoned at the fan-out deadline
    # never yields a MemberCompleted, so counting outcomes would report "2 of 2 ok" as "1 of 1 ok"
    # — erasing the shortfall from the one line that exists to summarize it.
    members_total = 0

    def _publish_terminal(event: ProgressEvent) -> None:
        nonlocal terminal_published
        terminal_published = True
        _publish(deps, event)

    try:
        outcomes: list[ModelOutcome] = []
        if plan.skip_fanout:
            # Publish SOMETHING on a passthrough/relay turn too — previously nothing was ever
            # published here, so a progress dashboard opened on such a turn showed a blank page
            # indistinguishable from a stuck/broken request. members_total=0 + detail=skip_reason
            # lets the dashboard render an honest "no fan-out this turn" state instead.
            _publish(
                deps,
                ProgressEvent(
                    kind="fanout_started",
                    ensemble=plan.ensemble,
                    members_total=0,
                    detail=plan.skip_reason,
                ),
            )
            log.info("fan-out skipped", reason=plan.skip_reason)
            if plan.skip_reason is not None:
                yield FanoutSkipped(plan.skip_reason)
            synth_messages = plan.client_messages
            if plan.skip_reason == "tool_continuation":
                synth_messages = _restore_relay_ids(deps, plan, synth_messages)
            # The instruction is stripped from the client message during plan resolution
            # regardless of path — without this, a passthrough/relay turn silently dropped it
            # (it was never re-attached anywhere on this branch).
            synth_messages = append_instruction(synth_messages, plan.instruction)
        else:
            members_total = len(plan.members)
            _publish(
                deps,
                ProgressEvent(
                    kind="fanout_started",
                    ensemble=plan.ensemble,
                    members_total=members_total,
                    members=tuple((m.identity, m.spec.model) for m in plan.members),
                ),
            )
            log.info(
                "fan-out started",
                members_total=members_total,
                members=[f"{m.identity}={m.spec.model}" for m in plan.members],
            )
            async for event in _fan_out(deps, plan, turn_type):
                if isinstance(event, MemberCompleted):
                    outcome = event.outcome
                    outcomes.append(outcome)
                    _record_member(deps, plan, outcome, turn_type)
                    # Fires for failed members too, so an operator can count members in and out;
                    # the cause of a failure stays on the warning _run_member already logs.
                    log.info(
                        "member completed",
                        llm=outcome.llm,
                        model=outcome.model,
                        status=outcome.status,
                        cached=outcome.cached,
                        duration_ms=round(outcome.duration_ms, 1),
                        tokens=outcome.usage.total_tokens,
                        cost_usd=round(outcome.cost_usd, 6),
                        attempts=outcome.attempts,
                        completed=len(outcomes),
                        members_total=members_total,
                        **({"error_kind": outcome.error_kind} if outcome.error_kind else {}),
                    )
                    _publish(
                        deps,
                        ProgressEvent(
                            kind="member_completed",
                            ensemble=plan.ensemble,
                            member=outcome.identity,
                            model=outcome.model,
                            status=outcome.status,
                            duration_ms=outcome.duration_ms,
                            members_total=members_total,
                            completed=len(outcomes),
                            cost_usd=outcome.cost_usd,
                            preview=_preview(
                                outcome.content if outcome.ok else outcome.error or ""
                            ),
                        ),
                    )
                    if deps.tracer is not None:
                        deps.tracer.observe(
                            request_id=deps.request_id,
                            ensemble=plan.ensemble,
                            role="fanout",
                            llm=outcome.llm,
                            model=outcome.model,
                            messages=plan.client_messages,
                            output=outcome.content,
                            usage=outcome.usage,
                            duration_ms=outcome.duration_ms,
                            cached=outcome.cached,
                            error=outcome.error,
                            status=outcome.status,
                            cost_usd=outcome.cost_usd,
                            error_kind=outcome.error_kind,
                            error_detail=outcome.error_detail,
                            finish_reason=outcome.finish_reason,
                        )
                elif isinstance(event, MemberAbandoned):
                    # The deadline warning above counts stragglers; this names them.
                    log.warning(
                        "member abandoned",
                        llm=event.identity,
                        model=event.model,
                        disposition="detached" if plan.detach_on_disconnect else "aborted",
                    )
                    # Reuses "member_completed" (not a new ProgressKind): the dashboard's
                    # fillCard is the only code path that clears a pending slot, and it already
                    # keys off ANY member_completed event regardless of status — so this needs no
                    # new dashboard event type, just a status value it renders distinctly (see
                    # progress.py). Publishes NO metric here (unlike a real MemberCompleted) — the
                    # spend, if any, is recorded later by _detach_member's completion callback, or
                    # never if the task was cancelled outright.
                    _publish(
                        deps,
                        ProgressEvent(
                            kind="member_completed",
                            ensemble=plan.ensemble,
                            member=event.identity,
                            model=event.model,
                            status="detached" if plan.detach_on_disconnect else "aborted",
                            duration_ms=(plan.fanout_deadline or 0.0) * 1000.0,
                            members_total=members_total,
                            completed=len(outcomes),
                            preview=(
                                "no result before the fan-out deadline — left to finish in the "
                                "background"
                                if plan.detach_on_disconnect
                                else "no result before the fan-out deadline — cancelled"
                            ),
                        ),
                    )
                yield event
            # Quorum first: refuse to synthesize on a thin panel. With the default min_results=1
            # this also replaces the all-failed fallback (0 ok -> 502); min_results=0 re-enables it.
            ok_count = sum(1 for o in outcomes if o.ok)
            if ok_count < plan.min_results:
                raise QuorumNotMet(
                    f"only {ok_count} of {len(outcomes)} members succeeded "
                    f"(min_results={plan.min_results})"
                )
            # vote/first: if members agree on a tool call, return it directly and skip synthesis.
            # Order by config (not fan-out completion) so `first` and vote tie-breaks are stable.
            if plan.tool_strategy in ("vote", "first"):
                rank = {member.identity: i for i, member in enumerate(plan.members)}
                ordered = sorted(outcomes, key=lambda o: rank.get(o.identity, len(rank)))
                selected = select_member_tool_call(
                    ordered, strategy=plan.tool_strategy, threshold=plan.vote_threshold
                )
                if selected:
                    # SynthesisStarted marks the work→answer boundary for the encoders (e.g. it
                    # closes an inline <think> block) even though no synthesizer call is made.
                    yield SynthesisStarted(plan.synth.llm_name, plan.synth.model)
                    for tool_event in _short_circuit_tool_events(deps, selected, outcomes):
                        yield tool_event
                    # This path used to `return` here with NO terminal progress event ever
                    # published — a dashboard watching a vote/first turn hung open forever.
                    _publish_terminal(
                        ProgressEvent(
                            kind="completed",
                            ensemble=plan.ensemble,
                            status="tool_calls",
                            # Members only — this path skips synthesis entirely.
                            cost_usd=sum(o.cost_usd for o in outcomes),
                        )
                    )
                    log.info(
                        "run completed",
                        status="tool_calls",
                        strategy=plan.tool_strategy,
                        members_ok=ok_count,
                        members_total=members_total,
                        total_cost_usd=round(sum(o.cost_usd for o in outcomes), 6),
                        elapsed_seconds=round(deps.clock.now() - run_started, 2),
                    )
                    return
            if any(o.ok for o in outcomes):
                synth_messages = build_synthesis_messages(
                    plan.client_messages,
                    outcomes,
                    prompt=plan.synth.prompt,
                    instruction=plan.instruction,
                )
            else:
                synth_messages = all_failed_message(outcomes)

        if plan.synth.anthropic_cache_ttl is not None:
            synth_messages = inject_anthropic_cache(
                synth_messages, ttl=plan.synth.anthropic_cache_ttl
            )
        yield SynthesisStarted(plan.synth.llm_name, plan.synth.model)
        _publish(
            deps,
            ProgressEvent(
                kind="synthesis_started",
                ensemble=plan.ensemble,
                member=plan.synth.llm_name,
                model=plan.synth.model,
            ),
        )
        log.info("synthesis started", llm=plan.synth.llm_name, model=plan.synth.model)
        usage = Usage()
        synth_llm_cost: float | None = None
        synth_attempts = 1
        finish: FinishReason = "stop"
        started_tools: set[int] = set()
        synth_text: list[str] = []
        synthesizing = True  # a real LLM call from here on — gates failure recording below
        synth_started = deps.clock.now()  # also read by the failure handlers below
        synth_stream = deps.client.stream(_synth_spec(plan, synth_messages))
        async for chunk in _stream_with_timeout(synth_stream, plan.synth.timeout_seconds):
            if chunk.attempts is not None:
                synth_attempts = chunk.attempts
            if chunk.content is not None or chunk.reasoning is not None:
                yield AnswerDelta(content=chunk.content, reasoning=chunk.reasoning)
                if chunk.content:
                    synth_text.append(chunk.content)
            if chunk.tool_call is not None:
                call = chunk.tool_call
                index = int(call.get("index", 0))
                if index not in started_tools:
                    started_tools.add(index)
                    # Mint a client-facing id; stash the provider-native one (incl. any
                    # `__thought__` signature) for a later relay, never leaking it to the client.
                    minted = _mint_call_id(deps)
                    provider_id = str(call.get("id") or "")
                    if provider_id and deps.custody is not None:
                        deps.custody.remember(minted, provider_id, plan.synth.llm_name)
                    yield ToolCallStarted(
                        index=index, call_id=minted, name=str(call.get("name") or "")
                    )
                fragment = call.get("arguments")
                if fragment:
                    yield ToolCallDelta(index=index, arguments_fragment=str(fragment))
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.cost_usd is not None:
                synth_llm_cost = chunk.cost_usd
            if chunk.finish_reason:
                finish = _coerce_finish(chunk.finish_reason)

        synth_duration_ms = (deps.clock.now() - synth_started) * 1000.0
        if plan.synth.pricing is not None:
            synth_cost = compute_cost(usage, plan.synth.pricing)
        else:
            synth_cost = synth_llm_cost or 0.0
            _warn_once_if_free(plan.synth.model, usage, synth_cost)
        _record_synth(
            deps,
            plan,
            usage,
            synth_cost,
            turn_type,
            finish_reason=finish,
            attempts=synth_attempts,
            duration_ms=synth_duration_ms,
        )
        if deps.tracer is not None:
            deps.tracer.observe(
                request_id=deps.request_id,
                ensemble=plan.ensemble,
                role="synthesis",
                llm=plan.synth.llm_name,
                model=plan.synth.model,
                messages=synth_messages,
                output="".join(synth_text),
                usage=usage,
                duration_ms=synth_duration_ms,
                status="ok",
                cost_usd=synth_cost,
                finish_reason=finish,
            )
        total_usage = usage
        for outcome in outcomes:
            total_usage = total_usage + outcome.usage
        total_cost = sum(o.cost_usd for o in outcomes) + synth_cost
        _publish_terminal(
            ProgressEvent(
                kind="completed",
                ensemble=plan.ensemble,
                status=finish,
                cost_usd=total_cost,
                preview=_preview("".join(synth_text)),
            ),
        )
        log.info(
            "run completed",
            status=finish,
            members_ok=sum(1 for o in outcomes if o.ok),
            members_total=members_total,
            synthesis_ms=round(synth_duration_ms, 1),
            total_tokens=total_usage.total_tokens,
            total_cost_usd=round(total_cost, 6),
            elapsed_seconds=round(deps.clock.now() - run_started, 2),
        )
        yield Completed(finish_reason=finish, usage=total_usage, total_cost_usd=total_cost)
    except MomError as exc:
        if synthesizing:
            # A failed synthesis call used to be recorded nowhere at all — see
            # _record_synth_failure. exc.kind/exc.detail/exc.attempts are the adapter's classified
            # triple (or the class-level "unknown"/None/1 default for a MomError that isn't an
            # UpstreamError, e.g. QuorumNotMet, which can't reach here since it's raised before
            # `synthesizing` is set).
            _record_synth_failure(
                deps,
                plan,
                turn_type,
                error=exc.safe_message,
                error_kind=exc.kind,
                error_detail=exc.detail,
                attempts=exc.attempts,
                duration_ms=_elapsed_ms(deps, synth_started),
            )
        _publish_terminal(
            ProgressEvent(kind="failed", ensemble=plan.ensemble, detail=exc.safe_message)
        )
        # safe_message, never exc.detail: the client-safe half is the one that is safe to log
        # unconditionally. Covers QuorumNotMet and synthesis failures, previously logged nowhere.
        log.warning(
            "run failed",
            code=exc.code,
            error_kind=exc.kind,
            error=exc.safe_message,
            synthesizing=synthesizing,
            elapsed_seconds=round(deps.clock.now() - run_started, 2),
        )
        yield PipelineFailed(code=exc.code, message=exc.safe_message, http_status=exc.http_status)
    except Exception as exc:
        if synthesizing:
            _record_synth_failure(
                deps,
                plan,
                turn_type,
                error="internal error",
                error_kind="unknown",
                error_detail=None,
                attempts=1,
                duration_ms=_elapsed_ms(deps, synth_started),
            )
        _publish_terminal(
            ProgressEvent(kind="failed", ensemble=plan.ensemble, detail="internal error")
        )
        # An internal bug here yields PipelineFailed without ever reaching the API error handler,
        # so this is the only place it can be seen. Type and source location, NOT exc_info: see
        # _exc_site for why a traceback can't be logged here.
        log.error(
            "run failed",
            code="internal_error",
            error_type=type(exc).__name__,
            error_site=_exc_site(exc),
            elapsed_seconds=round(deps.clock.now() - run_started, 2),
        )
        yield PipelineFailed(code="internal_error", message="Internal server error")
    finally:
        if not terminal_published:
            # Some path left without ever publishing completed/failed — most likely the
            # generator was torn down (a client disconnect propagates as GeneratorExit at
            # whatever `yield` was in flight). Without this, a progress-dashboard tab watching
            # the SAME request_id from a second connection would sit open until the bus's TTL
            # eventually evicts the channel, rather than resolving when the request actually
            # ended. `_publish` (not `_publish_terminal`) — no further reporting needed here.
            _publish(
                deps,
                ProgressEvent(kind="failed", ensemble=plan.ensemble, detail="client disconnected"),
            )
            # Log it too, with how long the client actually waited. A disconnect is invisible in
            # the metrics DB (the turn records no synthesis row — it simply stops), so without
            # this the only trace is a progress event that ages out of the bus within the hour.
            # The elapsed number is the diagnostic: it is a client/proxy read-timeout, and which
            # one is a question of *which* value it lands on, every time, across requests.
            log.warning(
                "run torn down before it finished — client disconnected",
                elapsed_seconds=round(deps.clock.now() - run_started, 1),
                reached_synthesis=synthesizing,
            )


async def collect(events: AsyncIterator[StreamEvent]) -> EnsembleResult:
    """Drain an event stream into a single result (raises on a terminal failure)."""
    text: list[str] = []
    reasoning: list[str] = []
    outcomes: list[ModelOutcome] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = Usage()
    cost = 0.0
    finish = "stop"
    async for event in events:
        if isinstance(event, MemberCompleted):
            outcomes.append(event.outcome)
        elif isinstance(event, AnswerDelta):
            if event.content:
                text.append(event.content)
            if event.reasoning:
                reasoning.append(event.reasoning)
        elif isinstance(event, ToolCallStarted):
            tool_calls[event.index] = {
                "id": event.call_id,
                "type": "function",
                "function": {"name": event.name, "arguments": ""},
            }
        elif isinstance(event, ToolCallDelta):
            if event.index in tool_calls:
                tool_calls[event.index]["function"]["arguments"] += event.arguments_fragment
        elif isinstance(event, Completed):
            finish, usage, cost = event.finish_reason, event.usage, event.total_cost_usd
        elif isinstance(event, PipelineFailed):
            # Preserve the failure's HTTP status/code so streaming and non-streaming agree.
            err = UpstreamError(event.message)
            err.http_status = event.http_status
            err.code = event.code
            raise err
    return EnsembleResult(
        text="".join(text),
        outcomes=tuple(outcomes),
        usage=usage,
        total_cost_usd=cost,
        finish_reason=finish,
        tool_calls=tuple(tool_calls[i] for i in sorted(tool_calls)),
        reasoning="".join(reasoning),
    )
