"""The one orchestration pipeline. Streaming and non-streaming are two consumers of it.

``run_ensemble`` emits a typed event stream; ``collect`` drains it into an ``EnsembleResult``.
There is no second code path, so the two modes cannot drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mom.domain.cost import compute_cost
from mom.domain.errors import MomError, UpstreamError, UpstreamTimeout
from mom.domain.events import (
    AnswerDelta,
    Completed,
    FanoutSkipped,
    FanoutStarted,
    FinishReason,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.metrics import CallMetric, MetricsSink, TurnType
from mom.domain.ports import CallSpec, Clock, CompletionChunk, LLMClient, Tracer
from mom.domain.prompt_caching import inject_anthropic_cache
from mom.domain.results import EnsembleResult, ModelOutcome, OutcomeStatus, Usage
from mom.domain.synthesis import all_failed_message, build_synthesis_messages
from mom.engine.plan import ExecutionPlan, PlannedMember
from mom.runtime.logging import get_logger


logger = get_logger("mom.engine")


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    client: LLMClient
    clock: Clock
    recorder: MetricsSink | None = None
    tracer: Tracer | None = None
    request_id: str = ""


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
            status="ok" if outcome.ok else "error",
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
        )
    )


def _record_synth(
    deps: PipelineDeps, plan: ExecutionPlan, usage: Usage, cost: float, turn_type: TurnType
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
            turn_type=turn_type,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_prompt_tokens=usage.cached_prompt_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
        )
    )


def _coerce_finish(value: str | None) -> FinishReason:
    match value:
        case "stop" | "length" | "tool_calls" | "content_filter" | "error":
            return value
        case _:
            return "stop"


async def _run_member(deps: PipelineDeps, member: PlannedMember) -> ModelOutcome:
    start = deps.clock.now()
    common = {"identity": member.identity, "llm": member.identity, "model": member.spec.model}
    try:
        timeout = member.spec.timeout_seconds
        completion = await asyncio.wait_for(deps.client.complete(member.spec), timeout=timeout)
    except TimeoutError:
        return ModelOutcome(**common, status="timeout", error="timed out")  # type: ignore[arg-type]
    except MomError as exc:
        return ModelOutcome(**common, status="error", error=exc.safe_message)  # type: ignore[arg-type]
    except Exception as exc:
        # Never surface raw provider/third-party text (it can carry keys, URLs, internal paths);
        # log it for the operator and return a safe, generic message.
        logger.warning("member call failed", llm=member.identity, error=str(exc))
        return ModelOutcome(**common, status="error", error="call failed")  # type: ignore[arg-type]
    duration_ms = (deps.clock.now() - start) * 1000.0
    status: OutcomeStatus = "ok" if completion.content.strip() else "empty"
    # Config pricing wins; otherwise fall back to the adapter's litellm cost. Cache hits cost $0.
    if completion.cached:
        cost = 0.0
    elif member.pricing is not None:
        cost = compute_cost(completion.usage, member.pricing)
    else:
        cost = completion.cost_usd or 0.0
    return ModelOutcome(
        **common,
        status=status,
        content=completion.content,
        reasoning=completion.reasoning,
        usage=completion.usage,
        cached=completion.cached,
        duration_ms=duration_ms,
        cost_usd=cost,
    )


async def _fan_out(deps: PipelineDeps, plan: ExecutionPlan) -> AsyncIterator[StreamEvent]:
    for member in plan.members:
        yield FanoutStarted(member.identity, member.spec.model)
    sem = asyncio.Semaphore(plan.max_concurrency)

    async def run(member: PlannedMember) -> ModelOutcome:
        async with sem:  # cap concurrent upstream calls
            return await _run_member(deps, member)

    tasks = [asyncio.create_task(run(m)) for m in plan.members]
    loop = asyncio.get_running_loop()
    deadline = None if plan.fanout_deadline is None else loop.time() + plan.fanout_deadline
    pending = set(tasks)
    try:
        while pending:
            timeout = None if deadline is None else max(0.0, deadline - loop.time())
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:  # deadline elapsed with nothing new — abandon the stragglers
                logger.warning("fan-out deadline exceeded", ensemble=plan.ensemble)
                break
            for task in done:
                yield MemberCompleted(await task)
    finally:
        for task in tasks:
            if not task.done():
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
        timeout_seconds=plan.synth.timeout_seconds,
    )


async def run_ensemble(plan: ExecutionPlan, deps: PipelineDeps) -> AsyncIterator[StreamEvent]:
    """Run an ensemble, emitting a typed event stream. Never raises — failures are events."""
    turn_type: TurnType = "relay" if plan.skip_reason == "tool_continuation" else "ensemble"
    try:
        outcomes: list[ModelOutcome] = []
        if plan.skip_fanout:
            if plan.skip_reason is not None:
                yield FanoutSkipped(plan.skip_reason)
            synth_messages = plan.client_messages
        else:
            async for event in _fan_out(deps, plan):
                if isinstance(event, MemberCompleted):
                    outcome = event.outcome
                    outcomes.append(outcome)
                    _record_member(deps, plan, outcome, turn_type)
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
                        )
                yield event
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
        usage = Usage()
        synth_llm_cost: float | None = None
        finish: FinishReason = "stop"
        started_tools: set[int] = set()
        synth_text: list[str] = []
        synth_stream = deps.client.stream(_synth_spec(plan, synth_messages))
        async for chunk in _stream_with_timeout(synth_stream, plan.synth.timeout_seconds):
            if chunk.content is not None or chunk.reasoning is not None:
                yield AnswerDelta(content=chunk.content, reasoning=chunk.reasoning)
                if chunk.content:
                    synth_text.append(chunk.content)
            if chunk.tool_call is not None:
                call = chunk.tool_call
                index = int(call.get("index", 0))
                if index not in started_tools:
                    started_tools.add(index)
                    yield ToolCallStarted(
                        index=index,
                        call_id=str(call.get("id") or ""),
                        name=str(call.get("name") or ""),
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

        if plan.synth.pricing is not None:
            synth_cost = compute_cost(usage, plan.synth.pricing)
        else:
            synth_cost = synth_llm_cost or 0.0
        _record_synth(deps, plan, usage, synth_cost, turn_type)
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
                duration_ms=0.0,
            )
        total_usage = usage
        for outcome in outcomes:
            total_usage = total_usage + outcome.usage
        total_cost = sum(o.cost_usd for o in outcomes) + synth_cost
        yield Completed(finish_reason=finish, usage=total_usage, total_cost_usd=total_cost)
    except MomError as exc:
        yield PipelineFailed(code=exc.code, message=exc.safe_message, http_status=exc.http_status)
    except Exception:
        yield PipelineFailed(code="internal_error", message="Internal server error")


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
