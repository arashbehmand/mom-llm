"""The one orchestration pipeline. Streaming and non-streaming are two consumers of it.

``run_ensemble`` emits a typed event stream; ``collect`` drains it into an ``EnsembleResult``.
There is no second code path, so the two modes cannot drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mom.domain.errors import MomError, UpstreamError
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
from mom.domain.ports import CallSpec, Clock, LLMClient
from mom.domain.results import EnsembleResult, ModelOutcome, OutcomeStatus, Usage
from mom.domain.synthesis import all_failed_message, build_synthesis_messages
from mom.engine.plan import ExecutionPlan, PlannedMember


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    client: LLMClient
    clock: Clock
    recorder: MetricsSink | None = None
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
    deps: PipelineDeps, plan: ExecutionPlan, usage: Usage, turn_type: TurnType
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
        return ModelOutcome(**common, status="error", error=str(exc))  # type: ignore[arg-type]
    duration_ms = (deps.clock.now() - start) * 1000.0
    status: OutcomeStatus = "ok" if completion.content.strip() else "empty"
    return ModelOutcome(
        **common,  # type: ignore[arg-type]
        status=status,
        content=completion.content,
        reasoning=completion.reasoning,
        usage=completion.usage,
        cached=completion.cached,
        duration_ms=duration_ms,
    )


async def _fan_out(deps: PipelineDeps, plan: ExecutionPlan) -> AsyncIterator[StreamEvent]:
    for member in plan.members:
        yield FanoutStarted(member.identity, member.spec.model)
    tasks = [asyncio.create_task(_run_member(deps, m)) for m in plan.members]
    try:
        for future in asyncio.as_completed(tasks):
            outcome = await future
            yield MemberCompleted(outcome)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


def _synth_spec(plan: ExecutionPlan, messages: list[dict[str, object]]) -> CallSpec:
    return CallSpec(
        llm_name=plan.synth.llm_name,
        model=plan.synth.model,
        messages=messages,
        params=dict(plan.synth.params),
        api=plan.synth.api,
        proxy_url_env=plan.synth.proxy_url_env,
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
                    outcomes.append(event.outcome)
                    _record_member(deps, plan, event.outcome, turn_type)
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

        yield SynthesisStarted(plan.synth.llm_name, plan.synth.model)
        usage = Usage()
        finish: FinishReason = "stop"
        started_tools: set[int] = set()
        async for chunk in deps.client.stream(_synth_spec(plan, synth_messages)):
            if chunk.content is not None or chunk.reasoning is not None:
                yield AnswerDelta(content=chunk.content, reasoning=chunk.reasoning)
            if chunk.tool_call is not None:
                call = chunk.tool_call
                index = int(call.get("index", 0))
                if index not in started_tools:
                    started_tools.add(index)
                    yield ToolCallStarted(
                        index=index,
                        call_id=str(call.get("id", "")),
                        name=str(call.get("name", "")),
                    )
                fragment = call.get("arguments")
                if fragment:
                    yield ToolCallDelta(index=index, arguments_fragment=str(fragment))
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = _coerce_finish(chunk.finish_reason)

        _record_synth(deps, plan, usage, turn_type)
        total_usage = usage
        for outcome in outcomes:
            total_usage = total_usage + outcome.usage
        total_cost = sum(o.cost_usd for o in outcomes)
        yield Completed(finish_reason=finish, usage=total_usage, total_cost_usd=total_cost)
    except MomError as exc:
        yield PipelineFailed(code=exc.code, message=exc.safe_message, http_status=exc.http_status)
    except Exception:
        yield PipelineFailed(code="internal_error", message="Internal server error")


async def collect(events: AsyncIterator[StreamEvent]) -> EnsembleResult:
    """Drain an event stream into a single result (raises on a terminal failure)."""
    text: list[str] = []
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
            raise UpstreamError(event.message)
    return EnsembleResult(
        text="".join(text),
        outcomes=tuple(outcomes),
        usage=usage,
        total_cost_usd=cost,
        finish_reason=finish,
        tool_calls=tuple(tool_calls[i] for i in sorted(tool_calls)),
    )
