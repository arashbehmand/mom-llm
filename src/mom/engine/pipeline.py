"""The one orchestration pipeline. Streaming and non-streaming are two consumers of it.

``run_ensemble`` emits a typed event stream; ``collect`` drains it into an ``EnsembleResult``.
There is no second code path, so the two modes cannot drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
)
from mom.domain.ports import CallSpec, Clock, LLMClient
from mom.domain.results import EnsembleResult, ModelOutcome, OutcomeStatus, Usage
from mom.domain.synthesis import all_failed_message, build_synthesis_messages
from mom.engine.plan import ExecutionPlan, PlannedMember


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    client: LLMClient
    clock: Clock


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
    try:
        outcomes: list[ModelOutcome] = []
        if plan.strategy == "passthrough":
            yield FanoutSkipped("passthrough")
            synth_messages = plan.client_messages
        else:
            async for event in _fan_out(deps, plan):
                if isinstance(event, MemberCompleted):
                    outcomes.append(event.outcome)
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
        async for chunk in deps.client.stream(_synth_spec(plan, synth_messages)):
            if chunk.content is not None or chunk.reasoning is not None:
                yield AnswerDelta(content=chunk.content, reasoning=chunk.reasoning)
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = _coerce_finish(chunk.finish_reason)

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
    usage = Usage()
    cost = 0.0
    finish = "stop"
    async for event in events:
        if isinstance(event, MemberCompleted):
            outcomes.append(event.outcome)
        elif isinstance(event, AnswerDelta):
            if event.content:
                text.append(event.content)
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
    )
