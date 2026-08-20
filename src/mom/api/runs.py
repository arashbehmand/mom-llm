"""Starting (or joining) the pipeline run behind a request.

The ONE place all three request surfaces — chat completions, Responses, Anthropic messages — turn
a resolved plan into an event stream, for the same reason ``sse.py`` is the one place they build
their streaming response: coalescing used to live in ``chat.py`` alone, so `/v1/responses` and
`/v1/messages` could not dedupe at all no matter how the deployment was configured. That is
exactly backwards for the client that motivated the feature — lobe-chat talks to `/v1/responses`,
and it is the one observed sending duplicate full-turn retries.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from mom.api.deps import Container
from mom.domain.events import StreamEvent
from mom.domain.request import ChatRequestIR
from mom.domain.requestkey import request_identity
from mom.engine.pipeline import PipelineDeps, run_ensemble
from mom.engine.plan import ExecutionPlan
from mom.runtime.logging import get_logger


logger = get_logger("mom.api.runs")


def pipeline_deps(container: Container, request_id: str) -> PipelineDeps:
    return PipelineDeps(
        client=container.client,
        clock=container.clock,
        recorder=container.metrics,
        tracer=container.tracer,
        bus=container.bus,
        request_id=request_id,
        ids=container.ids,
        custody=container.custody,
    )


def resolve_events(
    container: Container, plan: ExecutionPlan, ir: ChatRequestIR, request_id: str
) -> tuple[AsyncIterator[StreamEvent], str]:
    """Run the ensemble, or attach to an identical in-flight run when ``plan.dedupe``.

    Returns ``(events, leader_request_id)``: the second element equals ``request_id`` unless this
    call coalesced onto a run someone else started, in which case it's that original caller's id
    (see ``CoalesceRegistry.attach``) — which is what the caller reports in its response headers,
    so a coalesced follower's progress link points at the run actually doing the work.
    """

    def run() -> AsyncIterator[StreamEvent]:
        return run_ensemble(plan, pipeline_deps(container, request_id))

    # `plan.dedupe` is the per-request decision (config default + `<<SYSTEM>> dedupe:`); the None
    # check is for a container built without a registry at all (tests, embedded use).
    if container.coalesce is None or not plan.dedupe:
        return run(), request_id
    identity = request_identity(ir)
    events, leader_request_id = container.coalesce.attach(identity, request_id, run)
    if leader_request_id != request_id:
        logger.info(
            "request coalesced onto in-flight run",
            request_id=request_id,
            leader_request_id=leader_request_id,
            ensemble=plan.ensemble,
        )
    return events, leader_request_id
