"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from mom.api.auth import require_api_key
from mom.api.deps import Container, ContainerDep
from mom.api.encoders.chat import ChatFrame, build_completion, encode_sse, resolve_stream_profile
from mom.api.reqid import (
    PROGRESS_URL_HEADER,
    REQUEST_ID_HEADER,
    resolve_request_id,
    response_headers,
)
from mom.api.schemas.openai_chat import ChatCompletionRequest
from mom.api.sse import sse_response
from mom.api.translate import chat_request_to_ir
from mom.domain.events import StreamEvent
from mom.domain.request import ChatRequestIR, ToolChoice
from mom.domain.requestkey import request_identity
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import ExecutionPlan, resolve_plan
from mom.runtime.logging import get_logger


router = APIRouter(dependencies=[Depends(require_api_key)])
logger = get_logger("mom.api.chat")


def _tool_choice_repr(choice: ToolChoice) -> str:
    return choice if isinstance(choice, str) else f"function:{choice.name}"


def _pipeline_deps(container: Container, request_id: str) -> PipelineDeps:
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


def _resolve_events(
    container: Container, plan: ExecutionPlan, ir: ChatRequestIR, request_id: str
) -> tuple[AsyncIterator[StreamEvent], str]:
    """Run the ensemble, or — when ``server.dedupe.enabled`` — attach to an identical in-flight
    run instead. Returns ``(events, leader_request_id)``: the second element equals
    ``request_id`` unless this call coalesced onto a run someone else started, in which case it's
    that original caller's id (see ``CoalesceRegistry.attach``)."""

    def run() -> AsyncIterator[StreamEvent]:
        return run_ensemble(plan, _pipeline_deps(container, request_id))

    if container.coalesce is None:
        return run(), request_id
    identity = request_identity(ir)
    events, leader_request_id = container.coalesce.attach(identity, request_id, run)
    if leader_request_id != request_id:
        logger.info(
            "chat request coalesced onto in-flight run",
            request_id=request_id,
            leader_request_id=leader_request_id,
        )
    return events, leader_request_id


@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, container: ContainerDep, http_request: Request
) -> object:
    ir = chat_request_to_ir(req)
    plan = resolve_plan(container.catalog, ir)  # MomError -> handled by exception handler
    request_id = resolve_request_id(http_request.headers.get(REQUEST_ID_HEADER), container.ids)
    logger.debug(
        "chat request",
        request_id=request_id,
        model=ir.model,
        stream=ir.stream,
        effort=ir.effort,
        web_search=ir.web_search,
        tools=[t.name for t in ir.tools],
        tool_choice=_tool_choice_repr(ir.tool_choice),
        response_format=ir.response_format is not None,
    )
    events, leader_request_id = _resolve_events(container, plan, ir, request_id)
    coalesced = leader_request_id != request_id
    headers = response_headers(http_request, leader_request_id, container)
    if coalesced:
        headers["X-MoM-Coalesced"] = "1"
    frame = ChatFrame(
        id=container.ids.new_id("chatcmpl-mom"),
        created=int(container.clock.now()),
        model=ir.model,
    )
    if ir.stream:
        user_agent = http_request.headers.get("user-agent")
        profile = resolve_stream_profile(plan.stream_profile, user_agent)
        stream = encode_sse(
            events,
            frame,
            show_work=plan.show_work,
            include_usage=ir.include_usage,
            stream_profile=profile,
            progress_url=headers[PROGRESS_URL_HEADER],
        )
        heartbeat = container.catalog.config.server.stream_heartbeat
        heartbeat_seconds = heartbeat.total_seconds() if heartbeat is not None else None
        return sse_response(stream, headers=headers, heartbeat_seconds=heartbeat_seconds)
    result = await collect(events)
    response = build_completion(
        result, frame, show_work=plan.show_work, progress_url=headers[PROGRESS_URL_HEADER]
    )
    return JSONResponse(response.model_dump(), headers=headers)
