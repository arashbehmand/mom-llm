"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.chat import ChatFrame, build_completion, encode_sse, resolve_stream_profile
from mom.api.reqid import REQUEST_ID_HEADER, resolve_request_id
from mom.api.schemas.openai_chat import ChatCompletionRequest
from mom.api.sse import with_heartbeat
from mom.api.translate import chat_request_to_ir
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, container: ContainerDep, http_request: Request
) -> object:
    ir = chat_request_to_ir(req)
    plan = resolve_plan(container.catalog, ir)  # MomError -> handled by exception handler
    request_id = resolve_request_id(http_request.headers.get(REQUEST_ID_HEADER), container.ids)
    headers = {"X-Request-Id": request_id}
    deps = PipelineDeps(
        client=container.client,
        clock=container.clock,
        recorder=container.metrics,
        tracer=container.tracer,
        bus=container.bus,
        request_id=request_id,
        ids=container.ids,
        custody=container.custody,
    )
    frame = ChatFrame(
        id=container.ids.new_id("chatcmpl-mom"),
        created=int(container.clock.now()),
        model=ir.model,
    )
    if ir.stream:
        user_agent = http_request.headers.get("user-agent")
        profile = resolve_stream_profile(plan.stream_profile, user_agent)
        stream = encode_sse(
            run_ensemble(plan, deps),
            frame,
            show_work=plan.show_work,
            include_usage=ir.include_usage,
            stream_profile=profile,
        )
        heartbeat = container.catalog.config.server.stream_heartbeat
        if heartbeat is not None:  # keepalive comments so a slow fan-out doesn't idle-timeout
            stream = with_heartbeat(stream, heartbeat.total_seconds())
        return StreamingResponse(stream, media_type="text/event-stream", headers=headers)
    result = await collect(run_ensemble(plan, deps))
    response = build_completion(result, frame, show_work=plan.show_work)
    return JSONResponse(response.model_dump(), headers=headers)
