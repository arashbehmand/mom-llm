"""``POST /v1/responses`` (OpenAI Responses API — stateless subset)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.responses import build_response, encode_sse
from mom.api.reqid import (
    PROGRESS_URL_HEADER,
    REQUEST_ID_HEADER,
    resolve_request_id,
    response_headers,
)
from mom.api.schemas.responses import ResponsesRequest
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/responses")
async def responses(
    req: ResponsesRequest, container: ContainerDep, http_request: Request
) -> object:
    from mom.api.translate_responses import responses_request_to_ir

    ir = responses_request_to_ir(req, stream=req.stream)
    plan = resolve_plan(container.catalog, ir)
    request_id = resolve_request_id(http_request.headers.get(REQUEST_ID_HEADER), container.ids)
    headers = response_headers(http_request, request_id, container)
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
    response_id = container.ids.new_id("resp")
    created = int(container.clock.now())
    if req.stream:
        stream = encode_sse(
            run_ensemble(plan, deps),
            response_id=response_id,
            model=ir.model,
            created=created,
            show_work=plan.show_work,
            progress_url=headers[PROGRESS_URL_HEADER],
        )
        return StreamingResponse(stream, media_type="text/event-stream", headers=headers)
    result = await collect(run_ensemble(plan, deps))
    return JSONResponse(
        build_response(
            result,
            response_id=response_id,
            model=ir.model,
            created=created,
            show_work=plan.show_work,
            progress_url=headers[PROGRESS_URL_HEADER],
        ),
        headers=headers,
    )
