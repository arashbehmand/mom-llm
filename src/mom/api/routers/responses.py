"""``POST /v1/responses`` (OpenAI Responses API — stateless subset)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.responses import build_response, encode_sse
from mom.api.reqid import (
    PROGRESS_URL_HEADER,
    REQUEST_ID_HEADER,
    resolve_request_id,
    response_headers,
)
from mom.api.runs import resolve_events
from mom.api.schemas.responses import ResponsesRequest
from mom.api.sse import sse_response
from mom.engine.pipeline import collect
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
    events, leader_request_id = resolve_events(container, plan, ir, request_id)
    headers = response_headers(http_request, leader_request_id, container)
    if leader_request_id != request_id:
        headers["X-MoM-Coalesced"] = "1"
    response_id = container.ids.new_id("resp")
    created = int(container.clock.now())
    if req.stream:
        stream = encode_sse(
            events,
            response_id=response_id,
            model=ir.model,
            created=created,
            show_work=plan.show_work,
            progress_url=headers[PROGRESS_URL_HEADER],
            notices=plan.notices,
        )
        heartbeat = container.catalog.config.server.stream_heartbeat
        heartbeat_seconds = heartbeat.total_seconds() if heartbeat is not None else None
        return sse_response(stream, headers=headers, heartbeat_seconds=heartbeat_seconds)
    result = await collect(events)
    return JSONResponse(
        build_response(
            result,
            response_id=response_id,
            model=ir.model,
            created=created,
            show_work=plan.show_work,
            progress_url=headers[PROGRESS_URL_HEADER],
            notices=plan.notices,
        ),
        headers=headers,
    )
