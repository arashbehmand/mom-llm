"""``POST /v1/responses`` (OpenAI Responses API — stateless subset)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.responses import build_response, encode_sse
from mom.api.schemas.responses import ResponsesRequest
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/responses")
async def responses(req: ResponsesRequest, container: ContainerDep) -> object:
    from mom.api.translate_responses import responses_request_to_ir

    ir = responses_request_to_ir(req, stream=req.stream)
    plan = resolve_plan(container.catalog, ir)
    deps = PipelineDeps(
        client=container.client,
        clock=container.clock,
        recorder=container.metrics,
        tracer=container.tracer,
        request_id=container.ids.new_id("req"),
    )
    response_id = container.ids.new_id("resp")
    created = int(container.clock.now())
    if req.stream:
        stream = encode_sse(
            run_ensemble(plan, deps), response_id=response_id, model=ir.model, created=created
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    result = await collect(run_ensemble(plan, deps))
    return JSONResponse(
        build_response(result, response_id=response_id, model=ir.model, created=created)
    )
