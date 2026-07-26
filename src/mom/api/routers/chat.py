"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.chat import ChatFrame, build_completion, encode_sse
from mom.api.schemas.openai_chat import ChatCompletionRequest
from mom.api.translate import chat_request_to_ir
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, container: ContainerDep) -> object:
    ir = chat_request_to_ir(req)
    plan = resolve_plan(container.catalog, ir)  # MomError -> handled by exception handler
    deps = PipelineDeps(
        client=container.client,
        clock=container.clock,
        recorder=container.metrics,
        tracer=container.tracer,
        request_id=container.ids.new_id("req"),
    )
    frame = ChatFrame(
        id=container.ids.new_id("chatcmpl-mom"),
        created=int(container.clock.now()),
        model=ir.model,
    )
    if ir.stream:
        stream = encode_sse(
            run_ensemble(plan, deps),
            frame,
            show_work=plan.show_work,
            include_usage=ir.include_usage,
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    result = await collect(run_ensemble(plan, deps))
    response = build_completion(result, frame, show_work=plan.show_work)
    return JSONResponse(response.model_dump())


models_router = APIRouter(dependencies=[Depends(require_api_key)])


@models_router.get("/models")
async def list_models(container: ContainerDep) -> dict[str, object]:
    created = int(container.clock.now())
    data = [
        {"id": name, "object": "model", "created": created, "owned_by": "mom"}
        for name in container.catalog.ensembles
    ]
    return {"object": "list", "data": data}
