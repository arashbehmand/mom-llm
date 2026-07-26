"""``POST /v1/messages`` and ``POST /v1/messages/count_tokens`` (Anthropic-compatible)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.api.encoders.anthropic import build_message, encode_sse
from mom.api.schemas.anthropic import AnthropicMessage, CountTokensRequest, MessagesRequest
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/messages")
async def messages(req: MessagesRequest, container: ContainerDep) -> object:
    from mom.api.translate_anthropic import messages_request_to_ir

    ir = messages_request_to_ir(req, stream=req.stream)
    plan = resolve_plan(container.catalog, ir)
    deps = PipelineDeps(
        client=container.client,
        clock=container.clock,
        recorder=container.metrics,
        tracer=container.tracer,
        request_id=container.ids.new_id("req"),
    )
    message_id = container.ids.new_id("msg")
    if req.stream:
        stream = encode_sse(run_ensemble(plan, deps), message_id=message_id, model=ir.model)
        return StreamingResponse(stream, media_type="text/event-stream")
    result = await collect(run_ensemble(plan, deps))
    return JSONResponse(build_message(result, message_id=message_id, model=ir.model))


def _message_text(message: AnthropicMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = []
    for block in message.content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            content = block.get("content", "")
            parts.append(content if isinstance(content, str) else json.dumps(content))
        elif block.get("type") == "tool_use":
            parts.append(json.dumps(block.get("input", {})))
    return " ".join(parts)


def _estimate_tokens(req: CountTokensRequest) -> int:
    """A deliberately rough token estimate (~4 chars/token) over the visible transcript.

    Claude Code uses this only for context-window math; documented as an estimate.
    """
    chars = 0
    if isinstance(req.system, str):
        chars += len(req.system)
    elif isinstance(req.system, list):
        chars += sum(len(b.get("text", "")) for b in req.system)
    chars += sum(len(_message_text(m)) for m in req.messages)
    if req.tools:
        chars += len(json.dumps(req.tools))
    return max(1, chars // 4)


@router.post("/messages/count_tokens")
async def count_tokens(req: CountTokensRequest, _container: ContainerDep) -> dict[str, Any]:
    return {"input_tokens": _estimate_tokens(req)}
