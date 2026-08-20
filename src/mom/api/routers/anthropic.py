"""``POST /v1/messages`` and ``POST /v1/messages/count_tokens`` (Anthropic-compatible)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from mom.api.auth import require_api_key
from mom.api.deps import Container, ContainerDep
from mom.api.encoders.anthropic import build_message, encode_sse
from mom.api.reqid import REQUEST_ID_HEADER, resolve_request_id, response_headers
from mom.api.runs import resolve_events
from mom.api.schemas.anthropic import AnthropicMessage, CountTokensRequest, MessagesRequest
from mom.api.sse import sse_response
from mom.config.resolve import ResolvedCatalog
from mom.engine.pipeline import collect
from mom.engine.plan import resolve_plan


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/messages")
async def messages(req: MessagesRequest, container: ContainerDep, http_request: Request) -> object:
    from mom.api.translate_anthropic import messages_request_to_ir

    ir = messages_request_to_ir(req, stream=req.stream)
    plan = resolve_plan(container.catalog, ir)
    request_id = resolve_request_id(http_request.headers.get(REQUEST_ID_HEADER), container.ids)
    events, leader_request_id = resolve_events(container, plan, ir, request_id)
    headers = response_headers(http_request, leader_request_id, container)
    if leader_request_id != request_id:
        headers["X-MoM-Coalesced"] = "1"
    message_id = container.ids.new_id("msg")
    input_tokens = _input_tokens(container, req)
    if req.stream:
        stream = encode_sse(
            events,
            message_id=message_id,
            model=ir.model,
            input_tokens=input_tokens,
        )
        heartbeat = container.catalog.config.server.stream_heartbeat
        heartbeat_seconds = heartbeat.total_seconds() if heartbeat is not None else None
        return sse_response(stream, headers=headers, heartbeat_seconds=heartbeat_seconds)
    result = await collect(events)
    return JSONResponse(
        build_message(result, message_id=message_id, model=ir.model, input_tokens=input_tokens),
        headers=headers,
    )


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


def _estimate_tokens(req: CountTokensRequest | MessagesRequest) -> int:
    """A deliberately rough token estimate (~4 chars/token) over the visible transcript.

    The chars/4 fallback used when no model-aware estimator is available (or it fails). Both
    request shapes expose ``system`` / ``messages`` / ``tools``.
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


def _system_text(system: str | list[dict[str, Any]] | None) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return " ".join(b.get("text", "") for b in system)
    return ""


def _count_messages(req: CountTokensRequest | MessagesRequest) -> list[dict[str, Any]]:
    """Flatten an Anthropic request into OpenAI-shaped messages a tokenizer can count."""
    messages: list[dict[str, Any]] = []
    system = _system_text(req.system)
    if req.tools:  # fold tool schemas into the system text so their tokens are counted too
        tools_text = json.dumps(req.tools)
        system = f"{system}\n{tools_text}" if system else tools_text
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend({"role": m.role, "content": _message_text(m)} for m in req.messages)
    return messages


def _synth_model(catalog: ResolvedCatalog, model: str) -> str | None:
    """The real provider model of an ensemble's synthesizer (for model-aware token counting)."""
    ensemble = catalog.ensembles.get(model)
    if ensemble is None:
        return None
    llm = catalog.llms.get(ensemble.synthesizer.llm)
    return llm.model if llm is not None else None


def _input_tokens(container: Container, req: CountTokensRequest | MessagesRequest) -> int:
    """Model-aware input-token estimate via the injected estimator; chars/4 as a safe fallback.

    Counted against the ensemble's synthesizer model — the model whose context window governs
    the client-visible answer. Anything unresolved (no estimator, unknown ensemble) falls back.
    """
    estimator = container.token_estimator
    model = _synth_model(container.catalog, req.model)
    if estimator is not None and model is not None:
        return max(1, estimator.count(model=model, messages=_count_messages(req)))
    return _estimate_tokens(req)


@router.post("/messages/count_tokens")
async def count_tokens(req: CountTokensRequest, container: ContainerDep) -> dict[str, Any]:
    return {"input_tokens": _input_tokens(container, req)}
