"""
OpenAI-compatible API endpoints for the MoM service.

This module provides REST endpoints that follow the OpenAI Chat Completion API format,
making the MoM service compatible with existing OpenAI client libraries and tools.

Endpoints:
- GET /v1/models: List available MoM models
- POST /v1/chat/completions: Create a chat completion (supports streaming)

All endpoints require Bearer token authentication via the Authorization header.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import verify_bearer_token
from ..config import load_config
from ..events import MoMEvent
from .models import (
    ChatMessage,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionResponseChoice,
)

config = load_config()
openai_router = APIRouter(prefix="/v1", tags=["OpenAI"])


def _build_progress_url(request: Request) -> str | None:
    reporting_base_url = os.getenv("REPORTING_SERVICE_URL", "").strip()
    request_id = getattr(request.state, "request_id", None)
    if not reporting_base_url or not request_id:
        return None
    return f"{reporting_base_url}/progress/{request_id}"


async def _publish_progress_event(request: Request, event_type: str, data: dict[str, Any]) -> None:
    """Publish progress/status events for reporting when Redis publisher is available."""
    request_id = getattr(request.state, "request_id", None)
    redis_publisher = getattr(request.state, "redis_publisher", None)
    if not request_id or not redis_publisher or not redis_publisher.enabled:
        return

    await redis_publisher.publish(
        MoMEvent(
            type=event_type,
            request_id=request_id,
            timestamp=time.time(),
            data=data,
        )
    )


@openai_router.get("/models")
async def get_openai_models(request: Request, client_version: str | None = None):
    verify_bearer_token(request)
    # Codex CLI's model-picker refresh calls GET /v1/models?client_version=<v> and
    # expects its own catalog shape, {"models": [...]}. Handed OpenAI's list shape it
    # logs `failed to refresh available models: missing field 'models'` and falls back
    # to its bundled model metadata. Presence of the param switches the response shape;
    # absent -> OpenAI shape, unchanged for every other client.
    #
    # We answer with a valid but *empty* catalog rather than one entry per MoM model.
    # Codex requires every entry to carry `base_instructions` and uses whatever it
    # receives verbatim as that model's system prompt (protocol/src/openai_models.rs ::
    # deserialize_model_infos_with_legacy_base). MoM has no business owning Codex's
    # agent prompt, and a placeholder would silently replace it -- measured against
    # codex-cli 0.147.0, sending `"base_instructions": ""` dropped the whole 20.7 KB
    # prompt from Codex's request. An empty catalog clears the error and leaves Codex
    # on its own metadata, which is what it already uses today.
    if client_version is not None:
        return {"models": []}
    data = []
    for m_config in config.models:
        data.append(
            {
                "id": m_config.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "MoM-Service",
                "permission": [],
                "root": m_config.name,
                "parent": None,
            }
        )
    return {"object": "list", "data": data}


@openai_router.post("/chat/completions", response_model=OpenAIChatCompletionResponse)
async def chat_completions_openai(req_data: OpenAIChatCompletionRequest, request: Request):
    verify_bearer_token(request)
    from ..main import get_mom_model_config

    model_conf_check = get_mom_model_config(req_data.model)
    if not model_conf_check:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{req_data.model}' not found.",
                "type": "invalid_request_error",
            },
        )

    from ..main import _process_mom_chat_request, logger

    # Streaming path
    if req_data.stream:

        async def event_stream():
            response_id = f"mom-oai-{req_data.model}-{str(uuid.uuid4())}"
            complete_content = ""
            trace_obj = None
            send_done = True

            try:
                the_generator = await _process_mom_chat_request(
                    req_data.model,
                    [m.model_dump(exclude_none=True) for m in req_data.messages],
                    request,
                    stream=True,
                )

                if hasattr(request.state, "trace_obj"):
                    trace_obj = request.state.trace_obj

                async for chunk_dict in the_generator:
                    if isinstance(chunk_dict, dict):
                        if (
                            "choices" in chunk_dict
                            and isinstance(chunk_dict["choices"], list)
                            and len(chunk_dict["choices"]) > 0
                        ):
                            choice = chunk_dict["choices"][0]
                            delta = choice.get("delta")

                            if delta and delta.get("content") is not None:
                                complete_content += delta["content"]

                            yield f"data: {json.dumps(chunk_dict)}\n\n"

                        elif "error" in chunk_dict:
                            yield f"data: {json.dumps(chunk_dict)}\n\n"
                        else:
                            logger.warning(
                                f"Skipping unexpected chunk format from _process_mom_chat_request: {chunk_dict}"
                            )
                    else:
                        logger.warning(
                            f"Received unexpected chunk type from _process_mom_chat_request: {type(chunk_dict)}"
                        )

                if trace_obj and complete_content:
                    try:
                        openai_response_for_trace = OpenAIChatCompletionResponse(
                            id=response_id,
                            created=int(time.time()),
                            model=req_data.model,
                            choices=[
                                OpenAIChatCompletionResponseChoice(
                                    index=0,
                                    message=ChatMessage(role="assistant", content=complete_content),
                                )
                            ],
                        )
                        trace_obj.update(
                            output=openai_response_for_trace.model_dump(exclude_none=True)
                        )
                        logger.info("Successfully updated Langfuse trace for streaming response")
                    except Exception as e:
                        logger.error(f"Failed to update Langfuse trace for streaming response: {e}")

            except asyncio.CancelledError:
                send_done = False
                logger.info(
                    "Streaming client disconnected. request_id=%s",
                    getattr(request.state, "request_id", "unknown"),
                )
                await _publish_progress_event(
                    request,
                    "request_aborted",
                    {
                        "reason": "Client disconnected before completion",
                        "stage": "streaming",
                    },
                )
                if trace_obj:
                    try:
                        trace_obj.update(
                            level="ERROR",
                            status_message="Client disconnected before streaming completed",
                        )
                    except Exception as trace_error:
                        logger.error(
                            "Failed to update Langfuse trace on disconnect: %s", trace_error
                        )
                raise
            except Exception as e:
                logger.error(f"Error in streaming response generator: {str(e)}", exc_info=True)
                if trace_obj:
                    try:
                        trace_obj.update(
                            level="ERROR",
                            status_message=f"Streaming request failed: {str(e)}",
                        )
                    except Exception as trace_error:
                        logger.error(
                            "Failed to update Langfuse trace on streaming error: %s", trace_error
                        )
                await _publish_progress_event(
                    request,
                    "error",
                    {"error": str(e), "stage": "streaming"},
                )
                error_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req_data.model,
                    "error": {"message": str(e), "type": "server_error"},
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            if send_done:
                yield "data: [DONE]\n\n"

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
        progress_url = _build_progress_url(request)
        if progress_url:
            response.headers["X-MoM-Progress-Url"] = progress_url
        return response

    # Non-streaming path
    try:
        (
            final_content,
            thinking_context,
            usage,
            total_cost,
            mom_model_name,
            _thinking_embedded,
            _trace_obj,
        ) = await _process_mom_chat_request(
            req_data.model,
            [m.model_dump(exclude_none=True) for m in req_data.messages],
            request,
            stream=False,
        )

        response_id = f"mom-oai-{req_data.model}-{str(uuid.uuid4())}"
        response = OpenAIChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=mom_model_name,
            choices=[
                OpenAIChatCompletionResponseChoice(
                    index=0, message=ChatMessage(role="assistant", content=final_content or "")
                )
            ],
            usage=usage,
            thinking_context=thinking_context,
            total_cost_usd=total_cost,
        )
        resp_payload = response.model_dump(exclude_none=True)
    except HTTPException:
        raise
    except Exception as e:
        trace_obj = getattr(request.state, "trace_obj", None)
        if trace_obj:
            try:
                trace_obj.update(
                    level="ERROR",
                    status_message=f"Non-streaming request failed: {str(e)}",
                )
            except Exception as trace_error:
                logger.error(
                    "Failed to update Langfuse trace on non-streaming error: %s", trace_error
                )
        await _publish_progress_event(
            request,
            "error",
            {"error": str(e), "stage": "non_streaming"},
        )
        raise HTTPException(status_code=500, detail={"message": str(e), "type": "server_error"})
    response = JSONResponse(content=resp_payload)
    progress_url = _build_progress_url(request)
    if progress_url:
        response.headers["X-MoM-Progress-Url"] = progress_url
    return response
