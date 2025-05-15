import json
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import load_config
from .models import (
    ChatMessage,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionResponseChoice,
)

config = load_config()
openai_router = APIRouter(prefix="/v1", tags=["OpenAI"])


def check_token(request: Request):
    api_token = os.getenv("API_TOKEN")
    token = request.headers.get("Authorization", "")
    if api_token and token.replace("Bearer ", "") != api_token:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid or missing API token.",
                "type": "authentication_error",
            },
        )


@openai_router.get("/models")
async def get_openai_models():
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
async def chat_completions_openai(
    req_data: OpenAIChatCompletionRequest, request: Request
):
    check_token(request)
    model_conf_check = next(
        (m for m in config.models if m.name == req_data.model), None
    )
    if not model_conf_check:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{req_data.model}' not found.",
                "type": "invalid_request_error",
            },
        )

    from ..main import _process_mom_chat_request, logger

    if getattr(req_data, "stream", False):

        async def event_stream():
            response_id = f"mom-oai-{req_data.model}-{str(uuid.uuid4())}"
            index = 0
            try:
                # _process_mom_chat_request is an async def function that returns an async generator object when stream=True.
                # Await the async generator function call to get the async generator object
                the_generator = await _process_mom_chat_request(
                    req_data.model,
                    [m.dict(exclude_none=True) for m in req_data.messages],
                    request,
                    stream=True,
                )
                async for chunk in the_generator:
                    # Ensure chunk is a dictionary before processing
                    if isinstance(chunk, dict):
                        # Check if it's a LiteLLM chunk with choices
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            delta = choice.get("delta")
                            finish_reason = choice.get("finish_reason")

                            # Handle content delta
                            if delta and delta.get("content") is not None:
                                data = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req_data.model,
                                    "choices": [
                                        {
                                            "index": index,
                                            "delta": {"content": delta["content"]},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(data)}\n\n"

                            # Handle finish reason
                            if finish_reason is not None:
                                data = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req_data.model,
                                    "choices": [
                                        {
                                            "index": index,
                                            "delta": {},
                                            "finish_reason": finish_reason,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                        # Handle potential error chunks from LiteLLM or internal errors
                        elif "error" in chunk:
                            error_data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": req_data.model,
                                "error": chunk["error"],
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"
                        else:
                            # Log unexpected chunk format
                            logger.warning(f"Received unexpected chunk format: {chunk}")
                    else:
                        # Log unexpected chunk type
                        logger.warning(f"Received unexpected chunk type: {type(chunk)}")

            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                error_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req_data.model,
                    "error": {"message": str(e), "type": "server_error"},
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming (default) path
    try:
        (
            final_content,
            raw_thinking_ctx,
            concluding_usage,
            total_cost,
            mom_model_name_used,
            thinking_embedded,
            trace_obj,
        ) = await _process_mom_chat_request(
            req_data.model,
            [m.dict(exclude_none=True) for m in req_data.messages],
            request,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500, detail={"message": str(e), "type": "internal_server_error"}
        )

    response_id = f"mom-oai-{mom_model_name_used}-{str(uuid.uuid4())}"
    openai_response = OpenAIChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=mom_model_name_used,
        choices=[
            OpenAIChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content=final_content)
            )
        ],
        usage=concluding_usage,
        thinking_context=None if thinking_embedded else raw_thinking_ctx,
        total_cost_usd=total_cost if total_cost > 0 else None,
    )

    if trace_obj:
        try:
            trace_obj.update(output=openai_response.dict(exclude_none=True))
        except Exception as e:
            logger.error(f"OpenAI Endpoint: Failed to update Langfuse trace: {e}")

    return openai_response
