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
    UsageInfo # Import UsageInfo for potential accumulation (though complex for streaming)
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

    from ..main import _process_mom_chat_request, logger, LANGFUSE_CLIENT # Import LANGFUSE_CLIENT

    # Always return a StreamingResponse based on user feedback for UI compatibility
    async def event_stream():
        response_id = f"mom-oai-{req_data.model}-{str(uuid.uuid4())}"
        # index is handled within _process_mom_chat_request generator now
        # Accumulate the complete content for Langfuse trace update
        complete_content = ""
        trace_obj = None
        # Accumulate usage info if available in chunks (less common for streaming)
        # Or calculate based on collected info after the stream.
        # For now, we'll rely on the non-streaming path for accurate usage/cost in the response object.
        # Langfuse trace update will use the accumulated content.
        # accumulated_usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=None) # Usage accumulation is complex for streaming

        try:
            # Always call _process_mom_chat_request with stream=True to get the generator
            # The internal logic in _process_mom_chat_request handles whether the underlying
            # LLM calls are streaming or not and formats the output as dictionary chunks.
            the_generator = await _process_mom_chat_request(
                req_data.model,
                [m.dict(exclude_none=True) for m in req_data.messages],
                request,
                stream=True, # Always request streaming from _process_mom_chat_request
            )

            # Get the trace object from the request state if it exists
            if hasattr(request.state, "trace_obj"):
                trace_obj = request.state.trace_obj

            async for chunk_dict in the_generator: # Iterate over the dictionary chunks yielded by _process_mom_chat_request
                # Ensure chunk is a dictionary before processing
                if isinstance(chunk_dict, dict):
                    # Check if it's a LiteLLM chunk with choices and at least one choice
                    if "choices" in chunk_dict and isinstance(chunk_dict["choices"], list) and len(chunk_dict["choices"]) > 0:
                        choice = chunk_dict["choices"][0]
                        delta = choice.get("delta")
                        # finish_reason is included in the chunk_dict

                        # Handle content delta
                        if delta and delta.get("content") is not None:
                            # Accumulate content for Langfuse
                            complete_content += delta["content"]

                        # Yield the chunk as an SSE data block
                        yield f"data: {json.dumps(chunk_dict)}\n\n"

                    # Handle potential error chunks from LiteLLM or internal errors
                    elif "error" in chunk_dict:
                         yield f"data: {json.dumps(chunk_dict)}\n\n"
                    else:
                        # If it's a dictionary but not a standard chunk or error, log and skip for now.
                        logger.warning(f"Skipping unexpected chunk format from _process_mom_chat_request: {chunk_dict}")
                else:
                    # Log unexpected chunk type
                    logger.warning(f"Received unexpected chunk type from _process_mom_chat_request: {type(chunk_dict)}")

            # After streaming is done, update Langfuse trace with complete output
            if trace_obj and complete_content:
                try:
                    # Create a response object similar to non-streaming mode for trace output
                    # Note: Usage and cost might not be accurately available here for streaming
                    openai_response_for_trace = OpenAIChatCompletionResponse(
                        id=response_id,
                        created=int(time.time()), # Use current time for the final response object timestamp
                        model=req_data.model,
                        choices=[
                            OpenAIChatCompletionResponseChoice(
                                index=0,
                                message=ChatMessage(
                                    role="assistant",
                                    content=complete_content
                                )
                            )
                        ],
                        # usage=accumulated_usage if accumulated_usage.total_tokens > 0 else None, # Include usage if accumulated
                        # total_cost_usd=... # Cost might need separate calculation or omitted for streaming trace
                    )
                    # Update the trace with the complete output
                    trace_obj.update(output=openai_response_for_trace.dict(exclude_none=True))
                    logger.info("Successfully updated Langfuse trace for streaming response")
                except Exception as e:
                    logger.error(f"Failed to update Langfuse trace for streaming response: {e}")

        except Exception as e:
            logger.error(f"Error in streaming response generator: {str(e)}", exc_info=True)
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
