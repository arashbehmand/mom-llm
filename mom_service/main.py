from __future__ import annotations

import html
import logging
import os
import sys
import uuid
import json # Import json for streaming
import time # Import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import litellm
from dotenv import load_dotenv
from fastapi import (  # Keep APIRouter for now, might be needed by exception handler or other parts
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse # Import StreamingResponse

from .config import (  # Removed ModelConfig as MoMModelConfig alias, not used here
    MoMConfig,
    load_config,
)
from .core_logic import (
    _calculate_and_log_costs,
    _execute_concluding_call,
    _perform_fanout_calls, # This will now be an async generator
    _prepare_concluding_messages,
)

# Import models needed by _process_mom_chat_request and exception_handler
from .endpoints.models import (
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    ThinkingContextItem,
    UsageInfo,
)
from .endpoints.ollama_api import ollama_router

# Routers are now imported from their respective files
from .endpoints.openai_v1 import openai_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("--- mom_service.main.py: Logging configured ---")

LITELLM_VERBOSE_ENV = os.getenv("LITELLM_VERBOSE", "false").lower()
litellm.set_verbose = LITELLM_VERBOSE_ENV in ("true", "1", "yes")
logger.info(
    f"--- LiteLLM verbose logging {'ENABLED' if litellm.set_verbose else 'DISABLED'} ---"
)

# Load configuration
try:
    config: MoMConfig = load_config()
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# Initialize Langfuse if configured
LANGFUSE_CLIENT = None
if config.langfuse:
    try:
        from langfuse import Langfuse

        pub = os.getenv(config.langfuse.public_key_env)
        sec = os.getenv(config.langfuse.secret_key_env)
        host = os.getenv(config.langfuse.host_env)
        if pub and sec and host:
            LANGFUSE_CLIENT = Langfuse(public_key=pub, secret_key=sec, host=host)
            logger.info("--- Langfuse client initialized ---")
        else:
            logger.warning("--- Langfuse configured but missing env vars ---")
    except Exception as e:
        logger.error(f"Langfuse init error: {e}")

# --- Core MoM Chat Processing Logic ---


async def _process_mom_chat_request(
    mom_model_name: str,
    request_messages: List[Dict[str, Any]],
    fastapi_request_obj: Request,
    stream: bool = False,
) -> Union[
    Tuple[
        str,
        Optional[List[ThinkingContextItem]],
        Optional[UsageInfo],
        float,
        str,
        bool,
        Optional[Any],
    ],
    AsyncGenerator[Dict[str, Any], None], # Change return type for streaming to Dict[str, Any] chunks
]:
    """
    Processes a chat request using the MoM logic.
    Returns:
        - final_content_for_assistant: str (potentially with <think> tags)
        - raw_intermediate_thinking_context: Optional[List[ThinkingContextItem]]
        - concluding_llm_usage_info: Optional[UsageInfo]
        - total_request_cost: float
        - actual_mom_model_name_used: str
        - thinking_was_embedded: bool
        - trace_obj: Optional[Any] (Langfuse trace object)
        OR
        - async generator yielding streaming chunks in real-time if stream=True.
          Chunks are dictionaries formatted for OpenAI streaming response.
    """
    timeout = config.service.timeout_seconds
    model_conf = next((m for m in config.models if m.name == mom_model_name), None)
    if not model_conf:
        logger.error(f"MoM Model '{mom_model_name}' not found in configuration.")
        raise ValueError(f"MoM Model '{mom_model_name}' not found.")

    llm_map = {ld.name: ld for ld in config.llm_definitions}
    # thinking_was_embedded_in_content is now only relevant for non-streaming
    thinking_was_embedded_in_content = False

    trace = None
    if LANGFUSE_CLIENT:
        trace = LANGFUSE_CLIENT.trace(
            name=f"MoM-{model_conf.name}-{str(uuid.uuid4())[:8]}",
            user_id=fastapi_request_obj.headers.get("x-user-id", "anon"),
            metadata={
                "model_requested": model_conf.name,
                "num_messages": len(request_messages),
                "streaming": stream,
            },
            input={
                "model": mom_model_name,
                "messages": request_messages,
                "stream": stream,
            },
        )
        # Store trace in request state for access in streaming response
        if hasattr(fastapi_request_obj, "state"):
            fastapi_request_obj.state.trace_obj = trace

    # Step 1: Fan-out (now returns an async generator)
    # We will consume this generator differently based on stream mode
    fanout_results_generator = _perform_fanout_calls(
        model_conf, llm_map, request_messages, timeout, trace
    )

    intermediate_thinking_context: List[ThinkingContextItem] = [] # Collect results here for concluding call

    if stream:
        logger.info(
            "_process_mom_chat_request: Handling streaming request."
        )

        async def streaming_response_generator():
            response_id = f"mom-oai-{mom_model_name}-{str(uuid.uuid4())}"
            index = 0 # For OpenAI streaming chunk index
            thinking_block_open = False # Flag to track if <think> is open

            # Stream thinking context as it becomes available
            logger.info("Streaming fan-out thinking context...")
            async for thinking_item in fanout_results_generator:
                intermediate_thinking_context.append(thinking_item) # Collect for concluding call

                if model_conf.include_thinking_context:
                    if not thinking_block_open:
                        # Yield the opening <think> tag as the first thinking chunk
                        open_tag_data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": mom_model_name,
                            "choices": [
                                {
                                    "index": index,
                                    "delta": {"content": "<think>\n"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield open_tag_data
                        thinking_block_open = True

                    # Format individual thinking item content
                    escaped_item_content = html.escape(thinking_item.content)
                    thinking_chunk_content = (
                        f"Model: {html.escape(thinking_item.model)}\nContent: {escaped_item_content}\n---\n" # Add separator
                    )
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()), # Use current time for chunk
                        "model": mom_model_name,
                        "choices": [
                            {
                                "index": index,
                                "delta": {"content": thinking_chunk_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield data # Yield the dictionary chunk

            logger.info("Finished streaming fan-out thinking context.")

            # Close the <think> block after all fan-out results are streamed
            if thinking_block_open:
                 close_tag_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": mom_model_name,
                    "choices": [
                        {
                            "index": index,
                            "delta": {"content": "</think>\n"},
                            "finish_reason": None,
                        }
                    ],
                }
                 yield close_tag_data


            # Check if any successful fan-out content exists before concluding call
            successful_fanout_content_exists = any(
                item.content
                and not item.content.startswith("Error:")
                and not item.content.startswith("Warning:")
                for item in intermediate_thinking_context
            )
            if not successful_fanout_content_exists:
                logger.error("No successful fan-out responses with usable content.")
                if trace:
                    trace.update(
                        level="ERROR",
                        status_message="All fan-out calls failed or returned no usable content.",
                    )
                if not intermediate_thinking_context and model_conf.llms_to_query:
                     # Yield an error chunk if no fan-out results and LLMs were configured
                    error_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": mom_model_name,
                        "error": {"message": "All fan-out calls failed or returned no usable content.", "type": "internal_server_error"},
                    }
                    yield error_data
                    return # Stop the generator

            # Step 2: Prepare concluding messages (using the collected context)
            concl_msgs_for_llm = _prepare_concluding_messages(
                request_messages, intermediate_thinking_context, model_conf, config
            )

            # Step 3: Concluding LLM (streaming)
            concl_def = llm_map.get(model_conf.concluding_llm)
            if not concl_def:
                logger.error(
                    f"Concluding LLMDefinition '{model_conf.concluding_llm}' not found."
                )
                if trace:
                    trace.update(
                        level="ERROR",
                        status_message=f"Concluding LLMDef '{model_conf.concluding_llm}' not found.",
                    )
                # Yield an error chunk
                error_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": mom_model_name,
                    "error": {"message": f"Concluding LLM definition '{model_conf.concluding_llm}' not found.", "type": "internal_server_error"},
                }
                yield error_data
                return # Stop the generator


            logger.info(
                "_process_mom_chat_request: Calling _execute_concluding_call with stream=True to get generator"
            )
            gen_name_concl = f"concluding-{concl_def.name}" if trace else None
            # Await the async generator function call to get the async generator object
            the_concluding_generator = await _execute_concluding_call(
                concl_def,
                concl_msgs_for_llm,
                timeout,
                options={
                    "trace": trace,
                    "gen_name_concl": gen_name_concl,
                    "stream": True,
                },
            )
            logger.info(
                f"_process_mom_chat_request: Streaming concluding LLM response..."
            )

            # Stream the concluding model responses
            async for chunk in the_concluding_generator:
                 # For OpenAI streaming, yield chunk.choices[0] as dict if present
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    finish_reason = getattr(choice, "finish_reason", None)

                    # Only yield if there's content or a finish reason
                    if (delta and getattr(delta, "content", None) is not None) or finish_reason is not None:
                         data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()), # Use current time for chunk
                            "model": mom_model_name,
                            "choices": [
                                {
                                    "index": index,
                                    "delta": (
                                        {"content": delta.content}
                                        if delta
                                        and getattr(delta, "content", None) is not None
                                        else {}
                                    ),
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                         yield data # Yield the dictionary chunk
                # Handle potential error chunks from LiteLLM or internal errors
                elif hasattr(chunk, "error"):
                    error_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": mom_model_name,
                        "error": chunk["error"],
                    }
                    yield error_data # Yield the dictionary chunk
                else:
                    # Log unexpected chunk format
                    logger.warning(f"Received unexpected chunk format from concluding LLM: {chunk}")

            logger.info("Finished streaming concluding LLM response.")

            # Calculate total cost after all calls are done (for logging/trace)
            # Note: Cost calculation might be less precise in streaming as usage info
            # might not be fully available until the end of the stream for some models.
            # We'll calculate based on collected usage info.
            concluding_llm_usage_info = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=None) # Placeholder
            # In a real streaming scenario, you might need to accumulate usage info from chunks
            # if the LiteLLM response object isn't fully available until the end.
            # For now, we'll rely on the non-streaming path for accurate cost calculation in the return tuple.
            # For streaming, cost calculation might need to happen differently or be omitted from the stream itself.
            # Let's calculate based on the collected intermediate_thinking_context for fanout costs.
            # Concluding LLM cost might be harder to get accurately during streaming.

            # The total cost calculation below is primarily for the non-streaming return value and trace update.
            # For streaming, we don't return the tuple, so this calculation isn't strictly needed *within* the generator,
            # but it's needed for the trace update *after* the generator is consumed in the endpoint.
            # The endpoint will need to handle collecting usage info from the stream if possible.
            # For simplicity now, we'll calculate based on collected fanout usage.

            # The Langfuse trace update with output and usage should ideally happen in the endpoint
            # after the entire stream is consumed and aggregated.

        # Return the async generator object
        return streaming_response_generator()

    # Non-streaming (default) path
    logger.info("_process_mom_chat_request: Handling non-streaming request.")
    try:
        # Consume the fan-out generator into a list for the non-streaming path
        async for thinking_item in fanout_results_generator:
             intermediate_thinking_context.append(thinking_item)

        # Check if any successful fan-out content exists before concluding call
        successful_fanout_content_exists = any(
            item.content
            and not item.content.startswith("Error:")
            and not item.content.startswith("Warning:")
            for item in intermediate_thinking_context
        )
        if not successful_fanout_content_exists:
            logger.error("No successful fan-out responses with usable content.")
            if trace:
                trace.update(
                    level="ERROR",
                    status_message="All fan-out calls failed or returned no usable content.",
                )
            if not intermediate_thinking_context and model_conf.llms_to_query:
                raise ValueError(
                    "All configured fan-out LLM definitions were invalid or missing."
                )


        # Step 2: Prepare concluding messages
        concl_msgs_for_llm = _prepare_concluding_messages(
            request_messages, intermediate_thinking_context, model_conf, config
        )

        # Step 3: Concluding LLM (non-streaming)
        concl_def = llm_map.get(model_conf.concluding_llm)
        if not concl_def:
            logger.error(
                f"Concluding LLMDefinition '{model_conf.concluding_llm}' not found."
            )
            if trace:
                trace.update(
                    level="ERROR",
                    status_message=f"Concluding LLMDef '{model_conf.concluding_llm}' not found.",
                )
            raise ValueError("Concluding LLM definition not found in configuration.")

        gen_name_concl = f"concluding-{concl_def.name}" if trace else None

        # _execute_concluding_call with stream=False returns an async generator that yields one item
        concl_res_obj = None
        async for item in _execute_concluding_call(
            concl_def,
            concl_msgs_for_llm,
            timeout,
            options={
                "trace": trace,
                "gen_name_concl": gen_name_concl,
                "stream": False,
            },
        ):
            concl_res_obj = item
            break # Get the single item

        if (
            not concl_res_obj
            or not concl_res_obj.choices
            or not concl_res_obj.choices[0].message.content
        ):
            logger.error("Concluding LLM call failed or returned empty content.")
            if trace:
                trace.update(level="ERROR", status_message="Concluding LLM call failed.")
            raise ValueError("Concluding LLM failed to generate response.")

        final_content_from_concluding_llm = concl_res_obj.choices[0].message.content

        concluding_usage_data = concl_res_obj.usage or {}
        concluding_llm_usage_info = UsageInfo(
            prompt_tokens=getattr(concluding_usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(concluding_usage_data, "completion_tokens", 0),
            total_tokens=getattr(concluding_usage_data, "total_tokens", 0),
            cost=getattr(concluding_usage_data, "cost", None),
        )

        # Step 4: Embed thinking context if configured (only for non-streaming)
        final_content_to_return = final_content_from_concluding_llm
        if model_conf.include_thinking_context:
            thinking_steps_str_parts = []
            if intermediate_thinking_context:
                for item in intermediate_thinking_context:
                    escaped_item_content = html.escape(item.content)
                    thinking_steps_str_parts.append(
                        f"Model: {html.escape(item.model)}\nContent: {escaped_item_content}"
                    )
            if thinking_steps_str_parts:
                full_thinking_block = (
                    "<think>\n" + "\n---\n".join(thinking_steps_str_parts) + "\n</think>"
                )
                final_content_to_return = (
                    f"{full_thinking_block}\n{final_content_from_concluding_llm}"
                )
                thinking_was_embedded_in_content = True

        total_cost_accumulator = _calculate_and_log_costs(
            intermediate_thinking_context, concluding_llm_usage_info
        )

        if trace:
            pass # Trace update happens in the endpoint for non-streaming

        return (
            final_content_to_return,
            (intermediate_thinking_context if intermediate_thinking_context else None),
            concluding_llm_usage_info,
            total_cost_accumulator,
            model_conf.name,
            thinking_was_embedded_in_content,
            trace,
        )
    except ValueError as e:
         # Re-raise ValueErrors from within the non-streaming path
         raise e
    except Exception as e:
        # Catch any other unexpected errors in the non-streaming path
        logger.error(f"Unexpected error in non-streaming path: {e}", exc_info=True)
        if trace:
             trace.update(level="ERROR", status_message=f"Unexpected Error: {e}")
        raise ValueError(f"An unexpected error occurred: {e}") # Re-raise as ValueError


# --- Main FastAPI App Setup ---
app = FastAPI(title="MoM Service", version="0.2.0")

ALLOWED_CORS_ORIGINS = os.getenv("ALLOWED_CORS_ORIGINS", "")
if ALLOWED_CORS_ORIGINS:
    origins = [o.strip() for o in ALLOWED_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(_: Request, exc: HTTPException):
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_exceeded_error",
    }
    typ = mapping.get(exc.status_code, "api_error")
    error_content_detail = exc.detail
    if isinstance(error_content_detail, str):
        error_content_detail = {"message": error_content_detail, "type": typ}
    elif not isinstance(error_content_detail, dict):
        error_content_detail = {
            "message": "An unexpected error occurred.",
            "type": "api_error",
        }
    if "type" not in error_content_detail:
        error_content_detail["type"] = typ
    if "message" not in error_content_detail:
        error_content_detail["message"] = "Error"
    error_detail_obj = OpenAIErrorDetail(**error_content_detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=OpenAIErrorResponse(error=error_detail_obj).dict(exclude_none=True),
    )


if "openai" in config.service.exposed_apis:
    app.include_router(openai_router)
    logger.info("--- OpenAI compatible API enabled at /v1 ---")

if "ollama" in config.service.exposed_apis:
    app.include_router(
        ollama_router
    )  # Ollama API is included without a special exception handler
    # We'll handle Ollama-specific error formatting in the router itself
    logger.info("--- Ollama compatible API enabled at /ollama/api ---")

if not config.service.exposed_apis:

    @app.get("/")
    async def root():
        return {
            "message": "MoM service is running, but no APIs are exposed. Check configuration."
        }

    logger.warning("--- MoM Service: No APIs exposed in configuration ---")
elif "/" not in [route.path for route in app.routes]:

    @app.get("/")
    async def root_info():
        active_apis = ", ".join(config.service.exposed_apis)
        return {"message": f"MoM service is running. Active APIs: {active_apis}"}


@app.middleware("http")
async def langfuse_trace_middleware(request: Request, call_next):
    # This middleware is primarily for flushing the trace after the request is done.
    # Trace creation and initial setup happens within _process_mom_chat_request.
    # For streaming, the trace update with the final output needs to happen in the endpoint
    # after the stream is fully consumed.
    response = await call_next(request)
    if (
        LANGFUSE_CLIENT
        and hasattr(request.state, "trace_obj")
        and request.state.trace_obj
    ):
        try:
            # For streaming, the trace might still be active here.
            # The endpoint is responsible for calling trace.update() with the final output.
            # Flushing here ensures any completed spans/generations are sent.
            LANGFUSE_CLIENT.flush()
        except Exception as e:
            logger.error(f"Langfuse: Error in middleware flush: {e}")
    return response


logger.info(f"--- MoM Service configured with APIs: {config.service.exposed_apis} ---")
