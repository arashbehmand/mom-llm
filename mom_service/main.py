from __future__ import annotations

import html
import logging
import os
import sys
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import litellm
from dotenv import load_dotenv
from fastapi import (  # Keep APIRouter for now, might be needed by exception handler or other parts
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (  # Removed ModelConfig as MoMModelConfig alias, not used here
    MoMConfig,
    load_config,
)
from .core_logic import (
    _calculate_and_log_costs,
    _execute_concluding_call,
    _perform_fanout_calls,
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
    AsyncGenerator[str, None],
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
    """
    timeout = config.service.timeout_seconds
    model_conf = next((m for m in config.models if m.name == mom_model_name), None)
    if not model_conf:
        logger.error(f"MoM Model '{mom_model_name}' not found in configuration.")
        raise ValueError(f"MoM Model '{mom_model_name}' not found.")

    llm_map = {ld.name: ld for ld in config.llm_definitions}
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

    # Step 1: Fan-out
    intermediate_thinking_context = await _perform_fanout_calls(
        model_conf, llm_map, request_messages, timeout, trace
    )

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

    # Step 3: Concluding LLM
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

    if stream:
        logger.info(
            "_process_mom_chat_request: Calling _execute_concluding_call with stream=True to get generator"
        )
        # Await the async generator function call to get the async generator object
        the_generator = await _execute_concluding_call(
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
            f"_process_mom_chat_request: Returning generator of type {type(the_generator)}"
        )

        # Prepare thinking context if configured
        thinking_block = None
        if model_conf.include_thinking_context and intermediate_thinking_context:
            thinking_steps_str_parts = []
            for item in intermediate_thinking_context:
                escaped_item_content = html.escape(item.content)
                thinking_steps_str_parts.append(
                    f"Model: {html.escape(item.model)}\nContent: {escaped_item_content}"
                )
            if thinking_steps_str_parts:
                thinking_block = "<think>\n" + "\n---\n".join(thinking_steps_str_parts) + "\n</think>\n"
        
        # Wrap the generator to filter/transform only ModelResponse chunks for streaming
        async def filtered_generator():
            # First stream the thinking context if available
            if thinking_block:
                # Stream thinking context as first chunk
                data = {
                    "choices": [
                        {
                            "delta": {"content": thinking_block},
                            "finish_reason": None,
                        }
                    ]
                }
                yield data
            
            # Then stream the concluding model responses
            async for chunk in the_generator:
                # For OpenAI streaming, yield chunk.choices[0] as dict if present
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                    data = {
                        "choices": [
                            {
                                "delta": (
                                    {"content": delta.content}
                                    if delta
                                    and getattr(delta, "content", None) is not None
                                    else {}
                                ),
                                "finish_reason": finish_reason,
                            }
                        ]
                    }
                    yield data

        return filtered_generator()
    concl_res_obj = await _execute_concluding_call(
        concl_def,
        concl_msgs_for_llm,
        timeout,
        options={
            "trace": trace,
            "gen_name_concl": gen_name_concl,
            "stream": False,
        },
    )

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

    # Step 4: Embed thinking context if configured
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
        pass

    return (
        final_content_to_return,
        (intermediate_thinking_context if intermediate_thinking_context else None),
        concluding_llm_usage_info,
        total_cost_accumulator,
        model_conf.name,
        thinking_was_embedded_in_content,
        trace,
    )


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
    if LANGFUSE_CLIENT and hasattr(request.state, "trace_obj"):
        pass
    elif LANGFUSE_CLIENT:
        pass
    response = await call_next(request)
    if (
        LANGFUSE_CLIENT
        and hasattr(request.state, "trace_obj")
        and request.state.trace_obj
    ):
        try:
            LANGFUSE_CLIENT.flush()
        except Exception as e:
            logger.error(f"Langfuse: Error in middleware flush: {e}")
    return response


logger.info(f"--- MoM Service configured with APIs: {config.service.exposed_apis} ---")
