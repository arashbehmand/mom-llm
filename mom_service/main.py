from __future__ import annotations

import html
import logging
import os
import sys
import uuid
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import litellm
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import Langfuse

from .config import (
    MoMConfig,
    load_config,
)
from .core_logic import (
    _calculate_and_log_costs,
    _execute_concluding_call,
    _perform_fanout_calls,
    _prepare_concluding_messages,
)

from .endpoints.models import (
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    ThinkingContextItem,
    UsageInfo,
)
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

try:
    config: MoMConfig = load_config()
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

LANGFUSE_CLIENT = None
if config.langfuse:
    try:
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

app = FastAPI()

# Configure CORS from environment variable (comma-separated origins)
cors_env = os.getenv("ALLOWED_CORS_ORIGINS", "").strip()
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    # If not configured, default to allowing all (safest for local dev; restrict in prod)
    allowed_origins = ["*"]

logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    error_detail = OpenAIErrorDetail(
        message=exc.detail,
        type="invalid_request_error",
        param=None,
        code=None,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=OpenAIErrorResponse(error=error_detail).model_dump(),
    )

@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception):
    error_detail = OpenAIErrorDetail(
        message=str(exc),
        type="internal_server_error",
        param=None,
        code=None,
    )
    return JSONResponse(
        status_code=500,
        content=OpenAIErrorResponse(error=error_detail).model_dump(),
    )

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
    AsyncGenerator[Dict[str, Any], None],
]:
    logger.info(f"--- _process_mom_chat_request received model name: {mom_model_name} ---")
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
        if hasattr(fastapi_request_obj, "state"):
            fastapi_request_obj.state.trace_obj = trace

    fanout_results_generator = _perform_fanout_calls(
        model_conf, llm_map, request_messages, timeout, config, trace
    )

    intermediate_thinking_context: List[ThinkingContextItem] = []

    if stream:
        logger.info("_process_mom_chat_request: Handling streaming request.")

        async def streaming_response_generator():
            response_id = f"mom-oai-{mom_model_name}-{str(uuid.uuid4())}"
            index = 0
            thinking_block_open = False

            logger.info("Streaming fan-out thinking context...")
            async for thinking_item in fanout_results_generator:
                intermediate_thinking_context.append(thinking_item)

                if model_conf.include_thinking_context:
                    if not thinking_block_open:
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

                    item_content_cleaned = thinking_item.content.replace("<think>", "[inner_think]").replace("</think>", "[/inner_think]")
                    escaped_item_content = html.escape(item_content_cleaned)
                    thinking_chunk_content = (
                        f"Model: {html.escape(thinking_item.model)}\nContent: {escaped_item_content}\n---\n"
                    )
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": mom_model_name,
                        "choices": [
                            {
                                "index": index,
                                "delta": {"content": thinking_chunk_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield data

            logger.info("Finished streaming fan-out thinking context.")

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
                    error_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": mom_model_name,
                        "error": {"message": "All fan-out calls failed or returned no usable content.", "type": "internal_server_error"},
                    }
                    yield error_data
                    return

            concl_msgs_for_llm = _prepare_concluding_messages(
                request_messages, intermediate_thinking_context, model_conf, config
            )

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
                error_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": mom_model_name,
                    "error": {"message": f"Concluding LLM definition '{model_conf.concluding_llm}' not found.", "type": "internal_server_error"},
                }
                yield error_data
                return

            logger.info(
                "_process_mom_chat_request: Calling _execute_concluding_call with stream=True to get generator"
            )
            gen_name_concl = f"concluding-{concl_def.name}" if trace else None
            the_concluding_generator = await _execute_concluding_call(
                concl_def,
                concl_msgs_for_llm,
                timeout,
                config,
                options={
                    "trace": trace,
                    "gen_name_concl": gen_name_concl,
                    "stream": True,
                },
            )

            logger.info(
                "Streaming concluding LLM response..."
            )
            final_content_streamed = ""
            async for chunk in the_concluding_generator:
                if chunk.choices[0].delta.content:
                    final_content_streamed += chunk.choices[0].delta.content
                logger.debug("_process_mom_chat_request: Yielding concluding chunk: %s", chunk)
                yield chunk

            if trace:
                trace.update(output={"final_content_streamed": final_content_streamed})

        return streaming_response_generator()

    else: # Non-streaming
        async for item in fanout_results_generator:
            intermediate_thinking_context.append(item)

        concl_msgs_for_llm = _prepare_concluding_messages(
            request_messages, intermediate_thinking_context, model_conf, config
        )

        concl_def = llm_map.get(model_conf.concluding_llm)
        if not concl_def:
            raise ValueError(f"Concluding LLM '{model_conf.concluding_llm}' not found.")

        gen_name_concl = f"concluding-{concl_def.name}" if trace else None
        concluding_llm_response = await _execute_concluding_call(
            concl_def,
            concl_msgs_for_llm,
            timeout,
            config,
            options={
                "trace": trace,
                "gen_name_concl": gen_name_concl,
                "stream": False,
            },
        )

        final_content = ""
        if (
            concluding_llm_response
            and concluding_llm_response.choices
            and concluding_llm_response.choices[0].message
        ):
            final_content = concluding_llm_response.choices[0].message.content

        concluding_llm_usage_info = None
        if concluding_llm_response and concluding_llm_response.usage:
            concluding_llm_usage_info = UsageInfo.from_litellm_usage(
                concluding_llm_response.usage,
                response_obj=concluding_llm_response
            )

        total_request_cost = _calculate_and_log_costs(
            intermediate_thinking_context, concluding_llm_usage_info
        )
        if concluding_llm_usage_info:
            logger.info(f"Concluding LLM usage: {concluding_llm_usage_info.total_tokens} tokens, cost ${total_request_cost:.6f}")

        if trace:
            trace.update(
                output={"final_content": final_content},
                metadata={
                    "total_request_cost": total_request_cost,
                },
            )

        return (
            final_content,
            intermediate_thinking_context if model_conf.include_thinking_context else None,
            concluding_llm_usage_info,
            total_request_cost,
            mom_model_name,
            thinking_was_embedded_in_content,
            None,
        )

app.include_router(openai_router)

logger.info("--- MoM Service configured with OpenAI-compatible API at /v1 ---")
