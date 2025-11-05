import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional

import litellm
from litellm.utils import Choices, Message, ModelResponse, Usage

from . import metrics_db
from .config import LLMDefinition, MoMConfig

logger = logging.getLogger(__name__)

# SQLite database file path
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "llm_cache.db")
logger.info(f"Cache DB path: {CACHE_DB_PATH}")


def _init_cache_db():
    """Initializes the SQLite database and creates the cache table if it doesn't exist."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    request_messages TEXT,
                    response_json TEXT,
                    timestamp REAL
                )
            """
            )
            conn.commit()
        logger.info(f"SQLite cache database initialized at {CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing SQLite cache database: {e}")


# Initialize the database on startup
_init_cache_db()


def _generate_cache_key(
    llm_cfg: LLMDefinition,
    messages: list[dict[str, Any]],
    params: Optional[dict[str, Any]],
) -> str:
    """Generates a unique cache key based on LLM config, messages, and parameters."""
    key_data = {
        "llm_name": llm_cfg.name,
        "model": llm_cfg.model,
        "messages": messages,
        "params": dict(sorted((params or {}).items())),
    }
    json_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(json_string.encode("utf-8")).hexdigest()


def _get_cached_response(cache_key: str) -> Optional[ModelResponse]:
    """Retrieves a cached response from the SQLite database and reconstructs LiteLLM ModelResponse."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM cache WHERE key = ?", (cache_key,))
            row = cursor.fetchone()

        if not row:
            return None

        response_data = json.loads(row[0])
        choices = []
        if "choices" in response_data and response_data["choices"]:
            for choice_data in response_data["choices"]:
                message = None
                if "message" in choice_data and choice_data["message"]:
                    msg_data = choice_data["message"]
                    message = Message(
                        content=msg_data.get("content"),
                        role=msg_data.get("role"),
                        function_call=msg_data.get("function_call"),
                        tool_calls=msg_data.get("tool_calls"),
                    )
                choice = Choices(
                    finish_reason=choice_data.get("finish_reason"),
                    index=choice_data.get("index"),
                    message=message,
                    logprobs=choice_data.get("logprobs"),
                )
                choices.append(choice)

        usage = None
        if "usage" in response_data and response_data["usage"]:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
            )

        # Create ModelResponse and mark it as cached
        model_response = ModelResponse(
            id=response_data.get("id"),
            choices=choices,
            created=response_data.get("created"),
            model=response_data.get("model"),
            system_fingerprint=response_data.get("system_fingerprint"),
            usage=usage,
            object=response_data.get("object"),
        )
        model_response._is_cached = True  # pylint: disable=protected-access
        return model_response
    except Exception as e:
        logger.error(f"Error retrieving or reconstructing cached response for key {cache_key}: {e}")
        return None


def _cache_response(cache_key: str, messages: list[dict[str, Any]], response_obj: Any):
    """Caches a response in the SQLite database."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            serializable_response_data = litellm.utils.convert_to_dict(response_obj)
            response_json = json.dumps(serializable_response_data)
            request_messages_json = json.dumps(messages)
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, request_messages, response_json, timestamp) VALUES (?, ?, ?, ?)",
                (cache_key, request_messages_json, response_json, time.time()),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error caching response for key {cache_key}: {e}")


async def _call_lite_llm(
    llm_def: LLMDefinition,
    messages: list[dict[str, Any]],
    timeout: int,
    config: MoMConfig,
    options: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[Any, None]:
    """
    Calls the LiteLLM completion endpoint with retry logic, caching, and metrics recording.
    Yields the response object.
    For streaming, yields chunks and finally a special dict with usage info.
    """
    options = options or {}
    trace = options.get("trace")
    generation_name = options.get("generation_name", "generation")
    call_type = options.get("call_type", "unknown")
    stream = options.get("stream", False)
    request_id = options.get("request_id", "unknown")
    mom_model_name = options.get("mom_model_name", "unknown")

    # Build params with retry configuration from service config
    params = {
        "model": llm_def.model,
        "messages": messages,
        "stream": stream,
        "timeout": timeout,
        "num_retries": config.service.max_llm_retries,  # LiteLLM retry parameter
        **(llm_def.params or {}),  # LLM-specific params can override defaults (guard against None)
    }

    # For streaming, request usage info in the stream
    if stream:
        params["stream_options"] = {"include_usage": True}

    cache_key = _generate_cache_key(llm_def, messages, params)
    if config.service.cache_enabled:
        cached_response = _get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {llm_def.name} with key {cache_key[:8]}...")

            # Record metrics for cache hit
            if cached_response.usage:
                metrics_db.insert_metric_record(
                    metrics_db.MetricRecord(
                        request_id=request_id,
                        mom_model_name=mom_model_name,
                        llm_name=llm_def.name,
                        call_type=call_type,
                        prompt_tokens=getattr(cached_response.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(cached_response.usage, "completion_tokens", 0),
                        total_tokens=getattr(cached_response.usage, "total_tokens", 0),
                        cost=0.0,  # Cached responses have zero cost
                        duration_ms=0.0,  # Cache retrieval is essentially instant
                        status="CACHED",
                        error_message=None,
                        cache_hit=True,
                    )
                )

            yield cached_response
            return

    logger.info(f"Cache miss for {llm_def.name}. Calling API...")

    generation = None
    if trace:
        # Filter out complex types that Langfuse can't handle (dicts, objects)
        # Only include primitive types: str, int, float, bool, list
        langfuse_params = {}
        for k, v in params.items():
            if k == "messages":
                continue  # Skip messages as they're passed separately
            # Only include primitive types that Langfuse accepts
            if isinstance(v, (str, int, float, bool, list)):
                langfuse_params[k] = v
            elif isinstance(v, dict):
                # Convert dicts to JSON string for Langfuse
                import json
                langfuse_params[k] = json.dumps(v)

        generation = trace.generation(
            name=generation_name,
            metadata={"call_type": call_type, "llm_name": llm_def.name},
            input=messages,
            model=llm_def.model,
            model_parameters=langfuse_params,
        )

    start_time = time.time()
    response = None
    try:
        if stream:
            response_stream = await litellm.acompletion(**params)
            accumulated_usage = None
            complete_content = ""

            async for chunk in response_stream:
                # Convert LiteLLM streaming chunk objects to plain dicts so callers
                # (and the OpenAI-compatible endpoint) can consume them uniformly.
                try:
                    chunk_dict = litellm.utils.convert_to_dict(chunk)
                except Exception:
                    # Fallback: attempt to extract a string representation
                    try:
                        chunk_dict = {
                            "choices": [
                                {"delta": {"content": litellm.utils.get_response_string(chunk)}}
                            ]
                        }
                    except Exception:
                        chunk_dict = {"choices": [{"delta": {"content": str(chunk)}}]}

                # Capture usage info if present in chunk (final chunk contains usage)
                if "usage" in chunk_dict and chunk_dict["usage"]:
                    accumulated_usage = chunk_dict["usage"]

                # Accumulate content for Langfuse
                if "choices" in chunk_dict and chunk_dict["choices"]:
                    for choice in chunk_dict["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                complete_content += content

                yield chunk_dict

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Record metrics for streaming call with accumulated usage
            if accumulated_usage:
                from .endpoints.models import UsageInfo

                usage_info = UsageInfo.from_litellm_usage(
                    accumulated_usage,
                    response_obj=None,  # No full response object for streaming
                    is_cached=False,
                    pricing_config=llm_def.pricing,
                    model_name=llm_def.model,  # Pass model name for cost calculation
                )

                metrics_db.insert_metric_record(
                    metrics_db.MetricRecord(
                        request_id=request_id,
                        mom_model_name=mom_model_name,
                        llm_name=llm_def.name,
                        call_type=call_type,
                        prompt_tokens=usage_info.prompt_tokens or 0,
                        completion_tokens=usage_info.completion_tokens or 0,
                        total_tokens=usage_info.total_tokens or 0,
                        cost=usage_info.cost,
                        duration_ms=duration_ms,
                        status="SUCCESS",
                        error_message=None,
                        cache_hit=False,
                    )
                )

                # End Langfuse generation with usage info
                if generation:
                    generation.end(
                        output={"content": complete_content, "status": "streaming_completed"},
                        level="DEFAULT",
                        status_message="Streaming response completed successfully",
                        usage={
                            "input": accumulated_usage.get("prompt_tokens", 0),
                            "output": accumulated_usage.get("completion_tokens", 0),
                            "total": accumulated_usage.get("total_tokens", 0),
                        },
                    )
            else:
                logger.warning(f"No usage info received in streaming response for {llm_def.name}")
                # End Langfuse generation without usage
                if generation:
                    generation.end(
                        output={"content": complete_content, "status": "streaming_completed"},
                        level="DEFAULT",
                        status_message="Streaming response completed (no usage info)",
                    )
        else:
            response = await litellm.acompletion(**params)
            if config.service.cache_enabled:
                _cache_response(cache_key, messages, response)

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Record metrics for successful non-streaming call
            if response.usage:
                from .endpoints.models import UsageInfo

                usage_info = UsageInfo.from_litellm_usage(
                    response.usage,
                    response_obj=response,
                    is_cached=False,
                    pricing_config=llm_def.pricing,
                )

                metrics_db.insert_metric_record(
                    metrics_db.MetricRecord(
                        request_id=request_id,
                        mom_model_name=mom_model_name,
                        llm_name=llm_def.name,
                        call_type=call_type,
                        prompt_tokens=usage_info.prompt_tokens or 0,
                        completion_tokens=usage_info.completion_tokens or 0,
                        total_tokens=usage_info.total_tokens or 0,
                        cost=usage_info.cost,
                        duration_ms=duration_ms,
                        status="SUCCESS",
                        error_message=None,
                        cache_hit=False,
                    )
                )

            # End Langfuse generation with comprehensive output and metadata
            if generation:
                output_dict = litellm.utils.convert_to_dict(response)
                generation.end(
                    output=output_dict,
                    level="DEFAULT",
                    status_message="LLM call completed successfully",
                    usage=(
                        {
                            "input": response.usage.prompt_tokens if response.usage else 0,
                            "output": response.usage.completion_tokens if response.usage else 0,
                            "total": response.usage.total_tokens if response.usage else 0,
                        }
                        if response.usage
                        else None
                    ),
                )

            yield response

    except Exception as e:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        logger.error(f"LLM call to {llm_def.name} failed after {duration_ms:.2f}ms: {e}")

        # Record metrics for failed call
        metrics_db.insert_metric_record(
            metrics_db.MetricRecord(
                request_id=request_id,
                mom_model_name=mom_model_name,
                llm_name=llm_def.name,
                call_type=call_type,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0,
                duration_ms=duration_ms,
                status="FAILED",
                error_message=str(e)[:500],  # Truncate error message to avoid DB bloat
                cache_hit=False,
            )
        )

        if generation:
            generation.end(
                level="ERROR",
                status_message=str(e),
            )
        # Re-raise the exception to be handled by the caller
        raise
