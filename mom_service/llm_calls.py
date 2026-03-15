"""
LLM call management with caching, retries, and metrics tracking.

This module provides the low-level interface for making calls to various LLM providers
through LiteLLM, with built-in support for:
- Request caching (SQLite-based) to reduce costs and latency
- Automatic retries with exponential backoff for transient failures
- Comprehensive metrics recording for monitoring and cost tracking
- Support for both streaming and non-streaming responses

Key functions:
- _call_lite_llm: Main function for executing LLM calls with caching and retries
- _generate_cache_key: Creates deterministic cache keys from request parameters
- _get_cached_response: Retrieves responses from the cache database
- _cache_response: Stores successful responses in the cache

The cache is implemented using SQLite and stores requests/responses as JSON,
keyed by a SHA256 hash of the model configuration and messages.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import litellm
from litellm.utils import Choices, Message, ModelResponse, Usage

from . import metrics_db
from .config import LLMDefinition, MoMConfig
from .responses_api import call_responses_non_stream

logger = logging.getLogger(__name__)

COMPLETION_API_ROUTE = "completion"
RESPONSES_API_ROUTE = "responses"
REDACTED = "[REDACTED]"

CACHE_KEY_IGNORED_PARAMS = frozenset(
    {
        "api_key",
        "messages",
        "num_retries",
        "timeout",
    }
)

LANGFUSE_IGNORED_PARAMS = frozenset({"messages"})

SENSITIVE_PARAM_EXACT = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "token",
        "secret",
        "secret_key",
        "password",
        "credential",
        "credentials",
        "client_secret",
    }
)
SENSITIVE_PARAM_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_token",
    "_secret",
    "_password",
    "_credential",
    "_credentials",
)
SENSITIVE_PARAM_PREFIXES = (
    "api_key_",
    "apikey_",
    "token_",
    "secret_",
    "password_",
    "auth_",
    "authorization_",
)

# S3 presigned URL query parameters that change on every request.
# Stripping these ensures cache keys are stable across regenerations.
_S3_PRESIGNED_PARAMS = frozenset(
    {
        "X-Amz-Algorithm",
        "X-Amz-Credential",
        "X-Amz-Date",
        "X-Amz-Expires",
        "X-Amz-Security-Token",
        "X-Amz-Signature",
        "X-Amz-SignedHeaders",
    }
)

_PRESIGNED_URL_RE = re.compile(
    r"X-Amz-(?:Date|Signature|Credential|Expires|Algorithm|SignedHeaders|Security-Token)="
)


def _normalize_presigned_urls(text: str) -> str:
    """Strip volatile S3 presigned URL query params so cache keys stay stable."""
    if "X-Amz-" not in text:
        return text

    def _strip_url(match: re.Match) -> str:
        url = match.group(0)
        parsed = urlparse(url)
        if not parsed.query:
            return url
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k not in _S3_PRESIGNED_PARAMS}
        new_query = urlencode(filtered, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    return re.sub(r"https?://[^\s\"'<>]+", _strip_url, text)


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


def _is_sensitive_param_key(key: str) -> bool:
    """Return True when a key likely contains credentials or auth material."""
    key_lower = key.lower()
    if key_lower in SENSITIVE_PARAM_EXACT:
        return True
    if "authorization" in key_lower:
        return True
    if key_lower.endswith(SENSITIVE_PARAM_SUFFIXES):
        return True
    return key_lower.startswith(SENSITIVE_PARAM_PREFIXES)


def _normalize_nested_data(value: Any, redact_sensitive: bool) -> Any:
    """Normalize nested values for deterministic serialization and optional redaction."""
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        return _normalize_presigned_urls(value)

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            if _is_sensitive_param_key(key):
                if redact_sensitive:
                    normalized[key] = REDACTED
                continue
            normalized[key] = _normalize_nested_data(raw_value, redact_sensitive=redact_sensitive)
        return normalized

    if isinstance(value, (list, tuple)):
        return [_normalize_nested_data(item, redact_sensitive=redact_sensitive) for item in value]

    return str(value)


def _normalize_cache_params(params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Drop volatile/runtime params and sensitive keys from cache key generation."""
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in sorted((params or {}).items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if key in CACHE_KEY_IGNORED_PARAMS:
            continue
        if _is_sensitive_param_key(key):
            continue
        normalized[key] = _normalize_nested_data(raw_value, redact_sensitive=False)
    return normalized


def _build_langfuse_model_parameters(params: dict[str, Any], api_route: str) -> dict[str, Any]:
    """Build Langfuse-safe model parameters without sensitive values."""
    langfuse_params: dict[str, Any] = {}
    for raw_key, raw_value in params.items():
        key = str(raw_key)
        if key in LANGFUSE_IGNORED_PARAMS:
            continue
        if _is_sensitive_param_key(key):
            continue
        if isinstance(raw_value, (str, int, float, bool)):
            langfuse_params[key] = raw_value
            continue
        if isinstance(raw_value, (dict, list, tuple)):
            normalized = _normalize_nested_data(raw_value, redact_sensitive=True)
            langfuse_params[key] = json.dumps(normalized, sort_keys=True)
    langfuse_params["api_route"] = api_route
    return langfuse_params


def _generate_cache_key(
    llm_cfg: LLMDefinition,
    messages: list[dict[str, Any]],
    params: Optional[dict[str, Any]],
) -> str:
    """Generates a unique cache key based on LLM config, messages, and parameters."""
    key_data = {
        "llm_name": llm_cfg.name,
        "model": llm_cfg.model,
        "messages": _normalize_nested_data(messages, redact_sensitive=False),
        "params": _normalize_cache_params(params),
    }
    json_string = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_string.encode("utf-8")).hexdigest()


def _generate_legacy_cache_key(
    llm_cfg: LLMDefinition,
    messages: list[dict[str, Any]],
    params: Optional[dict[str, Any]],
) -> str:
    """
    Legacy cache key generator retained for backwards-compatible cache reads.

    This matches the previous behavior that directly hashed sorted raw params.
    """
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


def _select_api_route(llm_def: LLMDefinition, params: dict[str, Any]) -> str:
    api_mode = llm_def.api_mode

    if params.get("stream", False):
        if api_mode == "responses":
            logger.warning(
                "LLM '%s' configured with api_mode='responses' but stream=True; "
                "falling back to completion API.",
                llm_def.name,
            )
        return COMPLETION_API_ROUTE

    if api_mode == "responses":
        return RESPONSES_API_ROUTE

    return COMPLETION_API_ROUTE


def _extract_stream_error_message(chunk_dict: dict[str, Any]) -> str | None:
    """Extract provider error details from a streaming chunk when present."""
    error_payload = chunk_dict.get("error")
    if not error_payload:
        return None

    if isinstance(error_payload, dict):
        error_message = (
            error_payload.get("message")
            or error_payload.get("detail")
            or error_payload.get("error")
        )
        error_type = error_payload.get("type")
        if error_type and error_message:
            return f"{error_type}: {error_message}"
        if error_message:
            return str(error_message)
        return str(error_payload)

    return str(error_payload)


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

    # Resolve API key from environment variable (if configured)
    api_key = os.getenv(llm_def.api_key_env) if llm_def.api_key_env else None

    # Sanitize messages to remove provider-specific fields that may cause errors
    from .multimodal_utils import sanitize_messages_for_provider

    sanitized_messages = sanitize_messages_for_provider(messages, llm_def.model)

    # Build params with retry configuration from service config
    params = {
        "model": llm_def.model,
        "messages": sanitized_messages,
        "stream": stream,
        "timeout": timeout,
        "num_retries": config.service.max_llm_retries,  # LiteLLM retry parameter
        **(llm_def.params or {}),  # LLM-specific params can override defaults (guard against None)
    }

    # Only pass api_key explicitly if resolved; otherwise let LiteLLM resolve it
    if api_key:
        params["api_key"] = api_key

    api_route = _select_api_route(llm_def, params)

    # For streaming, request usage info in the stream
    if stream:
        params["stream_options"] = {"include_usage": True}

    # Generate cache key using sanitized messages for consistency
    cache_key_input = {**params, "_api_route": api_route}
    cache_key = _generate_cache_key(llm_def, sanitized_messages, cache_key_input)
    legacy_cache_key = _generate_legacy_cache_key(llm_def, sanitized_messages, cache_key_input)
    if config.service.cache_enabled:
        cached_response = _get_cached_response(cache_key)
        if not cached_response and legacy_cache_key != cache_key:
            cached_response = _get_cached_response(legacy_cache_key)
            if cached_response:
                logger.info(
                    "Legacy cache hit for %s with key %s... Rewriting using normalized key.",
                    llm_def.name,
                    legacy_cache_key[:8],
                )
                _cache_response(cache_key, sanitized_messages, cached_response)
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

            # Create Langfuse generation for cache hit
            if trace:
                langfuse_params = _build_langfuse_model_parameters(params, api_route)

                generation = trace.generation(
                    name=generation_name,
                    metadata={
                        "call_type": call_type,
                        "llm_name": llm_def.name,
                        "cached": True,
                        "api_route": api_route,
                    },
                    input=sanitized_messages,
                    model=llm_def.model,
                    model_parameters=langfuse_params,
                )

                # End generation immediately for cache hit with usage info
                if cached_response.usage:
                    output_dict = litellm.utils.convert_to_dict(cached_response)

                    # For cache hits, report actual tokens and $0 cost
                    actual_input = getattr(cached_response.usage, "prompt_tokens", 0)
                    actual_output = getattr(cached_response.usage, "completion_tokens", 0)

                    generation.end(
                        output=output_dict,
                        level="DEFAULT",
                        status_message="Cache hit - instant response (cost: $0)",
                        usage={
                            "input": actual_input,
                            "output": actual_output,
                            "total": actual_input + actual_output,
                        },
                        metadata={"cached": True},
                    )

                    # Cache hits have $0 cost
                    generation.update(
                        cost_details={
                            "total": 0.0,
                        }
                    )
                else:
                    generation.end(
                        output=litellm.utils.convert_to_dict(cached_response),
                        level="DEFAULT",
                        status_message="Cache hit - instant response (cost: $0)",
                        metadata={"cached": True},
                    )

            yield cached_response
            return

    logger.info(f"Cache miss for {llm_def.name}. Calling API...")

    generation = None
    if trace:
        langfuse_params = _build_langfuse_model_parameters(params, api_route)

        generation = trace.generation(
            name=generation_name,
            metadata={"call_type": call_type, "llm_name": llm_def.name, "api_route": api_route},
            input=sanitized_messages,
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
            chunk_count = 0

            async for chunk in response_stream:
                chunk_count += 1
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

                stream_error_message = _extract_stream_error_message(chunk_dict)
                if stream_error_message:
                    raise RuntimeError(
                        "Streaming call to "
                        f"{llm_def.name} returned provider error: {stream_error_message}"
                    )

                # Capture usage info if present in chunk (final chunk contains usage)
                # Only use it if it has non-zero values (final chunk)
                if "usage" in chunk_dict and chunk_dict["usage"]:
                    usage_dict = chunk_dict["usage"]
                    # Log the full usage dict to see what Gemini is actually sending
                    logger.info(f"[{llm_def.name}] Chunk {chunk_count} raw usage: {usage_dict}")

                    # Only accept usage if it has actual token counts
                    if usage_dict.get("total_tokens", 0) > 0:
                        # Always overwrite with latest usage (final chunk should have complete usage)
                        accumulated_usage = usage_dict
                        logger.info(
                            f"[{llm_def.name}] Chunk {chunk_count} usage: "
                            f"input={usage_dict.get('prompt_tokens')}, "
                            f"output={usage_dict.get('completion_tokens')}, "
                            f"total={usage_dict.get('total_tokens')}"
                        )

                # Accumulate content for Langfuse
                if "choices" in chunk_dict and chunk_dict["choices"]:
                    for choice in chunk_dict["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                complete_content += content

                yield chunk_dict

            logger.info(
                f"Streaming completed for {llm_def.name}: {chunk_count} chunks, "
                f"content_length={len(complete_content)}, usage_captured={accumulated_usage is not None}"
            )

            if accumulated_usage:
                logger.info(
                    f"Captured usage for {llm_def.name}: "
                    f"input={accumulated_usage.get('prompt_tokens')}, "
                    f"output={accumulated_usage.get('completion_tokens')}, "
                    f"total={accumulated_usage.get('total_tokens')}"
                )

                # Check for detailed token breakdown (reasoning vs text)
                completion_details = accumulated_usage.get("completion_tokens_details", {})
                prompt_details = accumulated_usage.get("prompt_tokens_details", {})

                if completion_details or prompt_details:
                    logger.info(
                        f"Token details for {llm_def.name}: "
                        f"completion_details={completion_details}, "
                        f"prompt_details={prompt_details}"
                    )

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Record metrics for streaming call with accumulated usage
            if accumulated_usage:
                from .cost_calculation import calculate_detailed_cost

                # Extract token counts and details
                prompt_tokens = accumulated_usage.get("prompt_tokens", 0)
                completion_tokens = accumulated_usage.get("completion_tokens", 0)
                total_tokens = accumulated_usage.get("total_tokens", 0)
                completion_details = accumulated_usage.get("completion_tokens_details", {})
                prompt_details = accumulated_usage.get("prompt_tokens_details", {})

                # Calculate detailed cost (with custom pricing if configured)
                total_cost, cost_breakdown = calculate_detailed_cost(
                    model_name=llm_def.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_tokens_details=completion_details,
                    pricing_config=llm_def.pricing,  # Use custom pricing from config
                )

                metrics_db.insert_metric_record(
                    metrics_db.MetricRecord(
                        request_id=request_id,
                        mom_model_name=mom_model_name,
                        llm_name=llm_def.name,
                        call_type=call_type,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost=total_cost,
                        duration_ms=duration_ms,
                        status="SUCCESS",
                        error_message=None,
                        cache_hit=False,
                    )
                )

                # End Langfuse generation with detailed tokens and costs
                if generation:
                    # Build detailed usage dict with flattened token details
                    usage_dict = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    }

                    # Add flattened completion_tokens_details (Langfuse style)
                    if completion_details:
                        for key, value in completion_details.items():
                            usage_dict[f"output_{key}"] = value

                    # Add flattened prompt_tokens_details
                    if prompt_details:
                        for key, value in prompt_details.items():
                            usage_dict[f"input_{key}"] = value

                    # Update with granular cost details BEFORE ending
                    if cost_breakdown:
                        generation.update(cost_details=cost_breakdown)
                        logger.info(
                            f"Updated Langfuse with detailed costs for {llm_def.name}: {cost_breakdown}"
                        )

                    # Report detailed tokens and costs
                    generation.end(
                        output={"content": complete_content, "status": "streaming_completed"},
                        level="DEFAULT",
                        status_message="Streaming response completed successfully",
                        usage=usage_dict,
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
            if api_route == RESPONSES_API_ROUTE:
                logger.info(
                    "Using LiteLLM Responses API for %s (api_mode=%s).",
                    llm_def.name,
                    llm_def.api_mode,
                )
                response = await call_responses_non_stream(llm_def, params, config)
            else:
                response = await litellm.acompletion(**params)
            if config.service.cache_enabled:
                _cache_response(cache_key, sanitized_messages, response)

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Record metrics for successful non-streaming call
            if response.usage:
                from .cost_calculation import calculate_detailed_cost

                # Extract token counts from response.usage
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

                # Extract detailed token breakdown if available
                # LiteLLM returns wrapper objects, convert to dicts
                completion_details_obj = getattr(response.usage, "completion_tokens_details", None)
                prompt_details_obj = getattr(response.usage, "prompt_tokens_details", None)

                # Convert wrapper objects to dicts (filter out None values)
                completion_details = {}
                if completion_details_obj:
                    try:
                        completion_details = {
                            k: v for k, v in vars(completion_details_obj).items() if v is not None
                        }
                    except (TypeError, AttributeError):
                        # If conversion fails, try as dict
                        if isinstance(completion_details_obj, dict):
                            completion_details = completion_details_obj

                prompt_details = {}
                if prompt_details_obj:
                    try:
                        prompt_details = {
                            k: v for k, v in vars(prompt_details_obj).items() if v is not None
                        }
                    except (TypeError, AttributeError):
                        # If conversion fails, try as dict
                        if isinstance(prompt_details_obj, dict):
                            prompt_details = prompt_details_obj

                if completion_details or prompt_details:
                    logger.info(
                        f"Non-streaming token details for {llm_def.name}: "
                        f"completion_details={completion_details}, "
                        f"prompt_details={prompt_details}"
                    )

                # Calculate detailed cost (with custom pricing if configured)
                total_cost, cost_breakdown = calculate_detailed_cost(
                    model_name=llm_def.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_tokens_details=completion_details,
                    pricing_config=llm_def.pricing,  # Use custom pricing from config
                )

                metrics_db.insert_metric_record(
                    metrics_db.MetricRecord(
                        request_id=request_id,
                        mom_model_name=mom_model_name,
                        llm_name=llm_def.name,
                        call_type=call_type,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost=total_cost,
                        duration_ms=duration_ms,
                        status="SUCCESS",
                        error_message=None,
                        cache_hit=False,
                    )
                )

            # End Langfuse generation with actual tokens and detailed costs
            if generation:
                output_dict = litellm.utils.convert_to_dict(response)

                # Report actual tokens with detailed breakdown
                if response.usage:
                    # Build detailed usage dict with flattened token details
                    usage_dict = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    }

                    # Add flattened completion_tokens_details (Langfuse style)
                    if completion_details:
                        for key, value in completion_details.items():
                            usage_dict[f"output_{key}"] = value

                    # Add flattened prompt_tokens_details
                    if prompt_details:
                        for key, value in prompt_details.items():
                            usage_dict[f"input_{key}"] = value

                    # Update with granular cost details BEFORE ending
                    if cost_breakdown:
                        generation.update(cost_details=cost_breakdown)
                        logger.info(
                            f"Updated Langfuse with detailed costs for {llm_def.name}: {cost_breakdown}"
                        )

                    generation.end(
                        output=output_dict,
                        level="DEFAULT",
                        status_message="LLM call completed successfully",
                        usage=usage_dict,
                    )
                else:
                    generation.end(
                        output=output_dict,
                        level="DEFAULT",
                        status_message="LLM call completed successfully",
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
