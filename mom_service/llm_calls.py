import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm.utils import Choices, Message, ModelResponse, Usage

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
    messages: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
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
        # Add a custom attribute to mark this as a cached response
        model_response._is_cached = True
        return model_response
    except Exception as e:
        logger.error(
            f"Error retrieving or reconstructing cached response for key {cache_key}: {e}"
        )
        return None


def _cache_response(
    cache_key: str, messages: List[Dict[str, Any]], response_obj: Any
):
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
    messages: List[Dict[str, Any]],
    timeout: int,
    config: MoMConfig,
    options: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Any, None]:
    """
    Calls the LiteLLM completion endpoint with retry logic and caching.
    Yields the response object.
    """
    options = options or {}
    trace = options.get("trace")
    generation_name = options.get("generation_name", "generation")
    call_type = options.get("call_type", "unknown")
    stream = options.get("stream", False)

    # Build params with retry configuration from service config
    params = {
        "model": llm_def.model,
        "messages": messages,
        "stream": stream,
        "timeout": timeout,
        "num_retries": config.service.max_llm_retries,  # LiteLLM retry parameter
        **llm_def.params,  # LLM-specific params can override defaults
    }

    cache_key = _generate_cache_key(llm_def, messages, params)
    if config.service.cache_enabled:
        cached_response = _get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {llm_def.name} with key {cache_key[:8]}...")
            yield cached_response
            return

    logger.info(f"Cache miss for {llm_def.name}. Calling API...")

    generation = None
    if trace:
        generation = trace.generation(
            name=generation_name,
            metadata={"call_type": call_type, "llm_name": llm_def.name},
            input=messages,
            model=llm_def.model,
            model_parameters={k: v for k, v in params.items() if k != "messages"},
        )

    start_time = time.time()
    try:
        if stream:
            response_stream = await litellm.acompletion(**params)
            async for chunk in response_stream:
                yield chunk
        else:
            response = await litellm.acompletion(**params)
            if config.service.cache_enabled:
                _cache_response(cache_key, messages, response)
            yield response

        end_time = time.time()
        if generation:
            output = (
                litellm.utils.convert_to_dict(response) if not stream else "STREAMED"
            )
            generation.end(output=output)

    except Exception as e:
        end_time = time.time()
        logger.error(f"LLM call to {llm_def.name} failed after {end_time - start_time:.2f}s: {e}")
        if generation:
            generation.end(
                level="ERROR",
                status_message=str(e),
            )
        # Re-raise the exception to be handled by the caller
        raise
