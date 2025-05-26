import asyncio
import logging
import os
import json
import hashlib
import sqlite3
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm.utils import ModelResponse, Usage, Choices, Message # Import LiteLLM specific types

# Import Langfuse types for tracing
from langfuse.api.resources.commons.types.observation_level import ObservationLevel

# No type imports from langfuse; use Any for trace/generation
from .config import LLMDefinition, MoMConfig # Import MoMConfig to access cache_enabled

logger = logging.getLogger(__name__)

# SQLite database file path
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "llm_cache.db")
logger.info(f"Cache DB path: {CACHE_DB_PATH}")

def _init_cache_db():
    """Initializes the SQLite database and creates the cache table if it doesn't exist."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                request_messages TEXT,
                response_json TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"SQLite cache database initialized at {CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing SQLite cache database: {e}")

# Initialize the database on startup
_init_cache_db()


def _generate_cache_key(llm_cfg: LLMDefinition, messages: List[Dict[str, Any]], params: Optional[Dict[str, Any]]) -> str:
    """Generates a unique cache key based on LLM config, messages, and parameters."""
    # Ensure consistent ordering of messages and params for stable key generation
    key_data = {
        "llm_name": llm_cfg.name,
        "model": llm_cfg.model,
        "messages": messages, # Assuming messages order is stable
        "params": dict(sorted((params or {}).items())) # Sort params
    }
    # Use json.dumps with sorted keys for consistent string representation
    json_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(json_string.encode('utf-8')).hexdigest()

def _get_cached_response(cache_key: str) -> Optional[ModelResponse]:
    """Retrieves a cached response from the SQLite database and reconstructs LiteLLM ModelResponse."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT response_json FROM cache WHERE key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            response_json = row[0]
            response_data = json.loads(response_json)

            # Reconstruct LiteLLM ModelResponse object from deserialized data
            # This requires mapping the dictionary structure back to LiteLLM object structure
            # We need to handle nested objects like choices, message, and usage

            choices = []
            if 'choices' in response_data and response_data['choices']:
                for choice_data in response_data['choices']:
                    message = None
                    if 'message' in choice_data and choice_data['message'] is not None:
                        message_data = choice_data['message']
                        # Reconstruct Message object
                        message = Message(
                            content=message_data.get('content'),
                            role=message_data.get('role'),
                            function_call=message_data.get('function_call'),
                            tool_calls=message_data.get('tool_calls'), # tool_calls should be list of dicts or None
                            # Add other message attributes if needed
                        )
                    # Reconstruct Choice object
                    choice = Choices(
                        finish_reason=choice_data.get('finish_reason'),
                        index=choice_data.get('index'),
                        message=message,
                        logprobs=choice_data.get('logprobs'),
                        # Add other choice attributes if needed
                    )
                    choices.append(choice)

            usage = None
            if 'usage' in response_data and response_data['usage'] is not None:
                usage_data = response_data['usage']
                # Reconstruct Usage object
                usage = Usage(
                    prompt_tokens=usage_data.get('prompt_tokens'),
                    completion_tokens=usage_data.get('completion_tokens'),
                    total_tokens=usage_data.get('total_tokens'),
                    # Add other usage attributes if needed
                )

            # Reconstruct ModelResponse object
            cached_response_obj = ModelResponse(
                id=response_data.get('id'),
                choices=choices,
                created=response_data.get('created'),
                model=response_data.get('model'),
                system_fingerprint=response_data.get('system_fingerprint'),
                usage=usage,
                object=response_data.get('object'),
                # Add other ModelResponse attributes if needed
            )

            return cached_response_obj
        return None
    except Exception as e:
        logger.error(f"Error retrieving or reconstructing cached response for key {cache_key}: {e}")
        return None

def _cache_response(cache_key: str, messages: List[Dict[str, Any]], response_obj: Any):
    """Caches a response in the SQLite database."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        # Use litellm.utils.convert_to_dict for robust serialization
        try:
            serializable_response_data = litellm.utils.convert_to_dict(response_obj)

            # Explicitly handle tool_calls serialization if convert_to_dict didn't fully convert them
            if 'choices' in serializable_response_data and serializable_response_data['choices']:
                for choice_data in serializable_response_data['choices']:
                    if 'message' in choice_data and choice_data['message'] is not None:
                        message_data = choice_data['message']
                        if 'tool_calls' in message_data and isinstance(message_data['tool_calls'], list):
                            processed_tool_calls = []
                            for tc in message_data['tool_calls']:
                                # Check if the item is still a LiteLLM object (like ToolCall)
                                if hasattr(tc, '__dict__'): # Simple check for object-like structure
                                    tool_call_dict = {
                                        'id': getattr(tc, 'id', None),
                                        'type': getattr(tc, 'type', None),
                                        'function': {}
                                    }
                                    if hasattr(tc, 'function') and hasattr(tc.function, '__dict__'):
                                         tool_call_dict['function'] = {
                                             'name': getattr(tc.function, 'name', None),
                                             'arguments': getattr(tc.function, 'arguments', None),
                                         }
                                    processed_tool_calls.append(tool_call_dict)
                                else:
                                    # If it's already a dictionary or other serializable type, keep it
                                    processed_tool_calls.append(tc)
                            message_data['tool_calls'] = processed_tool_calls


            response_json = json.dumps(serializable_response_data)
        except Exception as e:
             logger.error(f"Error serializing response data for key {cache_key} using litellm.utils.convert_to_dict: {e}")
             conn.close()
             return


        cursor.execute(
            "INSERT OR REPLACE INTO cache (key, request_messages, response_json, timestamp) VALUES (?, ?, ?, ?)",
            (cache_key, json.dumps(messages), response_json, time.time())
        )
        conn.commit()
        conn.close()
        logger.info(f"Cached response for key {cache_key}")
    except Exception as e:
        logger.error(f"Error caching response for key {cache_key}: {e}")


async def _call_lite_llm(
    llm_cfg: LLMDefinition,
    messages: List[Dict[str, Any]],
    timeout_val: int,
    config: MoMConfig, # Pass the config object
    options: Optional[dict] = None,
) -> AsyncGenerator[litellm.ModelResponse, None]:
    """
    Helper function to make an asynchronous call to an LLM using LiteLLM.
    Includes timeout, basic error handling, caching, and optional Langfuse tracing.
    Includes retry logic for specific exceptions.
    Always returns an async generator. For non-streaming, it yields one item then stops.
    For streaming, it yields chunks. On error, it yields nothing and stops.
    Caching is currently only implemented for non-streaming calls.
    """
    options = options or {}
    call_type = options.get("call_type", "Fan-out")
    trace = options.get("trace")
    generation_name = options.get("generation_name")
    stream = options.get("stream", False)
    logger.info(
        f"--- _call_lite_llm attempting to call: {llm_cfg.name} as {call_type} ---"
    )
    api_key = os.getenv(llm_cfg.api_key_env)
    if not api_key:
        logger.error(
            f"API key env variable {llm_cfg.api_key_env} not set for {call_type} LLM {llm_cfg.name}"
        )
        return

    model_name = llm_cfg.model
    logger.info(
        f"Calling {call_type} LLM: {llm_cfg.name} (Model: {model_name}) with timeout {timeout_val}s"
    )

    cache_key = None
    cached_response = None
    logger.info(f"Cache enabled: {config.service.cache_enabled}")
    if config.service.cache_enabled: # Check cache regardless of stream
        cache_key = _generate_cache_key(llm_cfg, messages, llm_cfg.params)
        logger.info(f"Generated cache key: {cache_key}")
        cached_response = _get_cached_response(cache_key)
        if not cached_response:
            logger.info(f"Cache miss for key: {cache_key}")

    if cached_response:
        logger.info(f"Cache hit for {llm_cfg.name} (Model: {model_name}). Returning cached response.")
        # Log cache hit in Langfuse if tracing is enabled
        if trace and generation_name:
             try:
                 trace.generation(
                     name=f"{generation_name}-cached",
                     input=messages,
                     output=cached_response.choices[0].message.content if hasattr(cached_response, "choices") and cached_response.choices else "<cached>",
                     model=model_name,
                     metadata={
                         "call_type": call_type,
                         "llm_name": llm_cfg.name,
                         "cached": True,
                         **(llm_cfg.params or {}),
                     },
                     level="DEFAULT", # Changed from ObservationLevel.DEFAULT to "DEFAULT"
                     status_message="Cache Hit"
                 )
             except Exception as e:
                 logger.error(f"Langfuse: Error logging cache hit for '{generation_name}': {e}")

        if stream: # Handle streaming cache hit
            logger.info(f"Handling streaming cache hit for {llm_cfg.name} (Model: {model_name}).")
            # For streaming cache hits, the cached response is a reconstructed ModelResponse object.
            # We need to convert this non-streaming object into a stream of chunks
            # that mimic the LiteLLM streaming output format.
            async def cached_streaming_generator():
                # Assuming cached_response is a ModelResponse object
                if hasattr(cached_response, "choices") and cached_response.choices:
                    choice = cached_response.choices[0] # Assuming one choice for simplicity
                    content = getattr(choice.message, "content", "") if hasattr(choice, "message") else ""
                    finish_reason = getattr(choice, "finish_reason", "stop") # Default to stop

                    # Yield content chunk(s)
                    if content:
                        # Yield content in chunks, respecting potential line breaks or just yielding the whole content
                        # For simplicity, yield the whole content as one chunk for now
                        yield {
                            "id": cached_response.id or "cached-response", # Use cached ID or a placeholder
                            "object": "chat.completion.chunk",
                            "created": cached_response.created or int(time.time()), # Use cached created or current time
                            "model": cached_response.model or model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None, # No finish reason in content chunk
                                }
                            ],
                        }
                    # Yield tool_calls chunk if present
                    if hasattr(cached_response.choices[0].message, "tool_calls") and cached_response.choices[0].message.tool_calls:
                         yield {
                            "id": cached_response.id or "cached-response",
                            "object": "chat.completion.chunk",
                            "created": cached_response.created or int(time.time()),
                            "model": cached_response.model or model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": cached_response.choices[0].message.tool_calls},
                                    "finish_reason": None,
                                }
                            ],
                        }

                    # Yield finish reason chunk
                    yield {
                        "id": cached_response.id or "cached-response",
                        "object": "chat.completion.chunk",
                        "created": cached_response.created or int(time.time()),
                        "model": cached_response.model or model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {}, # Empty delta for finish reason chunk
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                # Optional: Yield an empty chunk to signal end of stream (some clients expect this)
                # yield {} # Not strictly necessary for OpenAI format, but can help

            # Yield from the generator to return its contents
            async for chunk in cached_streaming_generator():
                yield chunk
            return # Return after yielding all chunks from the cached generator

        else: # Handle non-streaming cache hit (existing logic)
            logger.info(f"Handling non-streaming cache hit for {llm_cfg.name} (Model: {model_name}).")
            yield cached_response
            return # Return after yielding cached response

    # If no cache hit, proceed with API call logic
    current_generation = None
    if trace and generation_name:
        try:
            current_generation = trace.generation(
                name=generation_name,
                input=messages,
                model=model_name,
                metadata={
                    "call_type": call_type,
                    "llm_name": llm_cfg.name,
                    "cached": False, # Mark as not cached
                    **(llm_cfg.params or {}),
                },
            )
        except Exception as e:
            logger.error(
                f"Langfuse: Error creating generation '{generation_name}': {e}"
            )
            current_generation = None

    if stream: # Streaming API call logic (only if no cache hit)
        logger.info(
            f"Calling {call_type} LLM in streaming mode: {llm_cfg.name} (Model: {model_name})"
        )
        try:
            async_generator = await litellm.acompletion(
                model=model_name,
                messages=messages,
                api_key=api_key,
                stream=True,
                **(llm_cfg.params or {}),
            )

            streamed_content = "" # To capture content for Langfuse

            accumulated_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            model_id = None
            created_time = None
            response_id = None
            usage_data = None # Usage for streaming is often not available until the end

            async for chunk in async_generator:
                # Convert LiteLLM chunk object to dictionary before yielding
                try:
                    chunk_dict = litellm.utils.convert_to_dict(chunk)
                except Exception as e:
                    logger.error(f"Error converting LiteLLM chunk to dict: {e}")
                    # Optionally yield an error chunk here if conversion fails
                    continue # Skip this chunk if conversion fails

                yield chunk_dict # Yield the dictionary chunk

                # Accumulate content and tool calls for caching and Langfuse from the dictionary
                if "choices" in chunk_dict and isinstance(chunk_dict["choices"], list) and len(chunk_dict["choices"]) > 0:
                    choice = chunk_dict["choices"][0]
                    delta = choice.get("delta")

                    if delta:
                        if "content" in delta and delta["content"] is not None:
                            accumulated_content += delta["content"]
                        if "tool_calls" in delta and delta["tool_calls"] is not None:
                             # LiteLLM streams tool_calls as a list of deltas
                             # We need to reconstruct the full tool_calls list
                             # This is a simplified accumulation; a robust implementation
                             # might need to handle index and function arguments merging.
                             # For now, append the delta tool_calls.
                             # Ensure tool_calls are in a consistent format (list of dicts)
                             if isinstance(delta["tool_calls"], list):
                                 accumulated_tool_calls.extend(delta["tool_calls"])
                             else:
                                 logger.warning(f"Unexpected tool_calls format in delta: {delta['tool_calls']}")


                    if "finish_reason" in choice and choice["finish_reason"] is not None:
                        finish_reason = choice["finish_reason"]

                # Capture metadata from the first chunk (using dictionary access)
                if response_id is None and "id" in chunk_dict:
                    response_id = chunk_dict["id"]
                if model_id is None and "model" in chunk_dict:
                    model_id = chunk_dict["model"]
                if created_time is None and "created" in chunk_dict:
                    created_time = chunk_dict["created"]
                # Usage is typically in the last chunk, but not always guaranteed for streaming
                if "usage" in chunk_dict and chunk_dict["usage"] is not None:
                     # Convert usage dict back to LiteLLM Usage object for consistency
                     try:
                         usage_data = litellm.utils.Usage(**chunk_dict["usage"])
                     except Exception as e:
                         logger.warning(f"Error converting usage dict to LiteLLM Usage object: {e}")
                         usage_data = chunk_dict["usage"] # Keep as dict if conversion fails


            # After the stream is consumed, reconstruct the full response object for caching
            # This reconstruction is simplified and might need refinement for complex cases
            # like multiple choices or detailed logprobs.
            # Ensure accumulated_tool_calls is a list of dicts for the Message object
            final_tool_calls_for_message = [litellm.types.utils.ToolCall(**tc) if isinstance(tc, dict) else tc for tc in accumulated_tool_calls]

            reconstructed_response = ModelResponse(
                id=response_id or "streamed-response",
                choices=[
                    Choices(
                        finish_reason=finish_reason,
                        index=0, # Assuming single choice for simplicity
                        message=Message(
                            content=accumulated_content if accumulated_content else None,
                            role="assistant", # Assuming assistant role for model response
                            tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                        ),
                        # logprobs=... # Not accumulating logprobs in this example
                    )
                ],
                created=created_time or int(time.time()),
                model=model_id or model_name,
                usage=usage_data,
                object="chat.completion", # Assuming chat completion
                # system_fingerprint=... # Not accumulating system_fingerprint
            )

            # Cache the reconstructed response if caching is enabled
            if config.service.cache_enabled and cache_key:
                 _cache_response(cache_key, messages, reconstructed_response)
                 logger.info(f"Cached reconstructed streaming response for {llm_cfg.name} (Model: {model_name}).")


            if current_generation:
                try:
                    # Log the full accumulated output in Langfuse
                    output_content_for_langfuse = accumulated_content if accumulated_content else "<streamed>"
                    if accumulated_tool_calls:
                         # Include tool calls in Langfuse output representation if present
                         output_content_for_langfuse += f"\nTool Calls: {json.dumps([litellm.utils.convert_to_dict(tc) for tc in accumulated_tool_calls])}"

                    current_generation.end(
                        output=output_content_for_langfuse,
                        usage=usage_data # Use accumulated usage if available
                    )
                except Exception as e:
                    logger.error(
                        f"Langfuse: Error ending streaming generation '{generation_name}': {e}"
                    )

        except Exception as e:
            logger.error(
                f"Streaming call failed for {llm_cfg.name} (Model: {model_name}): {e}"
            )
            if current_generation:
                current_generation.end(
                    level="ERROR", status_message=f"Streaming Error: {e}"
                )
        return # Generator stops

    # Non-streaming API call logic (with retry, only if no cache hit)
    retries = 0
    last_exception = None
    while retries <= config.service.max_retries:
        try:
            logger.info(
                f"Attempt {retries + 1}/{config.service.max_retries + 1} for {call_type} LLM: {llm_cfg.name} (Model: {model_name})"
            )
            response_obj = await asyncio.wait_for(
                litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    api_key=api_key,
                    stream=False,
                    **(llm_cfg.params or {}),
                ),
                timeout=timeout_val,
            )
            logger.info(f"{call_type} LLM {llm_cfg.name} call successful on attempt {retries + 1}.")

            # Cache the response if caching is enabled (only non-streaming calls reach here)
            # Ensure response_obj is not a stream wrapper before caching
            if config.service.cache_enabled and cache_key and not hasattr(response_obj, '__aiter__'):
                 _cache_response(cache_key, messages, response_obj)
                 logger.info(f"Cached response for {llm_cfg.name} (Model: {model_name}).")
            elif config.service.cache_enabled and cache_key and hasattr(response_obj, '__aiter__'):
                 logger.warning(f"Attempted to cache a streaming response for {llm_cfg.name} (Model: {model_name}) in non-streaming path. Caching skipped.")


            if current_generation:
                try:
                    output_content = None
                    has_choices = hasattr(response_obj, "choices") and response_obj.choices
                    has_message = (
                        has_choices
                        and hasattr(response_obj.choices[0], "message")
                        and response_obj.choices[0].message
                    )
                    has_content = has_message and hasattr(
                        response_obj.choices[0].message, "content"
                    )
                    if response_obj and has_content:
                        output_content = response_obj.choices[0].message.content

                    usage_data = None
                    if hasattr(response_obj, "usage") and response_obj.usage is not None:
                        usage_data = litellm.utils.Usage(
                            prompt_tokens=getattr(response_obj.usage, "prompt_tokens", 0),
                            completion_tokens=getattr(
                                response_obj.usage, "completion_tokens", 0
                            ),
                            # cost might be available on response_obj directly or needs calculation
                        )
                    current_generation.end(output=output_content, usage=usage_data)
                except Exception as e:
                    logger.error(
                        f"Langfuse: Error ending generation '{generation_name}': {e}"
                    )
            yield response_obj
            return # Success, exit the retry loop

        except (asyncio.TimeoutError, litellm.exceptions.APIConnectionError, litellm.exceptions.RateLimitError, litellm.exceptions.APIError, AttributeError) as e:
            last_exception = e
            retries += 1
            if retries <= config.service.max_retries:
                logger.warning(
                    f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) failed on attempt {retries} with {type(e).__name__}. Retrying in {config.service.retry_delay_seconds} seconds..."
                )
                if current_generation:
                     # Log retry attempt in Langfuse
                     try:
                         current_generation.update(
                             level="WARNING",
                             status_message=f"Retry {retries}/{config.service.max_retries}: {type(e).__name__}",
                         )
                     except Exception as langfuse_e:
                         logger.error(f"Langfuse: Error updating generation for retry: {langfuse_e}")

                await asyncio.sleep(config.service.retry_delay_seconds)
            else:
                logger.error(
                    f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) failed after {config.service.max_retries} retries with {type(e).__name__}.",
                    exc_info=True,
                )
                if current_generation:
                    current_generation.end(
                        level="ERROR", status_message=f"Failed after retries: {type(e).__name__}"
                    )
                # Do not yield anything on final failure, the generator simply stops.
                return

        except Exception as e:
            # Catch any other unexpected errors immediately
            logger.error(
                f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) call failed with an unexpected error: {e}",
                exc_info=True,
            )
            if current_generation:
                current_generation.end(
                    level="ERROR", status_message=f"Unexpected Error: {type(e).__name__}"
                )
            # Do not yield anything on unexpected failure, the generator simply stops.
            return

    # If the loop finishes without success (shouldn't happen with the final return in try block),
    # it means all retries failed. The last_exception will contain the reason.
    # The generator will simply stop here.
    if last_exception and current_generation and current_generation.status != "ERROR":
         # Ensure Langfuse generation is marked as failed if it's not already
         current_generation.end(
             level="ERROR", status_message=f"Failed after retries: {type(last_exception).__name__}"
         )

    return # Ensure the generator always returns/stops
