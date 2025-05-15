import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm

# No type imports from langfuse; use Any for trace/generation
from .config import LLMDefinition

logger = logging.getLogger(__name__)


async def _call_lite_llm(
    llm_cfg: LLMDefinition,
    messages: List[Dict[str, Any]],
    timeout_val: int,
    options: Optional[dict] = None,
) -> AsyncGenerator[litellm.ModelResponse, None]:
    """
    Helper function to make an asynchronous call to an LLM using LiteLLM.
    Includes timeout, basic error handling, and optional Langfuse tracing.
    Always returns an async generator. For non-streaming, it yields one item then stops.
    For streaming, it yields chunks. On error, it yields nothing and stops.
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
        return  # MODIFIED: bare return for async generator

    model_name = llm_cfg.model
    logger.info(
        f"Calling {call_type} LLM: {llm_cfg.name} (Model: {model_name}) with timeout {timeout_val}s"
    )

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
                    **(llm_cfg.params or {}),
                },
            )
        except Exception as e:
            logger.error(
                f"Langfuse: Error creating generation '{generation_name}': {e}"
            )
            current_generation = None

    if stream:
        logger.info(
            f"Calling {call_type} LLM in streaming mode: {llm_cfg.name} (Model: {model_name})"
        )
        try:
            # Await the acompletion call to get the async generator object
            async_generator = await litellm.acompletion(
                model=model_name,
                messages=messages,
                api_key=api_key,
                stream=True,
                **(llm_cfg.params or {}),
            )
            
            # Track the accumulated content for Langfuse
            streamed_content = ""
            
            # Now iterate over the obtained async generator
            async for chunk in async_generator:
                # For OpenAI-like responses, extract and accumulate content
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and hasattr(delta, "content") and delta.content is not None:
                        streamed_content += delta.content
                yield chunk
            
            # Update Langfuse with the complete streamed content
            if current_generation:
                current_generation.end(
                    output=streamed_content if streamed_content else "<streamed>", 
                    usage=None
                )
        except Exception as e:
            logger.error(
                f"Streaming call failed for {llm_cfg.name} (Model: {model_name}): {e}"
            )
            if current_generation:
                current_generation.end(
                    level="ERROR", status_message=f"Streaming Error: {e}"
                )
        # The async for loop will naturally complete, no need for a final return in the generator.

    # Non-streaming path
    try:
        response_obj = await asyncio.wait_for(
            litellm.acompletion(
                model=model_name,
                messages=messages,
                api_key=api_key,
                **(llm_cfg.params or {}),
            ),
            timeout=timeout_val,
        )
        logger.info(f"{call_type} LLM {llm_cfg.name} call successful.")

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
                    usage_data = litellm.utils.Usage(  # Ensure this is the correct Usage object for Langfuse
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
        yield response_obj  # MODIFIED: yield response_obj
        return  # MODIFIED: bare return
    except asyncio.TimeoutError:
        logger.warning(
            f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) call timed out after {timeout_val} seconds."
        )
        if current_generation:
            current_generation.end(level="WARNING", status_message="Timeout")
        return  # MODIFIED: bare return
    except litellm.exceptions.APIConnectionError as e:
        logger.error(
            f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) API Connection Error: {e}",
            exc_info=True,
        )
        if current_generation:
            current_generation.end(
                level="ERROR", status_message=f"API Connection Error: {e}"
            )
        return  # MODIFIED: bare return
    except litellm.exceptions.RateLimitError as e:
        logger.error(
            f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) Rate Limit Error: {e}",
            exc_info=True,
        )
        if current_generation:
            current_generation.end(
                level="ERROR", status_message=f"Rate Limit Error: {e}"
            )
        return  # MODIFIED: bare return
    except litellm.exceptions.APIError as e:
        logger.error(
            f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) LiteLLM API Error: {e}",
            exc_info=True,
        )
        if current_generation:
            current_generation.end(
                level="ERROR", status_message=f"LiteLLM API Error: {e}"
            )
        return  # MODIFIED: bare return
    except Exception as e:
        logger.error(
            f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) call failed with an unexpected error: {e}",
            exc_info=True,
        )
        if current_generation:
            current_generation.end(
                level="ERROR", status_message=f"Unexpected Error: {e}"
            )
        return  # MODIFIED: bare return
