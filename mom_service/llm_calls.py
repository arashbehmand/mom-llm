import os
import asyncio
import logging
from typing import List, Dict, Any, Optional

import litellm
# No type imports from langfuse; use Any for trace/generation
from .config import LLMDefinition
logger = logging.getLogger(__name__)

async def _call_lite_llm(
    llm_cfg: LLMDefinition,
    messages: List[Dict[str, Any]],
    timeout_val: int,
    call_type: str = "Fan-out",
    trace: Optional[Any] = None, # Langfuse trace object, type unknown
    generation_name: Optional[str] = None # Name for this specific generation in Langfuse
) -> Optional[litellm.ModelResponse]:
    logger.info(f"--- _call_lite_llm attempting to call: {llm_cfg.name} as {call_type} ---") # Added test log
    """
    Helper function to make an asynchronous call to an LLM using LiteLLM.
    Includes timeout, basic error handling, and optional Langfuse tracing.
    Returns the full ModelResponse object on success, None on failure.
    """
    api_key = os.getenv(llm_cfg.api_key_env)
    if not api_key:
        logger.error(f"API key env variable {llm_cfg.api_key_env} not set for {call_type} LLM {llm_cfg.name}")
        return None
    
    model_name = llm_cfg.model
    # Provider-specific model name prefixing (e.g., "openai/") is often handled by LiteLLM
    # or should be included in the model name in config.yaml if necessary.
    # Example: if llm_cfg.provider == "openai" and not model_name.startswith("openai/"):
    # model_name = f"openai/{model_name}"
    # For now, assume model_name in config is sufficient or LiteLLM handles it.

    logger.info(f"Calling {call_type} LLM: {llm_cfg.name} (Model: {model_name}) with timeout {timeout_val}s")
    
    current_generation = None
    if trace and generation_name:
        try:
            current_generation = trace.generation(
                name=generation_name,
                input=messages,
                model=model_name,
                metadata={"call_type": call_type, "llm_name": llm_cfg.name, **(llm_cfg.params or {})},
            )
        except Exception as e:
            logger.error(f"Langfuse: Error creating generation '{generation_name}': {e}")
            current_generation = None

    try:
        response_obj = await asyncio.wait_for(
            litellm.acompletion(
                model=model_name,
                messages=messages,
                api_key=api_key,
                **(llm_cfg.params or {})
            ),
            timeout=timeout_val
        )
        logger.info(f"{call_type} LLM {llm_cfg.name} call successful.")
        
        if current_generation:
            try:
                output_content = None
                if response_obj and hasattr(response_obj, 'choices') and response_obj.choices and \
                   hasattr(response_obj.choices[0], 'message') and response_obj.choices[0].message and \
                   hasattr(response_obj.choices[0].message, 'content'):
                    output_content = response_obj.choices[0].message.content
                
                usage_data = None
                if hasattr(response_obj, "usage") and response_obj.usage is not None:
                     usage_data = litellm.utils.Usage(
                        prompt_tokens=getattr(response_obj.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(response_obj.usage, "completion_tokens", 0)
                    )

                current_generation.end(
                    output=output_content,
                    usage=usage_data
                )
            except Exception as e:
                logger.error(f"Langfuse: Error ending generation '{generation_name}': {e}")
        return response_obj
    except asyncio.TimeoutError:
        logger.warning(f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) call timed out after {timeout_val} seconds.")
        if current_generation: current_generation.end(level="WARNING", status_message="Timeout")
        return None
    except litellm.exceptions.APIConnectionError as e:
        logger.error(f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) API Connection Error: {e}", exc_info=True)
        if current_generation: current_generation.end(level="ERROR", status_message=f"API Connection Error: {e}")
        return None
    except litellm.exceptions.RateLimitError as e:
        logger.error(f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) Rate Limit Error: {e}", exc_info=True)
        if current_generation: current_generation.end(level="ERROR", status_message=f"Rate Limit Error: {e}")
        return None
    except litellm.exceptions.APIError as e: 
        logger.error(f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) LiteLLM API Error: {e}", exc_info=True)
        if current_generation: current_generation.end(level="ERROR", status_message=f"LiteLLM API Error: {e}")
        return None
    except Exception as e: 
        logger.error(f"{call_type} LLM {llm_cfg.name} (Model: {model_name}) call failed with an unexpected error: {e}", exc_info=True)
        if current_generation: current_generation.end(level="ERROR", status_message=f"Unexpected Error: {e}")
        return None
