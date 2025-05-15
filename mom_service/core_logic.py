import asyncio
import html
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple # Import AsyncGenerator, Tuple

import litellm

from .config import LLMDefinition, MoMConfig
from .endpoints.models import ThinkingContextItem, UsageInfo
from .llm_calls import _call_lite_llm

logger = logging.getLogger(__name__)

# New helper function
async def _call_and_return_with_def(llm_def: LLMDefinition, call_coroutine: AsyncGenerator[Any, None]) -> Tuple[LLMDefinition, Any]:
    """Helper to await a generator expected to yield one item and return it with the LLMDefinition."""
    result = None
    try:
        async for item in call_coroutine:
            result = item
            break # We expect only one item for non-streaming fanout
    except Exception as e:
        # If an exception occurs within the LLM call, return the definition and the exception
        return (llm_def, e)

    return (llm_def, result)


async def _perform_fanout_calls(
    model_conf: MoMConfig,
    llm_map: Dict[str, LLMDefinition],
    request_messages: List[Dict[str, Any]],
    timeout: int,
    trace: Optional[Any] = None,
) -> AsyncGenerator[ThinkingContextItem, None]:
    """
    Perform fan-out LLM calls and yield intermediate thinking context items as they complete.
    """
    tasks = []

    for idx, llm_name_to_query in enumerate(model_conf.llms_to_query):
        ld = llm_map.get(llm_name_to_query)
        if not ld:
            logger.warning(f"Fan-out LLMDefinition '{llm_name_to_query}' not found.")
            continue

        gen_name = f"fanout-{idx}-{ld.name}" if trace else None

        # Create the coroutine for the LLM call (non-streaming for fanout)
        llm_call_coroutine = _call_lite_llm(
            ld,
            request_messages,
            timeout,
            options={
                "call_type": "fanout",
                "trace": trace,
                "generation_name": gen_name,
                "stream": False,  # Explicitly False for fanout
            },
        )

        # Create a task for the helper that calls the LLM and returns the definition with the result/exception
        task = asyncio.create_task(_call_and_return_with_def(ld, llm_call_coroutine))
        tasks.append(task)

    # Use asyncio.as_completed to yield results as they finish
    for future in asyncio.as_completed(tasks):
        cost = None
        content_str = ""
        current_res_obj_fanout = None
        ld_fanout = None # Initialize ld_fanout

        # Initialize usage info to a default value *before* processing the future
        current_usage_info = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0)

        try:
            # Await the future to get the tuple (llm_def, result_or_exception)
            ld_fanout, result_or_exception = await future

            if isinstance(result_or_exception, Exception):
                 # An exception occurred within the helper coroutine (during the LLM call)
                 content_str = f"Error: Call to {ld_fanout.model} failed. Details: {html.escape(str(result_or_exception))}"
                 logger.error(f"Fan-out call to {ld_fanout.name} failed: {result_or_exception}")
                 # current_usage_info is already initialized to default
            else:
                 # The LLM call was successful, result_or_exception is the response object
                 current_res_obj_fanout = result_or_exception

        except Exception as e:
             # This catches exceptions from asyncio.as_completed itself or unexpected errors
             # within the loop, less likely but good to have.
             # If ld_fanout is None here, it's a more general error.
            if ld_fanout:
                 content_str = f"Error: An unexpected error occurred processing result for {ld_fanout.model}. Details: {html.escape(str(e))}"
                 logger.error(f"Unexpected error processing result for {ld_fanout.name}: {e}")
            else:
                 content_str = f"Error: An unexpected error occurred processing a fan-out result. Details: {html.escape(str(e))}"
                 logger.error(f"An unexpected error occurred processing an unattributed fan-out result: {e}")
            # current_usage_info is already initialized to default


        # Process the result if the call was successful (current_res_obj_fanout is not None)
        if current_res_obj_fanout is not None:
            if (
                hasattr(current_res_obj_fanout, "choices")
                and current_res_obj_fanout.choices
                and hasattr(current_res_obj_fanout.choices[0], "message")
                and current_res_obj_fanout.choices[0].message
                and hasattr(current_res_obj_fanout.choices[0].message, "content")
            ):
                content_str = current_res_obj_fanout.choices[0].message.content
                try:
                    cost = litellm.completion_cost(
                        completion_response=current_res_obj_fanout
                    )
                except Exception:
                    cost = None  # Error calculating cost

                usage_data = current_res_obj_fanout.usage or {}
                current_usage_info = UsageInfo(
                    prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                    completion_tokens=getattr(usage_data, "completion_tokens", 0),
                    total_tokens=getattr(usage_data, "total_tokens", 0),
                    cost=cost,
                )
            else:
                # Response object was malformed or unexpected, but no exception was raised by the call itself
                if not content_str: # Only set if no error message was already captured by an exception
                    content_str = f"Warning: Call to {ld_fanout.model} returned an empty or malformed response."
                # current_usage_info is already initialized to default

        # Yield the ThinkingContextItem as soon as it's ready
        # Ensure ld_fanout is available before yielding
        if ld_fanout:
             logger.info(f"DEBUG: Yielding ThinkingContextItem for {ld_fanout.model} with usage: {current_usage_info} (type: {type(current_usage_info)})") # Add debug print
             yield ThinkingContextItem(
                 model=ld_fanout.model, content=content_str, usage=current_usage_info
             )
        else:
             # If ld_fanout is None, it means the exception happened before we could get the definition.
             # We still yield an item, but with a generic model name indicating an error.
             logger.info(f"DEBUG: Yielding ThinkingContextItem for unknown_error_model with usage: {current_usage_info} (type: {type(current_usage_info)})") # Add debug print
             yield ThinkingContextItem(
                 model="unknown_error_model", content=content_str, usage=current_usage_info
             )


# Keep other functions as they are
# _prepare_concluding_messages
# _execute_concluding_call
# _calculate_and_log_costs


def _prepare_concluding_messages(
    request_messages: List[Dict[str, Any]],
    intermediate_thinking_context: List[ThinkingContextItem],
    model_conf: MoMConfig,
    config: MoMConfig,
) -> List[Dict[str, Any]]:
    """
    Prepare messages for the concluding LLM.
    """
    concl_msgs_for_llm = list(request_messages)
    concl_msgs_for_llm.append({"role": "user", "content": "<<<<<<>>>>>>"})  # Separator
    for item_ctx in intermediate_thinking_context:
        if (
            item_ctx.usage.cost is not None
            and not item_ctx.content.startswith("Error:")
            and not item_ctx.content.startswith("Warning:")
        ):
            concl_msgs_for_llm.append(
                {"role": "assistant", "content": item_ctx.content}
            )

    if model_conf.concluding_prompt:
        prompt_defs = (
            config.prompt_definitions
            if isinstance(config.prompt_definitions, list)
            else ([config.prompt_definitions] if config.prompt_definitions else [])
        )
        if prompt_defs:
            prompt_def_list = [
                p for p in prompt_defs if p.name == model_conf.concluding_prompt
            ]
            if prompt_def_list:
                concl_msgs_for_llm.append(
                    {"role": "user", "content": prompt_def_list[0].content}
                )
    return concl_msgs_for_llm


async def _execute_concluding_call(
    concl_def: LLMDefinition,
    concl_msgs_for_llm: List[Dict[str, Any]],
    timeout: int,
    options: Optional[dict] = None,
) -> Any:
    """
    Execute the concluding LLM call (streaming or not).
    """
    options = options or {}
    trace = options.get("trace")
    gen_name_concl = options.get("gen_name_concl")
    stream = options.get("stream", False)

    if stream:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "_execute_concluding_call: Calling _call_lite_llm for stream=True to get generator"
        )

        the_generator = _call_lite_llm(
            concl_def,
            concl_msgs_for_llm,
            timeout,
            options={
                "call_type": "concluding",
                "trace": trace,
                "generation_name": gen_name_concl,
                "stream": True,
            },
        )
        logger.info(
            f"_execute_concluding_call: Returning generator of type {type(the_generator)}"
        )
        return the_generator
    # For non-streaming, _call_lite_llm returns an async generator that yields one item
    concl_res_obj = None
    async for item in _call_lite_llm(
        concl_def,
        concl_msgs_for_llm,
        timeout,
        options={
            "call_type": "concluding",
            "trace": trace,
            "generation_name": gen_name_concl,
            "stream": False,
        },
    ):
        concl_res_obj = item
        break
    return concl_res_obj


def _calculate_and_log_costs(
    fanout_context: List[ThinkingContextItem],
    concluding_llm_usage_info: UsageInfo,
) -> float:
    """
    Calculate total cost from fan-out and concluding LLM usage.
    """
    total_cost_accumulator = 0.0
    for item in fanout_context:
        if item.usage and item.usage.cost is not None:
            total_cost_accumulator += item.usage.cost
    if concluding_llm_usage_info and concluding_llm_usage_info.cost is not None:
        total_cost_accumulator += concluding_llm_usage_info.cost
    return round(total_cost_accumulator, 8) if total_cost_accumulator > 0 else 0.0
