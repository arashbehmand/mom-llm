import asyncio
import html
from typing import Any, Dict, List, Optional

import litellm

from .config import LLMDefinition, MoMConfig
from .endpoints.models import ThinkingContextItem, UsageInfo
from .llm_calls import _call_lite_llm


async def _perform_fanout_calls(
    model_conf: MoMConfig,
    llm_map: Dict[str, LLMDefinition],
    request_messages: List[Dict[str, Any]],
    timeout: int,
    trace: Optional[Any] = None,
) -> List[ThinkingContextItem]:
    """
    Perform fan-out LLM calls and collect intermediate thinking context.
    """
    tasks = []
    fanout_llm_defs_in_order = []

    async def _get_single_item_from_gen(gen_func_call):
        """Helper to consume an async generator expected to yield one item."""
        async for item in gen_func_call:
            return item  # Return the first (and only expected) item
        return (
            None  # Should not happen if _call_lite_llm (non-stream) works as expected
        )

    for idx, llm_name_to_query in enumerate(model_conf.llms_to_query):
        ld = llm_map.get(llm_name_to_query)
        if not ld:
            continue
        fanout_llm_defs_in_order.append(ld)
        gen_name = f"fanout-{idx}-{ld.name}" if trace else None

        # _call_lite_llm returns an async generator.
        # We wrap its consumption in a coroutine for asyncio.gather.
        # For fanout, stream is implicitly False.
        tasks.append(
            _get_single_item_from_gen(
                _call_lite_llm(
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
            )
        )

    # Gather results from the wrapper coroutines
    fanout_results_objects = await asyncio.gather(*tasks, return_exceptions=True)

    intermediate_thinking_context: List[ThinkingContextItem] = []
    # fanout_results_objects now contains the actual response objects or exceptions
    for ld_fanout, current_res_obj_fanout in zip(
        fanout_llm_defs_in_order, fanout_results_objects
    ):  # Renamed to avoid conflict
        cost = None
        content_str = ""
        # res_obj_fanout is now current_res_obj_fanout from the zip

        if isinstance(
            current_res_obj_fanout, Exception
        ):  # Check if the result from gather is an exception
            content_str = f"Error: Call to {ld_fanout.model} failed. Details: {html.escape(str(current_res_obj_fanout))}"
            # current_res_obj_fanout is already the exception
        elif (
            current_res_obj_fanout is None
        ):  # Wrapper returned None (generator was empty)
            content_str = f"Warning: Call to {ld_fanout.model} returned no response (empty generator)."
        elif (
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

            usage_data = current_res_obj_fanout.usage or {}  # Corrected variable name
            current_usage_info = UsageInfo(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                completion_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
                cost=cost,
            )
            intermediate_thinking_context.append(
                ThinkingContextItem(
                    model=ld_fanout.model, content=content_str, usage=current_usage_info
                )
            )
        else:  # Handles errors from above or if res_obj_fanout is None/malformed
            if not content_str:  # If no specific error message was set yet
                content_str = f"Warning: Call to {ld_fanout.model} returned an empty or malformed response."
            usage_data_error = UsageInfo(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
            )
            intermediate_thinking_context.append(
                ThinkingContextItem(
                    model=ld_fanout.model, content=content_str, usage=usage_data_error
                )
            )
    return intermediate_thinking_context


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
