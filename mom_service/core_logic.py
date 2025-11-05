import asyncio
import html
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

# anext is available in Python 3.10+
from builtins import anext

from .config import LLMDefinition, MoMConfig, ModelConfig as MoMModelConfig
from .endpoints.models import ThinkingContextItem, UsageInfo
from .llm_calls import _call_lite_llm

logger = logging.getLogger(__name__)


async def _perform_fanout_calls(
    model_conf: "MoMModelConfig",
    llm_map: Dict[str, LLMDefinition],
    request_messages: List[Dict[str, Any]],
    timeout: int,
    config: MoMConfig,
    trace: Optional[Any] = None,
) -> AsyncGenerator[ThinkingContextItem, None]:
    """
    Perform fan-out LLM calls and yield intermediate thinking context items as they complete.
    Handles exceptions within the tasks gracefully.
    """

    # Helper to wrap the coroutine and capture the LLMDefinition
    async def call_and_return_with_def(
        llm_def: LLMDefinition, call_coroutine: AsyncGenerator[Any, None]
    ) -> Tuple[LLMDefinition, Any]:
        try:
            # For non-streaming fanout, we expect one result or nothing on error
            result = await anext(call_coroutine, None)
            return (llm_def, result)
        except Exception as e:
            # If the LLM call itself raises an exception, capture it here
            return (llm_def, e)

    tasks = []
    for idx, llm_name_to_query in enumerate(model_conf.llms_to_query):
        ld = llm_map.get(llm_name_to_query)
        if not ld:
            logger.warning(f"Fan-out LLMDefinition '{llm_name_to_query}' not found.")
            # Yield a specific error item for this missing definition
            yield ThinkingContextItem(
                model=f"unknown: {llm_name_to_query}",
                content=f"Error: LLM definition '{llm_name_to_query}' not found in config.yaml.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
            continue

        gen_name = f"fanout-{idx}-{ld.name}" if trace else None
        llm_call_coroutine = _call_lite_llm(
            ld,
            request_messages,
            timeout,
            config,
            options={
                "call_type": "fanout",
                "trace": trace,
                "generation_name": gen_name,
                "stream": False,
            },
        )
        task = asyncio.create_task(call_and_return_with_def(ld, llm_call_coroutine))
        tasks.append(task)

    for future in asyncio.as_completed(tasks):
        ld_fanout, result_or_exception = await future
        content_str = ""
        cost = None
        current_usage_info = UsageInfo(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

        if isinstance(result_or_exception, Exception):
            content_str = f"Error: Call to {ld_fanout.model} failed. Details: {html.escape(str(result_or_exception))}"
            logger.error(
                f"Fan-out call to {ld_fanout.name} failed: {result_or_exception}"
            )
        elif result_or_exception is None:
            content_str = f"Error: Call to {ld_fanout.model} returned no response (likely due to an internal error or timeout)."
            logger.error(f"Fan-out call to {ld_fanout.name} returned None.")
        else:
            current_res_obj_fanout = result_or_exception
            if (
                hasattr(current_res_obj_fanout, "choices")
                and current_res_obj_fanout.choices
                and hasattr(current_res_obj_fanout.choices[0], "message")
                and current_res_obj_fanout.choices[0].message
                and hasattr(current_res_obj_fanout.choices[0].message, "content")
                and current_res_obj_fanout.choices[0].message.content is not None
            ):
                content_str = current_res_obj_fanout.choices[0].message.content
                # Use from_litellm_usage helper for consistent cost tracking
                # Check if response is cached
                is_cached = getattr(current_res_obj_fanout, '_is_cached', False)
                current_usage_info = UsageInfo.from_litellm_usage(
                    current_res_obj_fanout.usage,
                    response_obj=current_res_obj_fanout,
                    is_cached=is_cached
                )
            else:
                content_str = f"Warning: Call to {ld_fanout.model} returned an empty or malformed response."

        yield ThinkingContextItem(
            model=ld_fanout.model, content=str(content_str), usage=current_usage_info
        )


def _prepare_concluding_messages(
    request_messages: List[Dict[str, Any]],
    intermediate_thinking_context: List[ThinkingContextItem],
    model_conf: "MoMModelConfig",
    config: MoMConfig,
) -> List[Dict[str, Any]]:
    """
    Prepare messages for the concluding LLM.
    """
    concl_msgs_for_llm = list(request_messages)
    # Filter out unsuccessful fan-out results before appending to the context
    successful_fanout_items = [
        item
        for item in intermediate_thinking_context
        if not item.content.startswith("Error:")
        and not item.content.startswith("Warning:")
    ]

    if not successful_fanout_items:
        # If no successful items, let the concluding LLM know.
        concl_msgs_for_llm.append(
            {
                "role": "user",
                "content": "For the above content, all initial LLM consultations failed or returned no usable content. Please provide a response based on the original query alone, perhaps with a note about the failure.",
            }
        )
    else:
        concl_msgs_for_llm.append(
            {
                "role": "user",
                "content": "For the above content, we have the following llm responses:\n<<<<<<>>>>>>\n",
            }
        )
        for item_ctx in successful_fanout_items:
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
            prompt_def = next(
                (p for p in prompt_defs if p.name == model_conf.concluding_prompt), None
            )
            if prompt_def:
                concl_msgs_for_llm.append({"role": "user", "content": prompt_def.content})
    return concl_msgs_for_llm


async def _execute_concluding_call(
    concl_def: LLMDefinition,
    concl_msgs_for_llm: List[Dict[str, Any]],
    timeout: int,
    config: MoMConfig,
    options: Optional[dict] = None,
) -> Any:
    """
    Execute the concluding LLM call.
    If streaming, it returns the async generator directly.
    If not streaming, it consumes the generator and returns the single result.
    """
    options = options or {}
    stream = options.get("stream", False)

    llm_call_generator = _call_lite_llm(
        concl_def,
        concl_msgs_for_llm,
        timeout,
        config,
        options=options,
    )

    if stream:
        return llm_call_generator
    else:
        return await anext(llm_call_generator, None)


def _calculate_and_log_costs(
    fanout_context: List[ThinkingContextItem],
    concluding_llm_usage_info: Optional[UsageInfo],
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
