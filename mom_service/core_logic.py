"""
Core orchestration logic for the Mixture of Models (MoM) service.

This module implements the fan-out/fan-in pattern for LLM orchestration:
1. Fan-out: Sends requests to multiple LLMs in parallel (query phase)
2. Aggregation: Collects and processes intermediate responses
3. Concluding: Synthesizes final response using a concluding LLM

Key functions:
- _perform_fanout_calls: Execute parallel LLM calls and yield thinking context items
- _prepare_concluding_messages: Prepare messages for the concluding LLM based on fan-out results
- _execute_concluding_call: Execute the final synthesis LLM call
- _calculate_and_log_costs: Aggregate costs from all LLM calls

The module handles error cases gracefully, including partial failures where some
fan-out calls succeed while others fail, ensuring the service remains resilient.
"""

import asyncio
import html
import logging

# anext is available in Python 3.10+
from builtins import anext
from collections.abc import AsyncGenerator
from typing import Any, Optional

from .config import LLMDefinition
from .config import ModelConfig as MoMModelConfig
from .config import MoMConfig
from .endpoints.models import LLMCallParams, ThinkingContextItem, UsageInfo
from .llm_calls import _call_lite_llm

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments,too-many-positional-arguments
async def call_llm(
    llm_def: LLMDefinition,
    params_obj: LLMCallParams,
    timeout: int,
    config: MoMConfig,
    options: Optional[dict] = None,
) -> AsyncGenerator[Any, None]:
    """
    Wrapper that accepts an LLMCallParams object and delegates to the low-level
    _call_lite_llm function. This reduces the number of positional/keyword
    arguments passed around in call sites.
    """
    options = options or {}
    # Ensure the stream flag from params_obj is available to the underlying call
    options = {**options, "stream": params_obj.stream}
    # Delegate to the low-level call with messages from params_obj
    async for item in _call_lite_llm(
        llm_def,
        params_obj.messages,
        timeout,
        config,
        options=options,
    ):
        yield item


async def _perform_fanout_calls(
    model_conf: "MoMModelConfig",
    llm_map: dict[str, LLMDefinition],
    request_messages: list[dict[str, Any]],
    timeout: int,
    config: MoMConfig,
    options: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[ThinkingContextItem, None]:
    """
    Perform fan-out LLM calls and yield intermediate thinking context items as they complete.
    Handles exceptions within the tasks gracefully.
    Automatically filters models based on multimodal content capability.
    options (dict) may contain 'trace' and 'request_id'.
    """

    # Helper to wrap the coroutine and capture the LLMDefinition
    async def call_and_return_with_def(
        llm_def: LLMDefinition, call_coroutine: AsyncGenerator[Any, None]
    ) -> tuple[LLMDefinition, Any]:
        try:
            # For non-streaming fanout, we expect one result or nothing on error
            result = await anext(call_coroutine, None)
            return (llm_def, result)
        except Exception as e:
            # If the LLM call itself raises an exception, capture it here
            return (llm_def, e)

    options = options or {}
    trace = options.get("trace")
    request_id = options.get("request_id", "unknown")

    # Filter models based on multimodal capability
    from .multimodal_utils import filter_multimodal_capable_models

    filtered_llms, skipped_llms = filter_multimodal_capable_models(
        model_conf.llms_to_query, llm_map, request_messages
    )

    # Yield warning items for skipped models
    for skipped_llm_name in skipped_llms:
        skipped_def = llm_map.get(skipped_llm_name)
        model_display = skipped_def.model if skipped_def else skipped_llm_name
        yield ThinkingContextItem(
            model=model_display,
            content="Warning: Model skipped due to lack of multimodal support for this request.",
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0),
        )

    tasks = []
    for idx, llm_name_to_query in enumerate(filtered_llms):
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
        # Use LLMCallParams to encapsulate model/messages/stream settings
        llm_call_params = LLMCallParams(
            model=ld.model,
            messages=request_messages,
            temperature=ld.params.get("temperature") if isinstance(ld.params, dict) else None,
            max_tokens=ld.params.get("max_tokens") if isinstance(ld.params, dict) else None,
            top_p=ld.params.get("top_p") if isinstance(ld.params, dict) else None,
            stream=False,
        )
        llm_call_generator = call_llm(
            ld,
            llm_call_params,
            timeout,
            config,
            options={
                "call_type": "fanout",
                "trace": trace,
                "generation_name": gen_name,
                "request_id": request_id,
                "mom_model_name": model_conf.name,
            },
        )
        task = asyncio.create_task(call_and_return_with_def(ld, llm_call_generator))
        tasks.append(task)

    for future in asyncio.as_completed(tasks):
        ld_fanout, result_or_exception = await future
        content_str = ""
        current_usage_info = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        if isinstance(result_or_exception, Exception):
            content_str = f"Error: Call to {ld_fanout.model} failed. Details: {html.escape(str(result_or_exception))}"
            logger.error(f"Fan-out call to {ld_fanout.name} failed: {result_or_exception}")
        elif result_or_exception is None:
            content_str = f"Error: Call to {ld_fanout.model} returned no response (likely due to an internal error or timeout)."
            logger.error(f"Fan-out call to {ld_fanout.name} returned None.")
        else:
            current_res_obj_fanout = result_or_exception
            has_choices = hasattr(current_res_obj_fanout, "choices") and bool(
                current_res_obj_fanout.choices
            )
            has_message = (
                has_choices
                and hasattr(current_res_obj_fanout.choices[0], "message")
                and bool(current_res_obj_fanout.choices[0].message)
            )
            has_content = (
                has_message
                and hasattr(current_res_obj_fanout.choices[0].message, "content")
                and current_res_obj_fanout.choices[0].message.content is not None
            )
            if has_choices and has_message and has_content:
                content_str = current_res_obj_fanout.choices[0].message.content
                # Use from_litellm_usage helper for consistent cost tracking
                # Check if response is cached
                is_cached = getattr(current_res_obj_fanout, "_is_cached", False)
                current_usage_info = UsageInfo.from_litellm_usage(
                    current_res_obj_fanout.usage,
                    response_obj=current_res_obj_fanout,
                    is_cached=is_cached,
                    pricing_config=ld_fanout.pricing,
                )
            else:
                content_str = (
                    f"Warning: Call to {ld_fanout.model} returned an empty or malformed response."
                )

        yield ThinkingContextItem(
            model=ld_fanout.model, content=str(content_str), usage=current_usage_info
        )


def _prepare_concluding_messages(
    request_messages: list[dict[str, Any]],
    intermediate_thinking_context: list[ThinkingContextItem],
    model_conf: "MoMModelConfig",
    config: MoMConfig,
) -> list[dict[str, Any]]:
    """
    Prepare messages for the concluding LLM.
    """
    concl_msgs_for_llm = list(request_messages)
    # Filter out unsuccessful fan-out results before appending to the context
    successful_fanout_items = [
        item
        for item in intermediate_thinking_context
        if not item.content.startswith("Error:") and not item.content.startswith("Warning:")
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
            concl_msgs_for_llm.append({"role": "assistant", "content": item_ctx.content})

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
    concl_msgs_for_llm: list[dict[str, Any]],
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

    # Ensure call_type is set for concluding calls
    if "call_type" not in options:
        options["call_type"] = "concluding"

    # Use LLMCallParams wrapper for concluding calls
    concl_params = LLMCallParams(
        model=concl_def.model,
        messages=concl_msgs_for_llm,
        temperature=(
            concl_def.params.get("temperature") if isinstance(concl_def.params, dict) else None
        ),
        max_tokens=(
            concl_def.params.get("max_tokens") if isinstance(concl_def.params, dict) else None
        ),
        top_p=concl_def.params.get("top_p") if isinstance(concl_def.params, dict) else None,
        stream=stream,
    )
    llm_call_generator = call_llm(concl_def, concl_params, timeout, config, options=options)

    if stream:
        return llm_call_generator
    return await anext(llm_call_generator, None)


def _calculate_and_log_costs(
    fanout_context: list[ThinkingContextItem],
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
