"""
Helpers for routing non-streaming calls through LiteLLM Responses API.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import litellm
from litellm.utils import Choices, Message, ModelResponse, Usage

from .config import LLMDefinition, MoMConfig


def _details_obj_to_dict(details_obj: Any) -> dict[str, Any]:
    if details_obj is None:
        return {}
    if isinstance(details_obj, dict):
        return {k: v for k, v in details_obj.items() if v is not None}
    try:
        return {k: v for k, v in vars(details_obj).items() if v is not None}
    except (TypeError, AttributeError):
        return {}


def _extract_responses_output_text(responses_obj: Any) -> str:
    output_text = getattr(responses_obj, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output_items = getattr(responses_obj, "output", None)
    if not isinstance(output_items, list):
        return ""

    text_parts: list[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        content_items = getattr(item, "content", None) or []
        if not isinstance(content_items, list):
            continue
        for content_item in content_items:
            text_value = getattr(content_item, "text", None)
            if isinstance(text_value, str):
                text_parts.append(text_value)

    return "".join(text_parts)


def _convert_responses_usage_to_litellm_usage(responses_usage: Any) -> Usage | None:
    if responses_usage is None:
        return None

    prompt_details = _details_obj_to_dict(getattr(responses_usage, "input_tokens_details", None))
    completion_details = _details_obj_to_dict(
        getattr(responses_usage, "output_tokens_details", None)
    )

    num_sources_used = getattr(responses_usage, "num_sources_used", None)
    if num_sources_used is not None:
        prompt_details["web_search_requests"] = num_sources_used

    return Usage(
        prompt_tokens=getattr(responses_usage, "input_tokens", None),
        completion_tokens=getattr(responses_usage, "output_tokens", None),
        total_tokens=getattr(responses_usage, "total_tokens", None),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        prompt_tokens_details=(prompt_details or None),
        completion_tokens_details=(completion_details or None),
    )


def _convert_responses_to_model_response(
    llm_def: LLMDefinition, responses_obj: Any
) -> ModelResponse:
    created_at_raw = getattr(responses_obj, "created_at", None)
    try:
        created_at = int(created_at_raw) if created_at_raw is not None else int(time.time())
    except (TypeError, ValueError):
        created_at = int(time.time())

    content = _extract_responses_output_text(responses_obj)
    usage = _convert_responses_usage_to_litellm_usage(getattr(responses_obj, "usage", None))

    return ModelResponse(
        id=getattr(responses_obj, "id", None),
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
        created=created_at,
        model=llm_def.model,
        usage=usage,
        object="chat.completion",
    )


def _flatten_message_content_to_strings(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten list-of-parts content back to plain strings for the Responses API.

    Some providers (e.g. XAI) reject the list-of-content-parts format
    (``[{"type": "text", "text": "..."}]``) that OpenAI chat completions
    accept.  When all parts in a message are text-only we collapse them
    into a single plain string.  Messages that contain non-text parts
    (e.g. images) are left unchanged.

    TODO: Remove this workaround once LiteLLM handles the normalization
    upstream (see: https://github.com/BerriAI/litellm/issues/XXXXX).
    """
    has_list_content = any(isinstance(m.get("content"), list) for m in messages)
    if not has_list_content:
        return messages

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and all(
            isinstance(p, dict) and p.get("type") == "text" for p in content
        ):
            # Only flatten if every part is a text part
            msg = {**msg, "content": "".join(p.get("text", "") for p in content)}
        normalized.append(msg)
    return normalized


def _build_responses_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": params["model"],
        "input": _flatten_message_content_to_strings(params["messages"]),
        "stream": False,
        "timeout": params.get("timeout"),
    }

    if params.get("api_key"):
        kwargs["api_key"] = params["api_key"]

    for key in ("tools", "tool_choice", "temperature", "top_p", "user"):
        if key in params and params[key] is not None:
            kwargs[key] = params[key]

    max_output_tokens = params.get("max_completion_tokens")
    if max_output_tokens is None:
        max_output_tokens = params.get("max_tokens")
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    return kwargs


async def call_responses_non_stream(
    llm_def: LLMDefinition,
    params: dict[str, Any],
    config: MoMConfig,
) -> ModelResponse:
    responses_kwargs = _build_responses_kwargs(params)
    max_attempts = max(1, config.service.max_llm_retries + 1)
    retry_delay = max(0, config.service.llm_retry_delay_seconds)

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            responses_obj = await litellm.aresponses(**responses_kwargs)
            return _convert_responses_to_model_response(llm_def, responses_obj)
        except Exception as exc:  # pragma: no cover - only exercised in integration/runtime
            last_exc = exc
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(retry_delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected responses API call failure without exception.")
