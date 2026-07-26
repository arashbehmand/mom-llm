"""LiteLLM adapter — the ONLY module that imports ``litellm``.

Normalizes provider responses/chunks into the domain's ``Completion`` / ``CompletionChunk`` and
usage into ``Usage`` (including provider prompt-cache tokens). This first version covers the chat
route; the Responses route and the provider-specific streaming quirks are layered on later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.results import Usage


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)
    cached = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
    reasoning = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0) if completion_details else 0
    )
    cache_write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cached_prompt_tokens=cached,
        cache_write_tokens=cache_write,
    )


def _call_params(spec: CallSpec) -> dict[str, Any]:
    params = dict(spec.params)
    params["model"] = spec.model
    params["messages"] = spec.messages
    if spec.timeout_seconds is not None:
        params["timeout"] = spec.timeout_seconds
    return params


class LiteLLMClient:
    """A single-model transport backed by ``litellm.acompletion``."""

    async def complete(self, spec: CallSpec) -> Completion:
        import litellm

        try:
            response = await litellm.acompletion(stream=False, **_call_params(spec))
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} call failed") from exc

        choice = response.choices[0]
        message = choice.message
        return Completion(
            content=message.content or "",
            reasoning=getattr(message, "reasoning_content", None),
            finish_reason=choice.finish_reason or "stop",
            usage=_usage(getattr(response, "usage", None)),
        )

    async def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        import litellm

        params = _call_params(spec)
        params["stream_options"] = {"include_usage": True}
        try:
            stream = await litellm.acompletion(stream=True, **params)
            async for part in stream:
                choices = getattr(part, "choices", None) or []
                delta = choices[0].delta if choices else None
                finish = choices[0].finish_reason if choices else None
                usage_raw = getattr(part, "usage", None)
                yield CompletionChunk(
                    content=getattr(delta, "content", None) if delta else None,
                    reasoning=getattr(delta, "reasoning_content", None) if delta else None,
                    finish_reason=finish,
                    usage=_usage(usage_raw) if usage_raw else None,
                )
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} stream failed") from exc
