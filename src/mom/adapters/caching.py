"""Caching middleware: a response cache in front of any ``LLMClient`` (non-streaming only).

Fan-out member calls are cached; the synthesizer stream is not. Cache hits cost $0 and are marked
so the pipeline records them as cache hits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

from mom.domain.cachekey import cache_key
from mom.domain.ports import CacheStore, CallSpec, Clock, Completion, CompletionChunk, LLMClient
from mom.domain.results import Usage


def _serialize(completion: Completion) -> str:
    usage = completion.usage
    return json.dumps(
        {
            "content": completion.content,
            "reasoning": completion.reasoning,
            "finish_reason": completion.finish_reason,
            "tool_calls": list(completion.tool_calls),
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            },
        }
    )


def _deserialize(body: str) -> Completion:
    data = json.loads(body)
    usage = data.get("usage", {})
    return Completion(
        content=data["content"],
        reasoning=data.get("reasoning"),
        finish_reason=data.get("finish_reason", "stop"),
        usage=Usage(**usage),
        tool_calls=tuple(data.get("tool_calls", ())),
        cached=True,
    )


class CachingClient:
    """Wraps an ``LLMClient`` with a response cache for non-streaming calls."""

    def __init__(self, inner: LLMClient, cache: CacheStore, clock: Clock) -> None:
        self._inner = inner
        self._cache = cache
        self._clock = clock

    async def complete(self, spec: CallSpec) -> Completion:
        key = cache_key(
            llm_name=spec.llm_name,
            model=spec.model,
            messages=spec.messages,
            params=spec.params,
        )
        now = self._clock.now()
        hit = await self._cache.get(key, now=now)
        if hit is not None:
            return _deserialize(hit)
        result = await self._inner.complete(spec)
        # Do not cache tool-call, empty, or truncated results — a `length`-truncated answer would
        # otherwise be served for every future identical request.
        if not result.tool_calls and result.content.strip() and result.finish_reason != "length":
            await self._cache.put(key, spec.llm_name, _serialize(result), now=now)
        return result

    def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        return self._inner.stream(spec)
