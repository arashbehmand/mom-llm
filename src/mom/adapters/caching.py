"""Caching middleware: a response cache in front of any ``LLMClient`` (non-streaming only).

Fan-out member calls are cached; the synthesizer stream is not. Cache hits cost $0 and are marked
so the pipeline records them as cache hits. Optional in-flight *coalescing* collapses identical
concurrent calls onto a single upstream request (the first computes; the rest await its result).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
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
    """Wraps an ``LLMClient`` with a response cache for non-streaming calls.

    With ``coalesce=True`` a per-cache-key registry of in-flight futures collapses identical
    concurrent ``complete`` calls onto one upstream request: the first (the leader) computes and
    stores; the rest (followers) await the leader's result and are marked cached so cost is
    counted once. Enabled via ``cache.coalesce`` at wiring time; off by default in the constructor.
    """

    def __init__(
        self, inner: LLMClient, cache: CacheStore, clock: Clock, *, coalesce: bool = False
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._clock = clock
        self._coalesce = coalesce
        self._inflight: dict[str, asyncio.Future[Completion]] = {}

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
        if not self._coalesce:
            return await self._complete_and_store(spec, key, now)
        return await self._coalesced_complete(spec, key, now)

    async def _coalesced_complete(self, spec: CallSpec, key: str, now: float) -> Completion:
        existing = self._inflight.get(key)
        if existing is not None:
            # A concurrent identical call is already in flight — share its result. `shield` so
            # this caller's cancellation/timeout can't abort the computation others depend on;
            # marking it cached keeps cost accounting to the one real upstream call.
            return replace(await asyncio.shield(existing), cached=True)
        # Between the cache miss above and this registration there is no `await`, so at most one
        # leader is ever created per key.
        future: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await self._complete_and_store(spec, key, now)
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved so an un-awaited future doesn't warn on GC
            raise
        else:
            future.set_result(result)
            return result
        finally:
            del self._inflight[key]

    async def _complete_and_store(self, spec: CallSpec, key: str, now: float) -> Completion:
        result = await self._inner.complete(spec)
        # Do not cache tool-call, empty, or truncated results — a `length`-truncated answer would
        # otherwise be served for every future identical request.
        if not result.tool_calls and result.content.strip() and result.finish_reason != "length":
            await self._cache.put(key, spec.llm_name, _serialize(result), now=now)
        return result

    def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        return self._inner.stream(spec)
