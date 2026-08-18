"""Response cache: store round-trip, TTL, eviction, the caching client, and in-flight coalescing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from mom.adapters.caching import CachingClient
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.results import Usage
from mom.store.cache import SqliteCacheStore
from mom.testing import FakeLLM, ManualClock


async def test_store_roundtrip(tmp_path: Path):
    store = await SqliteCacheStore.open(tmp_path / "c.db", ttl_seconds=100, max_bytes=10**6)
    await store.put("k", "a", "body", now=1000.0)
    assert await store.get("k", now=1001.0) == "body"
    assert await store.get("missing", now=1001.0) is None
    await store.close()


async def test_ttl_expiry(tmp_path: Path):
    store = await SqliteCacheStore.open(tmp_path / "c.db", ttl_seconds=10, max_bytes=10**6)
    await store.put("k", "a", "body", now=1000.0)
    assert await store.get("k", now=1005.0) == "body"
    assert await store.get("k", now=1020.0) is None  # expired -> deleted
    await store.close()


async def test_size_eviction_keeps_most_recently_used(tmp_path: Path):
    store = await SqliteCacheStore.open(tmp_path / "c.db", ttl_seconds=0, max_bytes=300)
    for i in range(10):
        await store.put(f"k{i}", "a", "x" * 100, now=1000.0 + i)
    stats = await store.stats()
    assert stats["bytes"] <= 300  # evicted down under the cap
    # LRU: the most-recently-used entries survive and the oldest are gone — NOT the reverse
    # (the v1-era SQL evicted the hottest rows and kept the cold ones).
    assert await store.get("k9", now=2000.0) == "x" * 100
    assert await store.get("k8", now=2000.0) == "x" * 100
    assert await store.get("k0", now=2000.0) is None
    assert await store.get("k1", now=2000.0) is None
    await store.close()


async def test_caching_client_dedups(tmp_path: Path):
    store = await SqliteCacheStore.open(tmp_path / "c.db", ttl_seconds=100, max_bytes=10**6)
    inner = FakeLLM(replies={"a": "hello"})
    client = CachingClient(inner, store, ManualClock())
    spec = CallSpec(llm_name="a", model="openai/a", messages=[{"role": "user", "content": "hi"}])

    first = await client.complete(spec)
    assert first.content == "hello"
    assert first.cached is False

    second = await client.complete(spec)
    assert second.cached is True
    assert second.content == "hello"
    assert len(inner.completions) == 1  # the underlying model was called only once
    await store.close()


class _GatedClient:
    """An ``LLMClient`` whose ``complete`` blocks on a gate so calls can be held overlapping.

    ``wait_for_calls(n)`` resolves once ``n`` calls have entered ``complete`` — a deterministic
    way to hold N requests in flight before releasing the gate, with no polling.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.gate = asyncio.Event()
        self._targets: list[tuple[int, asyncio.Future[None]]] = []

    def _mark_entry(self) -> None:
        self.calls += 1
        for target, fut in self._targets:
            if self.calls >= target and not fut.done():
                fut.set_result(None)

    async def complete(self, spec: CallSpec) -> Completion:
        self._mark_entry()
        await self.gate.wait()
        return Completion(
            content="hello", reasoning=None, finish_reason="stop", usage=Usage(prompt_tokens=10)
        )

    async def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk()  # unused; present to satisfy the LLMClient protocol

    async def wait_for_calls(self, n: int) -> None:
        if self.calls >= n:
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._targets.append((n, fut))
        async with asyncio.timeout(1.0):  # fail the test rather than hang on a regression
            await fut


class _MemCache:
    """An in-memory ``CacheStore`` whose ``get`` resolves inline (no loop turn).

    That makes coalescing deterministic in tests: a follower reaches the shared in-flight future
    before yielding, so it always attaches instead of racing the leader's cleanup.
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str, *, now: float) -> str | None:
        return self.data.get(key)

    async def put(self, key: str, llm: str, body: str, *, now: float) -> None:
        self.data[key] = body


def _spec() -> CallSpec:
    return CallSpec(llm_name="a", model="openai/a", messages=[{"role": "user", "content": "hi"}])


async def test_coalesce_collapses_concurrent_identical_calls():
    inner = _GatedClient()
    client = CachingClient(inner, _MemCache(), ManualClock(), coalesce=True)

    leader = asyncio.create_task(client.complete(_spec()))
    await inner.wait_for_calls(1)  # leader is in flight with its future registered
    follower = asyncio.create_task(client.complete(_spec()))
    inner.gate.set()
    first, second = await asyncio.gather(leader, follower)

    assert inner.calls == 1  # the follower shared the leader's single upstream call
    assert first.content == second.content == "hello"
    assert {first.cached, second.cached} == {False, True}  # follower marked cached (cost once)


async def test_no_coalesce_lets_concurrent_calls_hit_upstream_twice():
    inner = _GatedClient()
    client = CachingClient(inner, _MemCache(), ManualClock(), coalesce=False)

    tasks = [asyncio.create_task(client.complete(_spec())) for _ in range(2)]
    await inner.wait_for_calls(2)  # without coalescing both reach upstream
    inner.gate.set()
    await asyncio.gather(*tasks)

    assert inner.calls == 2


async def test_coalesced_followers_share_the_leaders_failure():
    class _FailingGated(_GatedClient):
        async def complete(self, spec: CallSpec) -> Completion:
            self._mark_entry()
            await self.gate.wait()
            raise RuntimeError("upstream boom")

    inner = _FailingGated()
    client = CachingClient(inner, _MemCache(), ManualClock(), coalesce=True)
    leader = asyncio.create_task(client.complete(_spec()))
    await inner.wait_for_calls(1)
    follower = asyncio.create_task(client.complete(_spec()))
    inner.gate.set()
    results = await asyncio.gather(leader, follower, return_exceptions=True)

    assert inner.calls == 1  # one shared upstream attempt (failed calls aren't cached)
    assert all(isinstance(r, RuntimeError) for r in results)  # both see the same failure
