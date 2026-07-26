"""Response cache: store round-trip, TTL, eviction, and the caching client."""

from __future__ import annotations

from pathlib import Path

from mom.adapters.caching import CachingClient
from mom.domain.ports import CallSpec
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
