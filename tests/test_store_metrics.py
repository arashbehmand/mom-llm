"""Metrics store: migrations, PRAGMAs, aggregation, and the bounded recorder."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mom.store.connection import user_version
from mom.store.metrics import MIGRATIONS, CallMetric, MetricsRecorder, MetricsStore


def _metric(**overrides: object) -> CallMetric:
    base: dict[str, object] = {
        "request_id": "req-1",
        "ts": 1000.0,
        "ensemble": "bmom",
        "llm": "gpt",
        "model": "openai/gpt-x",
        "role": "fanout",
        "status": "ok",
    }
    base.update(overrides)
    return CallMetric(**base)  # type: ignore[arg-type]


@pytest.fixture
async def store(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "metrics.db")
    yield store
    await store.close()


async def test_open_applies_migrations_and_pragmas(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "m.db")
    try:
        assert await user_version(store._conn) == len(MIGRATIONS)
        cursor = await store._conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0].lower() == "wal"
        assert await store.count() == 0
    finally:
        await store.close()


async def test_insert_and_aggregate(store: MetricsStore):
    await store.insert_many(
        [
            _metric(
                prompt_tokens=100, completion_tokens=50, cost_usd=0.01, cached_prompt_tokens=80
            ),
            _metric(role="synthesis", prompt_tokens=200, completion_tokens=90, cost_usd=0.02),
            _metric(status="error", turn_type="relay", error="boom"),
        ]
    )
    assert await store.count() == 3
    agg = await store.aggregate()
    assert agg["calls"] == 3
    assert agg["prompt_tokens"] == 300
    assert agg["completion_tokens"] == 140
    assert agg["cached_prompt_tokens"] == 80
    assert abs(agg["cost_usd"] - 0.03) < 1e-9
    assert agg["errors"] == 1
    assert agg["relay_calls"] == 1


async def test_aggregate_filters_by_time_and_ensemble(store: MetricsStore):
    await store.insert_many(
        [
            _metric(ts=10.0, ensemble="a"),
            _metric(ts=20.0, ensemble="b"),
            _metric(ts=30.0, ensemble="a"),
        ]
    )
    assert (await store.aggregate(ensemble="a"))["calls"] == 2
    assert (await store.aggregate(start=15.0))["calls"] == 2
    assert (await store.aggregate(start=15.0, end=25.0))["calls"] == 1


async def test_recorder_flush_writes_queued(store: MetricsStore):
    recorder = MetricsRecorder(store)
    for _ in range(5):
        recorder.record(_metric())
    await recorder.flush()
    assert await store.count() == 5


async def test_recorder_worker_drains(store: MetricsStore):
    recorder = MetricsRecorder(store, batch=2)
    await recorder.start()
    for i in range(4):
        recorder.record(_metric(request_id=f"req-{i}"))
    # Give the worker a few loop turns to drain.
    for _ in range(50):
        if await store.count() == 4:
            break
        await asyncio.sleep(0.005)
    await recorder.stop()
    assert await store.count() == 4


async def test_recorder_overflow_is_counted(store: MetricsStore):
    recorder = MetricsRecorder(store, maxsize=2)
    for _ in range(5):  # worker not started -> queue fills at 2, rest dropped
        recorder.record(_metric())
    assert recorder.dropped == 3
