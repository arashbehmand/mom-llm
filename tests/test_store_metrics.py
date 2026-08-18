"""Metrics store: migrations, PRAGMAs, aggregation, and the bounded recorder."""

from __future__ import annotations

import asyncio
from dataclasses import astuple, fields
from pathlib import Path

import pytest

from mom.store.connection import open_database, user_version
from mom.store.metrics import _INSERT_SQL, MIGRATIONS, CallMetric, MetricsRecorder, MetricsStore


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


class _FailOnceStore:
    """Wraps a real store; its first ``insert_many`` raises, the rest pass through."""

    def __init__(self, inner: MetricsStore) -> None:
        self._inner = inner
        self.calls = 0

    async def insert_many(self, metrics: list) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("disk full")
        await self._inner.insert_many(metrics)


async def test_recorder_counts_a_failed_batch_write_as_dropped(store: MetricsStore):
    """A batch that fails to insert used to vanish from `dropped` entirely — the operator-facing
    counter under-reported real loss at exactly the moment (a DB write actually failing) it
    mattered most."""
    failing = _FailOnceStore(store)
    recorder = MetricsRecorder(failing, batch=10)  # type: ignore[arg-type]
    await recorder.start()
    for i in range(3):
        recorder.record(_metric(request_id=f"req-{i}"))
    for _ in range(200):
        if recorder.dropped == 3:
            break
        await asyncio.sleep(0.005)
    await recorder.stop()
    assert recorder.dropped == 3
    assert await store.count() == 0  # the failed batch was never written, consistent with "dropped"


# ------------------------------------------------------------------------------------------------
# Schema v2: the column-arity guard, and migrating over pre-existing v1 data.
# ------------------------------------------------------------------------------------------------


def test_call_metric_arity_matches_insert_sql():
    """astuple(m) fills the ``?`` placeholders positionally, in dataclass-field order — a field
    inserted out of order between two same-typed columns would silently swap data rather than
    raise. This guards the one invariant that can't be caught any other way."""
    assert len(fields(CallMetric)) == _INSERT_SQL.count("?")
    assert len(astuple(_metric())) == _INSERT_SQL.count("?")


async def test_new_v2_fields_round_trip(store: MetricsStore):
    metric = _metric(
        status="empty",
        finish_reason="length",
        error_kind="context_length",
        error_detail="truncated at 4096 tokens",
        attempts=2,
    )
    await store.insert_many([metric])
    cursor = await store._conn.execute(
        "SELECT status, finish_reason, error_kind, error_detail, attempts FROM llm_calls"
    )
    row = await cursor.fetchone()
    await cursor.close()
    assert tuple(row) == ("empty", "length", "context_length", "truncated at 4096 tokens", 2)


async def test_migration_over_existing_v1_data_preserves_rows(tmp_path: Path):
    """The migration that matters most: production has real v1 rows on disk. Build a genuine v1
    schema (only MIGRATIONS[:1] applied), insert with the v1 INSERT shape, then reopen with the
    full migration list and confirm every row survives with the new columns defaulted sanely."""
    path = tmp_path / "v1.db"
    v1_conn = await open_database(path, MIGRATIONS[:1])
    assert await user_version(v1_conn) == 1
    await v1_conn.execute(
        "INSERT INTO llm_calls ("
        "request_id, ts, ensemble, llm, model, role, status, cache_hit, turn_type, "
        "prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens, "
        "cache_write_tokens, total_tokens, cost_usd, duration_ms, error"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "req-old",
            500.0,
            "bmom",
            "gpt",
            "openai/gpt-x",
            "fanout",
            "error",
            0,
            "ensemble",
            100,
            0,
            0,
            0,
            0,
            100,
            0.01,
            250.0,
            "call failed",
        ),
    )
    await v1_conn.commit()
    await v1_conn.close()

    store = await MetricsStore.open(path)  # applies MIGRATIONS[1:] on top of the existing v1 data
    try:
        assert await user_version(store._conn) == len(MIGRATIONS)
        assert await store.count() == 1
        cursor = await store._conn.execute(
            "SELECT request_id, status, error, finish_reason, error_kind, error_detail, attempts "
            "FROM llm_calls"
        )
        row = await cursor.fetchone()
        await cursor.close()
        # Pre-migration data is untouched; the new columns default to NULL/1, never backfilled
        # from a guess.
        assert tuple(row) == ("req-old", "error", "call failed", None, None, None, 1)
        # The widened CHECK constraint accepts a status value that didn't exist under v1.
        await store.insert_many([_metric(status="aborted")])
        assert await store.count() == 2
    finally:
        await store.close()


async def test_aggregate_distinguishes_empty_timeout_from_error(store: MetricsStore):
    await store.insert_many(
        [
            _metric(status="ok"),
            _metric(status="empty"),
            _metric(status="timeout"),
            _metric(status="error"),
            _metric(status="aborted"),
        ]
    )
    agg = await store.aggregate()
    assert agg["calls"] == 5
    assert agg["errors"] == 4  # everything that isn't 'ok'
    assert agg["empty"] == 1
    assert agg["timeouts"] == 1


async def test_aggregate_billable_calls_excludes_cache_hits(store: MetricsStore):
    await store.insert_many(
        [
            _metric(cache_hit=True),
            _metric(cache_hit=True),
            _metric(cache_hit=False),
        ]
    )
    agg = await store.aggregate()
    assert agg["cache_hits"] == 2
    assert agg["billable_calls"] == 1


async def test_group_by_status(store: MetricsStore):
    await store.insert_many([_metric(status="ok"), _metric(status="ok"), _metric(status="empty")])
    rows = await store.aggregate_by("status")
    assert {row["status"]: row["calls"] for row in rows} == {"ok": 2, "empty": 1}


async def test_group_by_ensemble(store: MetricsStore):
    await store.insert_many([_metric(ensemble="a"), _metric(ensemble="a"), _metric(ensemble="b")])
    rows = await store.aggregate_by("ensemble")
    assert {row["ensemble"]: row["calls"] for row in rows} == {"a": 2, "b": 1}


# ------------------------------------------------------------------------------------------------
# estimated_cache_savings: mean non-cached cost per (llm, model) x that pair's cache-hit count.
# ------------------------------------------------------------------------------------------------


async def test_estimated_cache_savings_multiplies_mean_cost_by_hit_count(store: MetricsStore):
    await store.insert_many(
        [
            _metric(llm="a", model="m/a", cache_hit=False, cost_usd=0.02),
            _metric(llm="a", model="m/a", cache_hit=False, cost_usd=0.04),  # mean = 0.03
            _metric(llm="a", model="m/a", cache_hit=True, cost_usd=0.0),
            _metric(llm="a", model="m/a", cache_hit=True, cost_usd=0.0),
            _metric(llm="a", model="m/a", cache_hit=True, cost_usd=0.0),  # 3 hits x 0.03
        ]
    )
    savings = await store.estimated_cache_savings()
    assert abs(savings - 0.09) < 1e-9


async def test_estimated_cache_savings_is_zero_with_no_cache_hits(store: MetricsStore):
    await store.insert_many([_metric(cache_hit=False, cost_usd=0.05)])
    assert await store.estimated_cache_savings() == 0.0


async def test_estimated_cache_savings_zero_for_a_pair_never_seen_uncached(store: MetricsStore):
    # A cache hit for (llm, model) that has NO non-cached row in the window has no cost baseline
    # to estimate from — contributes 0, not a crash or a made-up number.
    await store.insert_many([_metric(llm="only-cached", cache_hit=True, cost_usd=0.0)])
    assert await store.estimated_cache_savings() == 0.0


async def test_estimated_cache_savings_respects_the_ensemble_window(store: MetricsStore):
    await store.insert_many(
        [
            _metric(ensemble="x", llm="a", cache_hit=False, cost_usd=0.10),
            _metric(ensemble="x", llm="a", cache_hit=True, cost_usd=0.0),
            _metric(ensemble="y", llm="a", cache_hit=False, cost_usd=1.0),
            _metric(ensemble="y", llm="a", cache_hit=True, cost_usd=0.0),
        ]
    )
    assert abs(await store.estimated_cache_savings(ensemble="x") - 0.10) < 1e-9
