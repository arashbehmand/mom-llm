"""Metrics store: one STRICT ``llm_calls`` table, batched writes off the request path.

Writes go through :class:`MetricsRecorder` — a bounded queue drained by a single worker — so a
slow disk can never backpressure token streaming (overflow drops a row and bumps a counter). The
schema records provider prompt-cache tokens (``cached_prompt_tokens`` / ``cache_write_tokens``)
and the ``turn_type`` ('ensemble' vs relay continuation) so the relay cost win is queryable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import contextlib
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from mom.store.connection import open_database


Role = Literal["fanout", "synthesis"]
Status = Literal["ok", "error"]
TurnType = Literal["ensemble", "relay"]

MIGRATIONS: tuple[str, ...] = (
    # v1: initial schema
    """
    CREATE TABLE llm_calls (
        id                   INTEGER PRIMARY KEY,
        request_id           TEXT    NOT NULL,
        ts                   REAL    NOT NULL,
        ensemble             TEXT    NOT NULL,
        llm                  TEXT    NOT NULL,
        model                TEXT,
        role                 TEXT    NOT NULL CHECK (role IN ('fanout', 'synthesis')),
        status               TEXT    NOT NULL CHECK (status IN ('ok', 'error')),
        cache_hit            INTEGER NOT NULL DEFAULT 0,
        turn_type            TEXT    NOT NULL DEFAULT 'ensemble',
        prompt_tokens        INTEGER,
        completion_tokens    INTEGER,
        reasoning_tokens     INTEGER,
        cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens   INTEGER NOT NULL DEFAULT 0,
        total_tokens         INTEGER,
        cost_usd             REAL    CHECK (cost_usd IS NULL OR cost_usd >= 0),
        duration_ms          REAL,
        error                TEXT
    ) STRICT;
    CREATE INDEX ix_llm_calls_ts ON llm_calls (ts);
    CREATE INDEX ix_llm_calls_request ON llm_calls (request_id);
    CREATE INDEX ix_llm_calls_ensemble_ts ON llm_calls (ensemble, ts);
    """,
)

# Static SQL — column identifiers are fixed constants; all values are bound parameters.
_INSERT_SQL = (
    "INSERT INTO llm_calls ("
    "request_id, ts, ensemble, llm, model, role, status, cache_hit, turn_type, "
    "prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens, "
    "cache_write_tokens, total_tokens, cost_usd, duration_ms, error"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_AGGREGATE_SELECT = (
    "SELECT "
    "COUNT(*) AS calls, "
    "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
    "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
    "COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens, "
    "COALESCE(SUM(cached_prompt_tokens), 0) AS cached_prompt_tokens, "
    "COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens, "
    "COALESCE(SUM(cost_usd), 0.0) AS cost_usd, "
    "SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors, "
    "SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits, "
    "SUM(CASE WHEN turn_type = 'relay' THEN 1 ELSE 0 END) AS relay_calls "
    "FROM llm_calls"
)


@dataclass(frozen=True, slots=True)
class CallMetric:
    request_id: str
    ts: float
    ensemble: str
    llm: str
    model: str | None
    role: Role
    status: Status
    cache_hit: bool = False
    turn_type: TurnType = "ensemble"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: float | None = None
    error: str | None = None


class MetricsStore:
    """Thin async wrapper over the ``llm_calls`` table."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: str | Path) -> MetricsStore:
        return cls(await open_database(path, MIGRATIONS))

    async def close(self) -> None:
        await self._conn.close()

    async def insert_many(self, metrics: Sequence[CallMetric]) -> None:
        if not metrics:
            return
        # astuple follows field declaration order, which matches the INSERT column order.
        await self._conn.executemany(_INSERT_SQL, [astuple(m) for m in metrics])
        await self._conn.commit()

    async def count(self) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) FROM llm_calls")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0

    async def aggregate(
        self, *, start: float | None = None, end: float | None = None, ensemble: str | None = None
    ) -> dict[str, Any]:
        """Single-pass usage/cost aggregation over an optional time/ensemble window."""
        where: list[str] = []
        params: list[Any] = []
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts < ?")
            params.append(end)
        if ensemble is not None:
            where.append("ensemble = ?")
            params.append(ensemble)
        sql = _AGGREGATE_SELECT
        if where:
            sql = f"{_AGGREGATE_SELECT} WHERE {' AND '.join(where)}"
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row) if row is not None else {}


class MetricsRecorder:
    """Buffers metrics on a bounded queue and drains them to the store off the hot path."""

    def __init__(self, store: MetricsStore, *, maxsize: int = 1000, batch: int = 100) -> None:
        self._store = store
        self._queue: asyncio.Queue[CallMetric] = asyncio.Queue(maxsize=maxsize)
        self._batch = batch
        self._dropped = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def dropped(self) -> int:
        return self._dropped

    def record(self, metric: CallMetric) -> None:
        """Enqueue a metric (sync, never awaits). Drops + counts on overflow."""
        try:
            self._queue.put_nowait(metric)
        except asyncio.QueueFull:
            self._dropped += 1

    async def _drain_batch(self) -> None:
        first = await self._queue.get()
        batch = [first]
        while len(batch) < self._batch:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._store.insert_many(batch)

    async def _run(self) -> None:
        while True:
            await self._drain_batch()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def flush(self) -> None:
        """Write any queued metrics immediately (used on shutdown)."""
        pending: list[CallMetric] = []
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                pending.append(self._queue.get_nowait())
        await self._store.insert_many(pending)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.flush()
