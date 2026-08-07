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
from dataclasses import astuple
from pathlib import Path
from typing import Any

import aiosqlite

from mom.domain.metrics import CallMetric
from mom.runtime.logging import get_logger
from mom.store.connection import open_database


logger = get_logger("mom.metrics")


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
    # v2: widen `status` to the full OutcomeStatus vocabulary ('empty'/'timeout'/'aborted' used to
    # collapse into 'error', hiding a real failure mode — see domain/results.py OutcomeStatus) and
    # add finish_reason/error_kind/error_detail/attempts. SQLite can't ALTER a CHECK constraint, so
    # this is the documented create-copy-drop-rename, wrapped in an explicit transaction:
    # `executescript` runs the whole block as given (it does not open one on its own), and without
    # BEGIN/COMMIT a failure between DROP and RENAME would destroy the table rather than roll back.
    # Historical rows get finish_reason/error_kind/error_detail = NULL and attempts = 1 (the field
    # didn't exist yet; 1 is the neutral "no retries recorded" default) — deliberately NOT
    # backfilled from a guess.
    """
    BEGIN;
    CREATE TABLE llm_calls_v2 (
        id                   INTEGER PRIMARY KEY,
        request_id           TEXT    NOT NULL,
        ts                   REAL    NOT NULL,
        ensemble             TEXT    NOT NULL,
        llm                  TEXT    NOT NULL,
        model                TEXT,
        role                 TEXT    NOT NULL CHECK (role IN ('fanout', 'synthesis')),
        status               TEXT    NOT NULL
                                     CHECK (status IN
                                            ('ok', 'empty', 'error', 'timeout', 'skipped',
                                             'aborted')),
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
        error                TEXT,
        finish_reason        TEXT,
        error_kind           TEXT,
        error_detail         TEXT,
        attempts             INTEGER NOT NULL DEFAULT 1
    ) STRICT;
    INSERT INTO llm_calls_v2 (
        id, request_id, ts, ensemble, llm, model, role, status, cache_hit, turn_type,
        prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens,
        cache_write_tokens, total_tokens, cost_usd, duration_ms, error,
        finish_reason, error_kind, error_detail, attempts
    )
    SELECT
        id, request_id, ts, ensemble, llm, model, role, status, cache_hit, turn_type,
        prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens,
        cache_write_tokens, total_tokens, cost_usd, duration_ms, error,
        NULL, NULL, NULL, 1
    FROM llm_calls;
    DROP TABLE llm_calls;
    ALTER TABLE llm_calls_v2 RENAME TO llm_calls;
    CREATE INDEX ix_llm_calls_ts ON llm_calls (ts);
    CREATE INDEX ix_llm_calls_request ON llm_calls (request_id);
    CREATE INDEX ix_llm_calls_ensemble_ts ON llm_calls (ensemble, ts);
    COMMIT;
    """,
)

# Static SQL — column identifiers are fixed constants; all values are bound parameters.
_INSERT_SQL = (
    "INSERT INTO llm_calls ("
    "request_id, ts, ensemble, llm, model, role, status, cache_hit, turn_type, "
    "prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens, "
    "cache_write_tokens, total_tokens, cost_usd, duration_ms, error, "
    "finish_reason, error_kind, error_detail, attempts"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# The aggregate column list, reused by the single-window aggregate and the grouped variant.
_AGGREGATE_COLUMNS = (
    "COUNT(*) AS calls, "
    "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
    "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
    "COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens, "
    "COALESCE(SUM(cached_prompt_tokens), 0) AS cached_prompt_tokens, "
    "COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens, "
    "COALESCE(SUM(cost_usd), 0.0) AS cost_usd, "
    # `<> 'ok'` (not `= 'error'`): status now also carries 'empty'/'timeout'/'aborted', all of
    # which are failures too — the old `= 'error'` check would silently under-count them.
    "SUM(CASE WHEN status <> 'ok' THEN 1 ELSE 0 END) AS errors, "
    "SUM(CASE WHEN status = 'empty' THEN 1 ELSE 0 END) AS empty, "
    "SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) AS timeouts, "
    "SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits, "
    # A cache hit costs $0 by construction; everything else was a real (attempted) upstream call.
    "SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) AS billable_calls, "
    "SUM(CASE WHEN turn_type = 'relay' THEN 1 ELSE 0 END) AS relay_calls"
)
_AGGREGATE_SELECT = f"SELECT {_AGGREGATE_COLUMNS} FROM llm_calls"  # noqa: S608 (constant columns)

# Grouping dimension -> the SQL group expression. Values are fixed identifiers/expressions (never
# interpolated from request input), so the group clause is not a SQL-injection surface. ``member``
# groups by the member identity (the ``llm`` column); ``day`` buckets the epoch ``ts`` by UTC date.
GROUP_DIMENSIONS: dict[str, str] = {
    "member": "llm",
    "turn_type": "turn_type",
    "day": "date(ts, 'unixepoch')",
    "ensemble": "ensemble",
    "status": "status",
}


def _window(
    start: float | None, end: float | None, ensemble: str | None
) -> tuple[list[str], list[Any]]:
    """Build the shared ``WHERE`` fragments + bound params for a time/ensemble window."""
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
    return where, params


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
        where, params = _window(start, end, ensemble)
        sql = _AGGREGATE_SELECT
        if where:
            sql = f"{_AGGREGATE_SELECT} WHERE {' AND '.join(where)}"
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row) if row is not None else {}

    async def aggregate_by(
        self,
        dimension: str,
        *,
        start: float | None = None,
        end: float | None = None,
        ensemble: str | None = None,
    ) -> list[dict[str, Any]]:
        """Grouped usage/cost aggregation (``SQL GROUP BY``) over an optional window.

        ``dimension`` is one of ``member`` / ``turn_type`` / ``day``. Each returned row carries the
        group key (under the dimension's name) plus the same aggregate columns as :meth:`aggregate`,
        ordered by the group key.
        """
        expr = GROUP_DIMENSIONS.get(dimension)
        if expr is None:
            raise ValueError(
                f"unknown grouping dimension {dimension!r} (expected one of "
                f"{sorted(GROUP_DIMENSIONS)})"
            )
        where, params = _window(start, end, ensemble)
        # expr/dimension are fixed constants from GROUP_DIMENSIONS; only window values are bound.
        sql = f"SELECT {expr} AS {dimension}, {_AGGREGATE_COLUMNS} FROM llm_calls"  # noqa: S608
        if where:
            sql += f" WHERE {' AND '.join(where)}"
        sql += f" GROUP BY {expr} ORDER BY {expr}"
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def estimated_cache_savings(
        self, *, start: float | None = None, end: float | None = None, ensemble: str | None = None
    ) -> float:
        """Estimated $ saved by cache hits over an optional time/ensemble window.

        "Estimated": a cache hit's row carries ``cost_usd = 0`` (it really was free), so there is
        no per-hit real cost to sum — this instead assumes each hit would have cost the SAME as
        the mean *non-cached* call for the same ``(llm, model)`` in the window. A reasonable
        proxy (token counts vary call to call, so it is not exact), and the only honest way to
        answer "roughly how much is the cache saving us" from data mom actually has.
        """
        where, params = _window(start, end, ensemble)
        extra = f" AND {' AND '.join(where)}" if where else ""
        # constant columns; only window values (in `extra`, bound via `params`) vary -> not a
        # SQL-injection surface, same reasoning as _AGGREGATE_SELECT above.
        avg_sql = (
            f"SELECT llm, model, AVG(cost_usd) AS avg_cost FROM llm_calls "  # noqa: S608
            f"WHERE cache_hit = 0 AND cost_usd IS NOT NULL{extra} GROUP BY llm, model"
        )
        hits_sql = (
            f"SELECT llm, model, COUNT(*) AS hits FROM llm_calls "  # noqa: S608
            f"WHERE cache_hit = 1{extra} GROUP BY llm, model"
        )
        avg_cursor = await self._conn.execute(avg_sql, params)
        avg_rows = await avg_cursor.fetchall()
        await avg_cursor.close()
        avg_cost = {(row["llm"], row["model"]): row["avg_cost"] for row in avg_rows}

        hits_cursor = await self._conn.execute(hits_sql, params)
        hits_rows = await hits_cursor.fetchall()
        await hits_cursor.close()

        return float(
            sum(avg_cost.get((row["llm"], row["model"]), 0.0) * row["hits"] for row in hits_rows)
        )


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
            if self._dropped % 100 == 1:
                logger.warning("metrics queue full; dropping metrics", dropped=self._dropped)

    async def _pull_batch(self) -> list[CallMetric]:
        first = await self._queue.get()
        batch = [first]
        while len(batch) < self._batch:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch: list[CallMetric] = []
            try:
                batch = await self._pull_batch()
                await self._store.insert_many(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed insert must not kill the writer for the process lifetime: drop the
                # batch, log, and back off briefly so a persistent DB error can't tight-spin. This
                # used to drop the batch WITHOUT counting it — under-reporting real loss on top of
                # the queue-overflow drops in `record()`, at exactly the moment (a DB write
                # actually failing) an operator most needs the real number.
                if batch:
                    self._dropped += len(batch)
                logger.warning(
                    "metrics batch write failed; batch dropped",
                    exc_info=True,
                    dropped_now=len(batch),
                    dropped_total=self._dropped,
                )
                await asyncio.sleep(0.5)

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
