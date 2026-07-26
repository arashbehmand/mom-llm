"""Response cache store: SQLite with TTL and size-cap eviction.

Stores non-streaming member completions keyed by the bit-compatible cache key. Eviction is exact
and synchronous (delete oldest ``last_used_at`` until under the cap), done inside the write
transaction — no background scanner.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from mom.store.connection import open_database


MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE entries (
        key          TEXT    PRIMARY KEY,
        llm          TEXT    NOT NULL,
        body         TEXT    NOT NULL,
        size_bytes   INTEGER NOT NULL,
        created_at   REAL    NOT NULL,
        last_used_at REAL    NOT NULL,
        hits         INTEGER NOT NULL DEFAULT 0
    ) STRICT;
    CREATE INDEX ix_entries_last_used ON entries (last_used_at);
    """,
)


class SqliteCacheStore:
    """TTL + size-capped response cache."""

    def __init__(self, conn: aiosqlite.Connection, *, ttl_seconds: float, max_bytes: int) -> None:
        self._conn = conn
        self._ttl = ttl_seconds
        self._max_bytes = max_bytes

    @classmethod
    async def open(
        cls, path: str | Path, *, ttl_seconds: float, max_bytes: int
    ) -> SqliteCacheStore:
        conn = await open_database(path, MIGRATIONS)
        return cls(conn, ttl_seconds=ttl_seconds, max_bytes=max_bytes)

    async def close(self) -> None:
        await self._conn.close()

    async def get(self, key: str, *, now: float) -> str | None:
        cursor = await self._conn.execute(
            "SELECT body, created_at FROM entries WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        body, created_at = row["body"], row["created_at"]
        if self._ttl and now - created_at > self._ttl:
            await self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
            await self._conn.commit()
            return None
        await self._conn.execute(
            "UPDATE entries SET last_used_at = ?, hits = hits + 1 WHERE key = ?", (now, key)
        )
        await self._conn.commit()
        return str(body)

    async def put(self, key: str, llm: str, body: str, *, now: float) -> None:
        size = len(body.encode("utf-8"))
        await self._conn.execute(
            "INSERT OR REPLACE INTO entries "
            "(key, llm, body, size_bytes, created_at, last_used_at, hits) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (key, llm, body, size, now, now),
        )
        await self._evict_if_needed()
        await self._conn.commit()

    async def _evict_if_needed(self) -> None:
        cursor = await self._conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM entries")
        row = await cursor.fetchone()
        await cursor.close()
        total = int(row[0]) if row else 0
        if total <= self._max_bytes:
            return
        target = int(self._max_bytes * 0.9)
        # Keep the most-recently-used rows whose running size stays within `target`; delete the
        # rest (the least-recently-used). A window sum newest->oldest marks the overflow tail:
        # any row whose cumulative size (from the newest down to and including it) exceeds the
        # target is beyond the cap, i.e. one of the oldest rows.
        await self._conn.execute(
            """
            DELETE FROM entries WHERE key IN (
                SELECT key FROM (
                    SELECT key, SUM(size_bytes) OVER (
                        ORDER BY last_used_at DESC, key DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cumulative
                    FROM entries
                )
                WHERE cumulative > ?
            )
            """,
            (target,),
        )

    async def stats(self) -> dict[str, int]:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(size_bytes), 0) AS bytes, "
            "COALESCE(SUM(hits), 0) AS hits FROM entries"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row) if row is not None else {}

    async def clear(self) -> int:
        """Delete every cached entry; return how many rows were removed."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM entries")
        row = await cursor.fetchone()
        await cursor.close()
        removed = int(row[0]) if row else 0
        await self._conn.execute("DELETE FROM entries")
        await self._conn.commit()
        return removed


class NullCacheStore:
    """A cache that never hits (used when caching is disabled)."""

    async def get(self, key: str, *, now: float) -> str | None:  # noqa: ARG002
        return None

    async def put(self, key: str, llm: str, body: str, *, now: float) -> None:  # noqa: ARG002
        return None
