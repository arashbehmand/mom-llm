"""aiosqlite connection setup: WAL, sane PRAGMAs, and ``user_version`` migrations.

One long-lived connection per database (aiosqlite runs it on a dedicated worker thread, so
writes serialise naturally and never block the event loop). Schema evolves through an ordered
list of migration scripts applied via ``PRAGMA user_version`` — no ORM, no Alembic.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite


_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA temp_store = MEMORY",
)


async def _apply_migrations(conn: aiosqlite.Connection, migrations: tuple[str, ...]) -> None:
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    version = int(row[0]) if row else 0
    await cursor.close()
    while version < len(migrations):
        await conn.executescript(migrations[version])
        version += 1
        await conn.execute(f"PRAGMA user_version = {version}")  # int, not user input
    await conn.commit()


async def open_database(path: str | Path, migrations: tuple[str, ...]) -> aiosqlite.Connection:
    """Open (creating parent dirs), configure PRAGMAs, and migrate a SQLite database."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    for pragma in _PRAGMAS:
        await conn.execute(pragma)
    await _apply_migrations(conn, migrations)
    return conn


async def user_version(conn: aiosqlite.Connection) -> int:
    """Return the database's current schema version."""
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    await cursor.close()
    return int(row[0]) if row else 0
