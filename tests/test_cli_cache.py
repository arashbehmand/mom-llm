"""``mom cache stats`` / ``mom cache purge`` over the resolved data_dir."""

from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

from mom.cli import app
from mom.store.cache import SqliteCacheStore


runner = CliRunner()


def _seed_cache(db: Path, entries: int = 2) -> None:
    async def go() -> None:
        store = await SqliteCacheStore.open(db, ttl_seconds=100.0, max_bytes=1 << 30)
        for i in range(entries):
            await store.put(f"k{i}", "a", f"body-{i}", now=1.0)
        await store.close()

    asyncio.run(go())


def _remaining(db: Path) -> int:
    async def go() -> int:
        store = await SqliteCacheStore.open(db, ttl_seconds=100.0, max_bytes=1 << 30)
        try:
            return (await store.stats()).get("entries", 0)
        finally:
            await store.close()

    return asyncio.run(go())


def test_cache_stats_reports_entries(tmp_path: Path):
    _seed_cache(tmp_path / "cache.db", entries=2)
    result = runner.invoke(app, ["cache", "stats", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "entries: 2" in result.stdout


def test_cache_stats_on_missing_db_reports_empty(tmp_path: Path):
    result = runner.invoke(app, ["cache", "stats", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "empty" in result.stdout


def test_cache_purge_clears_entries(tmp_path: Path):
    _seed_cache(tmp_path / "cache.db", entries=3)
    result = runner.invoke(app, ["cache", "purge", "--data-dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "purged 3" in result.stdout
    assert _remaining(tmp_path / "cache.db") == 0


def test_cache_purge_requires_confirmation(tmp_path: Path):
    _seed_cache(tmp_path / "cache.db", entries=2)
    result = runner.invoke(app, ["cache", "purge", "--data-dir", str(tmp_path)], input="n\n")
    assert result.exit_code != 0  # aborted at the prompt
    assert _remaining(tmp_path / "cache.db") == 2  # nothing removed


def test_data_dir_follows_a_discovered_config(tmp_path: Path):
    """`mom cache` and the gateway must mean the same database. They used to be able to disagree:
    this path read one config file and ignored MOM_CONFIG_OVERLAY, while the server merged it."""
    data = tmp_path / "elsewhere"
    (Path.cwd() / "mom.yaml").write_text(
        "version: 2\n"
        f"storage: {{ data_dir: {data} }}\n"
        "llms: { a: { model: openai/a } }\n"
        "ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["cache", "stats"])
    assert result.exit_code == 0, result.output
    assert str(data / "cache.db") in result.stdout


def test_data_dir_falls_back_silently_with_no_config(tmp_path: Path):
    """These commands only ever wanted a directory, and answered without a config before
    discovery existed — a miss must not become a hard failure."""
    result = runner.invoke(app, ["cache", "stats"])
    assert result.exit_code == 0, result.output
    assert "cache: empty" in result.stdout
