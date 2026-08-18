"""``mom metrics usage`` over the resolved data_dir."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from typer.testing import CliRunner

from mom.cli import app
from mom.store.metrics import CallMetric, MetricsStore


runner = CliRunner()


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


def _seed(db: Path, metrics: list[CallMetric]) -> None:
    async def go() -> None:
        store = await MetricsStore.open(db)
        try:
            await store.insert_many(metrics)
        finally:
            await store.close()

    asyncio.run(go())


def test_metrics_usage_on_missing_db_reports_empty(tmp_path: Path):
    result = runner.invoke(app, ["metrics", "usage", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "empty" in result.stdout


def test_metrics_usage_reports_calls_cost_and_status_breakdown(tmp_path: Path):
    _seed(
        tmp_path / "metrics.db",
        [
            _metric(status="ok", cost_usd=0.02),
            _metric(status="ok", cost_usd=0.03, cache_hit=True),
            _metric(status="error", cost_usd=0.0, error="boom"),
            _metric(status="empty", cost_usd=0.01),
        ],
    )
    result = runner.invoke(app, ["metrics", "usage", "--data-dir", str(tmp_path), "--days", "0"])
    assert result.exit_code == 0, result.output
    assert "calls:        4" in result.stdout
    assert "cache hits:   1" in result.stdout
    assert "errors:     2" in result.stdout  # status <> 'ok': the 'error' row AND the 'empty' row
    assert "empty:      1" in result.stdout


def test_metrics_usage_days_window_excludes_old_rows(tmp_path: Path):
    now = time.time()
    _seed(
        tmp_path / "metrics.db",
        [_metric(ts=now - 3600), _metric(ts=now - 100 * 86400)],  # one recent, one 100 days old
    )
    db_arg = ["--data-dir", str(tmp_path)]
    result_all = runner.invoke(app, ["metrics", "usage", *db_arg, "--days", "0"])
    assert "calls:        2" in result_all.stdout  # all-time: both rows

    result_7d = runner.invoke(app, ["metrics", "usage", *db_arg, "--days", "7"])
    assert "calls:        1" in result_7d.stdout  # 7-day window: only the recent row


def test_metrics_usage_grouped_by_member(tmp_path: Path):
    _seed(
        tmp_path / "metrics.db",
        [_metric(llm="a"), _metric(llm="a"), _metric(llm="b")],
    )
    result = runner.invoke(
        app, ["metrics", "usage", "--data-dir", str(tmp_path), "--days", "0", "--by", "member"]
    )
    assert result.exit_code == 0, result.output
    assert "by member:" in result.stdout
    assert "a" in result.stdout
    assert "b" in result.stdout


def test_metrics_usage_restricts_to_one_ensemble(tmp_path: Path):
    _seed(
        tmp_path / "metrics.db",
        [_metric(ensemble="x"), _metric(ensemble="x"), _metric(ensemble="y")],
    )
    result = runner.invoke(
        app, ["metrics", "usage", "--data-dir", str(tmp_path), "--days", "0", "--ensemble", "x"]
    )
    assert result.exit_code == 0, result.output
    assert "calls:        2" in result.stdout


def test_metrics_usage_rejects_unknown_grouping_dimension(tmp_path: Path):
    _seed(tmp_path / "metrics.db", [_metric()])
    result = runner.invoke(app, ["metrics", "usage", "--data-dir", str(tmp_path), "--by", "bogus"])
    assert result.exit_code == 1
    assert "unknown grouping dimension" in result.output
