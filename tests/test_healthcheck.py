"""CLI healthcheck exits non-zero when the endpoint is unreachable."""

from __future__ import annotations

from typer.testing import CliRunner

from mom.cli import app


runner = CliRunner()


def test_healthcheck_unreachable_exits_1():
    # Port 9 (discard) with nothing listening -> connection refused -> exit 1.
    result = runner.invoke(
        app, ["healthcheck", "--url", "http://127.0.0.1:9/health", "--timeout", "0.2"]
    )
    assert result.exit_code == 1
