"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from mom import __version__
from mom.cli import app


runner = CliRunner()


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help exits 0/2 depending on Typer version but always renders help
    assert "Mixture-of-Models" in result.stdout
