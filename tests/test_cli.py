"""CLI smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

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


def test_serve_targets_the_bootstrapping_factory(monkeypatch):
    """Not `create_app`: `serve_app` resolves config and secrets in the process that serves —
    which under --reload is the child, not this one."""
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda target, **kw: calls.append((target, kw)), raising=False
    )
    result = runner.invoke(app, ["serve", "--port", "1234"])
    assert result.exit_code == 0, result.output
    target, kwargs = calls[0]
    assert target == "mom.api.app:serve_app"
    assert kwargs["factory"] is True
    assert kwargs["port"] == 1234


def test_serve_exports_its_flags_for_the_uvicorn_child(tmp_path: Path, monkeypatch):
    """uvicorn imports the factory and calls it with no arguments, in a child process under
    --reload, so flags travel by environment or not at all. Absolute, because the child's working
    directory is not guaranteed to be this one."""
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None, raising=False)
    config = tmp_path / "c.yaml"
    config.write_text("version: 2\n", encoding="utf-8")
    overlay = tmp_path / "o.yaml"
    overlay.write_text("{}\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["serve", "--config", str(config), "--overlay", str(overlay), "--auth-from-opencode"],
    )
    assert result.exit_code == 0, result.output
    assert os.environ["MOM_CONFIG"] == str(config.resolve())
    assert os.environ["MOM_CONFIG_OVERLAY"] == str(overlay.resolve())
    assert os.environ["MOM_AUTH_FROM_OPENCODE"] == "1"


def test_serve_without_flags_exports_nothing(monkeypatch):
    """An omitted flag must not clobber an env var the operator set deliberately."""
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None, raising=False)
    monkeypatch.setenv("MOM_CONFIG", "/from/the/environment.yaml")

    assert runner.invoke(app, ["serve"]).exit_code == 0
    assert os.environ["MOM_CONFIG"] == "/from/the/environment.yaml"
