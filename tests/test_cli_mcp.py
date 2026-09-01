"""``mom mcp`` — the stdio transport's container build, logging sink, and cleanup."""

from __future__ import annotations

from pathlib import Path
import sys
from textwrap import dedent

import pytest
from typer.testing import CliRunner

from mom.cli import app
from mom.config.resolve import ConfigError


runner = CliRunner()

CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
    ensembles:
      e:
        members: [{ llm: a }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return path


@pytest.fixture(autouse=True)
def log_sinks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which standard stream `mom mcp` sends logs to, without reconfiguring structlog.

    Named rather than captured by identity because CliRunner swaps `sys.stdout`/`sys.stderr` for
    the duration of the invocation — the objects are gone by the time a test asserts. Recording
    instead of really configuring also matters: structlog caches a bound logger per module, so a
    real reconfigure here would leave every later test logging into a closed capture buffer.
    """
    sinks: list[str] = []

    def _record(*, level: str, fmt: str, stream=None) -> None:
        sinks.append(
            "stderr" if stream is sys.stderr else "stdout" if stream in (None, sys.stdout) else "?"
        )

    monkeypatch.setattr("mom.runtime.logging.configure_logging", _record)
    return sinks


def test_mcp_command_is_registered():
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "stdio" in result.stdout


def test_serves_the_tools_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    served: dict[str, object] = {}

    async def _fake_serve(self) -> None:
        served["tools"] = [tool.name for tool in (await self.list_tools())]

    monkeypatch.setattr("mcp.server.mcpserver.MCPServer.run_stdio_async", _fake_serve)
    result = runner.invoke(
        app, ["mcp", "--config", str(_config_file(tmp_path)), "--data-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert served["tools"] == [
        "list_llms",
        "list_ensembles",
        "consult",
        "runs",
        "usage",
        "cache_stats",
    ]
    # The container it built is the real one: it opened (and closed) the same databases the
    # gateway uses, so a consult here is recorded where `mom metrics usage` will find it.
    assert (tmp_path / "metrics.db").exists()


def test_logs_go_to_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_sinks: list[str]):
    """stdout is the JSON-RPC channel; a log line there disconnects the client."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr("mcp.server.mcpserver.MCPServer.run_stdio_async", _noop)
    result = runner.invoke(
        app, ["mcp", "--config", str(_config_file(tmp_path)), "--data-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert log_sinks == ["stderr"]


def test_reports_a_missing_config_rather_than_starting(monkeypatch: pytest.MonkeyPatch):
    """With nothing to find, the error names every path it looked in.

    A search path moves the answer to "which file is it reading?" out of one env var, so the
    failure has to carry it instead — `MOM_CONFIG is unset` would now be a half-truth.
    """
    monkeypatch.delenv("MOM_CONFIG", raising=False)
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ConfigError)
    message = str(result.exception)
    assert "no config found" in message
    for expected in (".mom/config.yaml", "mom.yaml", "MOM_CONFIG"):
        assert expected in message


def test_discovers_a_project_config_with_no_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The point of the whole feature: an MCP client entry is `{"command": "mom",
    "args": ["mcp"]}` — no --config, no env block, whatever directory it launches in."""
    (tmp_path / "cwd" / "mom.yaml").write_text(CONFIG)
    result = runner.invoke(app, ["mcp", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_cli_overrides_beat_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`config_file` carries a validation alias, so the override has to go through model_copy —
    passing it as a plain field name would silently not bind. It also has to go in as a `Path`:
    the field is typed `Path | None` and model_copy does not validate, so a `str` here would be a
    lie every caller downstream has to keep re-wrapping."""
    from mom.runtime.bootstrap import bootstrap

    monkeypatch.setenv("MOM_CONFIG", str(tmp_path / "not-this-one.yaml"))
    booted = bootstrap(config=_config_file(tmp_path), data_dir=tmp_path / "data")
    assert booted.settings.config_file == tmp_path / "config.yaml"
    assert booted.settings.data_dir == tmp_path / "data"
    assert booted.sources.files == (tmp_path / "config.yaml",)
