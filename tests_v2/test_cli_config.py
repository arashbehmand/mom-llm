"""`mom config validate` / `mom config show` CLI tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mom.cli import app


runner = CliRunner()
EXAMPLE = str(Path(__file__).parent.parent / "config.example.yaml")


def test_validate_ok():
    result = runner.invoke(app, ["config", "validate", EXAMPLE])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_validate_rejects_bad_config(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 2\nllms: { a: { model: x/y } }\n"
        "ensembles: { e: { members: [{ llm: ghost }], synthesizer: { llm: a } } }\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["config", "validate", str(bad)])
    assert result.exit_code == 1
    # Error goes to stderr (err=True); Click may separate the streams by version.
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "unknown llm" in combined


def test_show_ensemble_renders_effort_matrix():
    result = runner.invoke(app, ["config", "show", EXAMPLE, "bmom"])
    assert result.exit_code == 0
    assert "skip/medium/high" in result.stdout  # flash excluded at low
    assert "pass/pass/pass" in result.stdout  # gemini relays client effort


def test_show_unknown_ensemble_exits_1():
    result = runner.invoke(app, ["config", "show", EXAMPLE, "ghost"])
    assert result.exit_code == 1
