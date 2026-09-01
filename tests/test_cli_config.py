"""`mom config validate` / `show` / `where` CLI tests.

The explicit-path invocations here are the back-compat contract: making the positional optional
must not break the form every doc and script already uses.
"""

from __future__ import annotations

import os
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


# ---- discovery ----------------------------------------------------------------------------------
CONFIG = (
    "version: 2\n"
    "llms: { a: { model: openai/a } }\n"
    "ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }\n"
)


def _project(tmp_path: Path) -> Path:
    """A project config in the (hermetic, empty) working directory the fixture chdir'd into."""
    path = Path.cwd() / "mom.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_validate_with_no_path_uses_discovery(tmp_path: Path):
    _project(tmp_path)
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.stdout


def test_validate_with_no_path_and_nothing_to_find_lists_what_it_checked(tmp_path: Path):
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 1
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "no config found" in combined
    assert "mom.yaml" in combined


def test_show_with_no_path_uses_discovery(tmp_path: Path):
    _project(tmp_path)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "ensemble e" in result.stdout


def test_show_reads_a_lone_argument_as_an_ensemble_when_it_is_not_a_file(tmp_path: Path):
    """Both positionals are optional now, so `show e` is ambiguous with `show ./mom.yaml`.
    Resolved by looking: an argument that is not a file on disk was meant as an ensemble."""
    _project(tmp_path)
    result = runner.invoke(app, ["config", "show", "e"])
    assert result.exit_code == 0, result.output
    assert "ensemble e" in result.stdout


def test_show_rejects_a_lone_argument_that_is_neither(tmp_path: Path):
    _project(tmp_path)
    result = runner.invoke(app, ["config", "show", "ghost"])
    assert result.exit_code == 1


# ---- mom config where ---------------------------------------------------------------------------
def test_where_reports_checked_found_and_merge_order(tmp_path: Path):
    config = _project(tmp_path)
    result = runner.invoke(app, ["config", "where"])
    assert result.exit_code == 0, result.output
    assert "mode: discovery" in result.stdout
    assert str(config) in result.stdout
    assert "merge order" in result.stdout
    assert ".mom/config.yaml" in result.stdout  # the candidate it did *not* use is still shown


def test_where_answers_even_with_no_config_at_all(tmp_path: Path):
    """A search path is hardest to reason about exactly when it produced nothing, so `where`
    must not require a resolvable config the way validate/show do."""
    result = runner.invoke(app, ["config", "where"])
    assert result.exit_code == 0, result.output
    assert "mom has no config to serve" in result.stdout


def test_where_reports_pinning(tmp_path: Path):
    pinned = tmp_path / "pinned.yaml"
    pinned.write_text(CONFIG, encoding="utf-8")
    result = runner.invoke(app, ["config", "where", str(pinned)])
    assert result.exit_code == 0, result.output
    assert "pinned (discovery off)" in result.stdout


def test_where_prints_secret_names_never_values(tmp_path: Path):
    """A leak here would be silent and permanent — `where` is the one command whose whole job is
    to print what it found about credentials."""
    (Path.cwd() / ".env").write_text("OPENAI_API_KEY=super-secret-value\n", encoding="utf-8")
    (Path.cwd() / "auth.json").write_text(
        '{"ANTHROPIC_API_KEY": "another-secret"}\n', encoding="utf-8"
    )
    _project(tmp_path)

    result = runner.invoke(app, ["config", "where"])
    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY" in result.stdout
    assert "ANTHROPIC_API_KEY" in result.stdout
    assert "super-secret-value" not in result.stdout
    assert "another-secret" not in result.stdout


def test_where_does_not_apply_the_secrets_it_describes(tmp_path: Path, monkeypatch):
    (Path.cwd() / ".env").write_text("OPENAI_API_KEY=preview-only\n", encoding="utf-8")
    _project(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert runner.invoke(app, ["config", "where"]).exit_code == 0
    assert os.environ.get("OPENAI_API_KEY") != "preview-only"
