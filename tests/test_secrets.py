"""The secrets search path: precedence, `auth.json`, and the opencode bridge.

Fixtures are always written under `tmp_path` — a test must never read a developer's real `.env`,
and after this change that includes never reading their real `~/.mom` or opencode credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mom.runtime.bootstrap import bootstrap
from mom.runtime.discovery import ConfigSources, discover
from mom.runtime.secrets import (
    collect_secrets,
    dotenv_files,
    opencode_auth_path,
    resolve_secrets,
)


CONFIG = "version: 2\nllms: { a: { model: openai/a } }\nensembles: {}\n"


def _write(path: Path, text: str, *, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)
    return path


def _collected(tree: Path, **kwargs: object):
    sources = discover(cwd=tree / "proj", home=tree / "home")
    return collect_secrets(sources, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _write(tmp_path / "proj" / "mom.yaml", CONFIG)
    (tmp_path / "home").mkdir(exist_ok=True)
    return tmp_path


def _applied(collected) -> dict[str, str]:
    """The names each source won, flattened — what `resolve_secrets` would set."""
    return {name: str(src.path) for src in collected for name in src.applied}


# ---- precedence ---------------------------------------------------------------------------------
def test_process_env_beats_every_file(tree: Path, monkeypatch: pytest.MonkeyPatch):
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    resolve_secrets(_collected(tree), apply=True)
    assert os.environ["OPENAI_API_KEY"] == "from-process"


def test_project_beats_user_and_dotenv_beats_auth_json(tree: Path):
    """Order within a level is .env then auth.json; order across levels is project then user."""
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=project-env\n")
    _write(tree / "proj" / "auth.json", json.dumps({"OPENAI_API_KEY": "project-auth"}))
    _write(tree / "home" / ".mom" / ".env", "OPENAI_API_KEY=user-env\nXAI_API_KEY=user-only\n")

    resolved = resolve_secrets(_collected(tree), environ={}, apply=False)
    applied = _applied(resolved)
    assert applied["OPENAI_API_KEY"] == str(tree / "proj" / ".env")
    assert applied["XAI_API_KEY"] == str(tree / "home" / ".mom" / ".env")


def test_apply_false_never_touches_the_environment(tree: Path, monkeypatch: pytest.MonkeyPatch):
    """`mom config where` previews this. Asking where a key would come from must not change
    where it comes from."""
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=preview\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = resolve_secrets(_collected(tree), environ={}, apply=False)
    assert "OPENAI_API_KEY" in _applied(resolved)
    assert "OPENAI_API_KEY" not in os.environ


def test_mom_names_reach_settings_but_not_the_process_environment(
    tree: Path, monkeypatch: pytest.MonkeyPatch
):
    """This is an auth gateway: MOM_API_TOKEN has no business being visible to every subprocess.
    It reaches Settings through the dotenv source instead."""
    _write(tree / "home" / ".mom" / ".env", "MOM_API_TOKEN=secret-token\n")
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    booted = bootstrap()
    assert "MOM_API_TOKEN" not in os.environ
    assert booted.settings.api_token is not None
    assert booted.settings.api_token.get_secret_value() == "secret-token"


def test_dotenv_files_are_ordered_lowest_precedence_first(tree: Path):
    """pydantic-settings layers a sequence with *later* files overriding earlier, which is the
    reverse of collection order."""
    _write(tree / "proj" / ".env", "MOM_HOST=project\n")
    _write(tree / "home" / ".mom" / ".env", "MOM_HOST=user\n")

    files = dotenv_files(_collected(tree))
    assert files == (tree / "home" / ".mom" / ".env", tree / "proj" / ".env")


# ---- auth.json ----------------------------------------------------------------------------------
def test_auth_json_is_a_flat_env_name_map(tree: Path):
    _write(tree / "proj" / "auth.json", json.dumps({"ANTHROPIC_API_KEY": "sk-x"}), mode=0o600)
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert dict(source.values) == {"ANTHROPIC_API_KEY": "sk-x"}
    assert source.warning is None


def test_auth_json_wider_than_0600_warns_but_still_loads(tree: Path):
    _write(tree / "proj" / "auth.json", json.dumps({"ANTHROPIC_API_KEY": "sk-x"}), mode=0o644)
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert dict(source.values) == {"ANTHROPIC_API_KEY": "sk-x"}
    assert source.warning is not None
    assert "0644" in source.warning


def test_a_group_readable_dotenv_does_not_warn(tree: Path):
    """A .env is conventionally 0644. Warning on every one of them would train the operator to
    skip the line, which is how the warning that matters gets missed."""
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=x\n", mode=0o644)
    source = next(s for s in _collected(tree) if s.kind == "dotenv" and s.found)
    assert source.warning is None


def test_malformed_auth_json_warns_and_is_skipped_rather_than_fatal(tree: Path):
    """Secrets are best-effort — a missing key surfaces as a clear failure on the call that
    needed it. A malformed *config*, by contrast, is fatal."""
    _write(tree / "proj" / "auth.json", "{not json", mode=0o600)
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert source.values == {}
    assert source.warning is not None


def test_auth_json_reports_mom_names_rather_than_dropping_them(tree: Path):
    _write(tree / "proj" / "auth.json", json.dumps({"MOM_API_TOKEN": "t"}), mode=0o600)
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert source.values == {}
    assert source.warning is not None
    assert "MOM_API_TOKEN" in source.warning


def test_auth_json_skips_non_string_and_malformed_keys(tree: Path):
    _write(
        tree / "proj" / "auth.json",
        json.dumps({"OPENAI_API_KEY": "ok", "lowercase": "x", "NESTED": {"a": 1}}),
        mode=0o600,
    )
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert dict(source.values) == {"OPENAI_API_KEY": "ok"}
    assert source.warning is not None


# ---- opencode bridge ----------------------------------------------------------------------------
OPENCODE = {
    "anthropic": {"type": "api", "key": "sk-ant"},
    "google": {"type": "api", "key": "goog"},
    "openai": {"type": "oauth", "access": "a", "refresh": "r", "expires": 0},
    "zai-coding-plan": {"type": "api", "key": "zai"},
}


def _opencode(tree: Path) -> Path:
    return _write(
        tree / "home" / ".local" / "share" / "opencode" / "auth.json",
        json.dumps(OPENCODE),
        mode=0o600,
    )


def test_opencode_is_ignored_unless_the_flag_is_set(tree: Path):
    _opencode(tree)
    assert not [s for s in _collected(tree) if s.kind == "opencode"]


def test_opencode_maps_api_entries_and_skips_oauth(tree: Path):
    """An oauth entry holds a refresh/access pair opencode renews, not an API key — handing one
    to litellm would fail at the provider with an opaque 401."""
    _opencode(tree)
    source = next(
        s
        for s in _collected(tree, auth_from_opencode=True, home=tree / "home")
        if s.kind == "opencode"
    )
    assert dict(source.values) == {"ANTHROPIC_API_KEY": "sk-ant", "GEMINI_API_KEY": "goog"}
    assert "OPENAI_API_KEY" not in source.values  # oauth
    assert source.warning is not None
    assert "zai-coding-plan" in source.warning  # no mom equivalent, reported not fatal


def test_opencode_is_lowest_precedence(tree: Path):
    _opencode(tree)
    _write(tree / "proj" / ".env", "ANTHROPIC_API_KEY=mine\n")
    resolved = resolve_secrets(
        _collected(tree, auth_from_opencode=True, home=tree / "home"), environ={}, apply=False
    )
    assert _applied(resolved)["ANTHROPIC_API_KEY"] == str(tree / "proj" / ".env")


def test_opencode_honours_xdg_data_home(tree: Path):
    assert opencode_auth_path(tree / "home", tree / "data") == (
        tree / "data" / "opencode" / "auth.json"
    )
    assert opencode_auth_path(tree / "home", None) == (
        tree / "home" / ".local" / "share" / "opencode" / "auth.json"
    )


def test_a_missing_opencode_file_is_not_an_error(tree: Path):
    source = next(
        s
        for s in _collected(tree, auth_from_opencode=True, home=tree / "home")
        if s.kind == "opencode"
    )
    assert source.found is False


# ---- leak regression ----------------------------------------------------------------------------
def test_secret_values_never_appear_in_a_repr(tree: Path):
    """A frozen dataclass renders its fields into tracebacks and log lines. This one holds live
    API keys, so `values` is excluded from repr — assert it stays that way."""
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=super-secret-value\n")
    _write(tree / "proj" / "auth.json", json.dumps({"XAI_API_KEY": "another-secret"}), mode=0o600)

    collected = _collected(tree)
    assert "super-secret-value" not in repr(collected)
    assert "another-secret" not in repr(collected)


def test_the_motivating_case_a_cwd_dotenv_key_reaches_the_provider_lookup(
    tree: Path, monkeypatch: pytest.MonkeyPatch
):
    """Before discovery, a provider key in ./.env reached `os.getenv` only because litellm calls
    `load_dotenv()` on import — and that walks up from litellm's *install* directory, so it
    worked only when .venv happened to sit inside the repo. Now it is deliberate, and litellm
    need not be imported at all."""
    from mom.adapters.litellm_client import _resolve_api_key

    _write(tree / "proj" / ".env", "OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    bootstrap()
    assert _resolve_api_key(("OPENAI_API_KEY",)) == "from-dotenv"


def test_secret_dirs_are_probed_even_where_no_config_lives(tree: Path):
    """`collect_secrets` reports a miss rather than omitting the path, so `mom config where` can
    show that it looked."""
    collected = collect_secrets(ConfigSources(pinned=False, secret_dirs=(tree / "nowhere",)))
    assert [(s.kind, s.found) for s in collected] == [("dotenv", False), ("auth_json", False)]
