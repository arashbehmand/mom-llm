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


# ---- empty values are not definitions ------------------------------------------------------------
def test_an_empty_value_does_not_shadow_a_real_one_below_it(tree: Path, monkeypatch):
    """`.env.example` shipped LANGFUSE_*="" for a while, so this is the shape a copied example
    produces: an empty at the project level over a real value at the user level."""
    _write(tree / "proj" / ".env", 'OPENAI_API_KEY=""\n')
    _write(tree / "home" / ".mom" / ".env", "OPENAI_API_KEY=real-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolve_secrets(_collected(tree), apply=True)
    assert os.environ["OPENAI_API_KEY"] == "real-key"


def test_an_empty_process_variable_is_overridden_not_merely_reported(
    tree: Path, monkeypatch: pytest.MonkeyPatch
):
    """`taken` tests truthiness while `setdefault` tested presence — so an empty in the process
    env was reported as replaced and then left empty."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=from-file\n")

    resolve_secrets(_collected(tree), apply=True)
    assert os.environ["OPENAI_API_KEY"] == "from-file"


def test_applied_never_names_something_the_environment_did_not_receive(tree: Path, monkeypatch):
    """The invariant behind the two tests above: the report and the write agree."""
    monkeypatch.setenv("XAI_API_KEY", "")
    _write(tree / "proj" / ".env", 'XAI_API_KEY=real\nGROQ_API_KEY=""\nCOHERE_API_KEY=c\n')

    for source in resolve_secrets(_collected(tree), apply=True):
        for name in source.applied:
            assert os.environ.get(name) == source.values[name]
    assert "GROQ_API_KEY" not in os.environ  # an empty line defines nothing at all


def test_an_unreadable_dotenv_never_reaches_settings(tree: Path):
    """Reported as a warning, not handed to `Settings(_env_file=…)` — python-dotenv would reopen
    it there and raise, turning a soft warning into a startup failure."""
    path = _write(tree / "proj" / ".env", "OPENAI_API_KEY=x\n", mode=0o000)
    try:
        collected = _collected(tree)
        source = next(s for s in collected if s.path == path)
        assert source.found is False
        assert source.warning is not None
        assert path not in dotenv_files(collected)
    finally:
        path.chmod(0o600)


# ---- what `config where` needs to be able to say -------------------------------------------------
def test_mom_names_are_reported_separately_rather_than_as_nothing(tree: Path):
    """A file whose whole contribution is MOM_API_TOKEN is the file authenticating the gateway.
    It cannot report as having contributed nothing."""
    _write(tree / "home" / ".mom" / ".env", "MOM_API_TOKEN=t\n")
    resolved = resolve_secrets(_collected(tree), environ={}, apply=False)
    source = next(s for s in resolved if s.path == tree / "home" / ".mom" / ".env")
    assert source.settings_names == ("MOM_API_TOKEN",)
    assert source.applied == ()


def test_a_file_beaten_by_the_environment_is_distinguishable_from_an_empty_one(tree: Path):
    _write(tree / "proj" / ".env", "OPENAI_API_KEY=loser\n")
    resolved = resolve_secrets(_collected(tree), environ={"OPENAI_API_KEY": "winner"}, apply=False)
    source = next(s for s in resolved if s.path == tree / "proj" / ".env")
    assert source.shadowed == ("OPENAI_API_KEY",)
    assert source.applied == ()


def test_opencode_reports_oauth_providers_it_could_not_use(tree: Path):
    """The most useful thing the bridge can say: a provider mom fully supports, authenticated by
    a route mom cannot use. It used to `continue` before recording anything."""
    _opencode(tree)
    source = next(
        s
        for s in _collected(tree, auth_from_opencode=True, home=tree / "home")
        if s.kind == "opencode"
    )
    assert source.warning is not None
    assert "oauth" in source.warning
    assert "openai" in source.warning  # the oauth entry in OPENCODE


def test_a_rejected_auth_json_key_is_counted_not_printed(tree: Path):
    """A reversed mapping puts the credential in the key position, and this module promises
    never to print one."""
    _write(
        tree / "proj" / "auth.json",
        json.dumps({"sk-ant-super-secret": "ANTHROPIC_API_KEY"}),
        mode=0o600,
    )
    source = next(s for s in _collected(tree) if s.kind == "auth_json" and s.found)
    assert source.warning is not None
    assert "sk-ant-super-secret" not in source.warning
    assert "skipped 1 entry" in source.warning


def test_user_secret_dirs_follow_candidate_order_not_where_the_yaml_landed(tmp_path: Path):
    """`~/.mom` outranks `~/.config/mom` for config; secrets must not invert that just because
    the XDG dir happened to hold the config.yaml."""
    home = tmp_path / "home"
    _write(home / ".config" / "mom" / "config.yaml", CONFIG)
    (tmp_path / "proj").mkdir(exist_ok=True)
    sources = discover(cwd=tmp_path / "proj", home=home)
    assert sources.secret_dirs == (
        tmp_path / "proj",
        home / ".mom",
        home / ".config" / "mom",
    )


def test_unpinned_discovery_leaves_config_file_none(tree: Path, monkeypatch):
    """`settings.config_file` is only ever the pin. A discovered `.env` can carry MOM_CONFIG —
    correctly ignored for discovery — but pydantic would still bind it, leaving `config_file`
    naming a file that was never loaded while `sources.files` holds the merge that ran."""
    _write(tree / "home" / ".mom" / ".env", "MOM_CONFIG=/etc/never-loaded.yaml\n")
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    booted = bootstrap()
    assert booted.sources.pinned is False
    assert booted.sources.files == (tree / "proj" / "mom.yaml",)
    assert booted.settings.config_file is None


def test_a_project_dotenv_beats_a_user_one_for_mom_settings(tree: Path, monkeypatch):
    """The load-bearing assumption: pydantic-settings layers a sequence of `_env_file`s
    later-wins. All MOM_* precedence rests on it, and only the path *order* was asserted — a
    dependency bump that changed the merge direction would have passed the suite silently.
    """
    _write(tree / "home" / ".mom" / ".env", "MOM_API_TOKEN=user\nMOM_LOG_LEVEL=WARNING\n")
    _write(tree / "proj" / ".env", "MOM_API_TOKEN=project\n")
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    settings = bootstrap().settings
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "project"
    assert settings.log_level == "WARNING"  # the user level still fills what the project omits
