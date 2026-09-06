"""Config discovery: the search path, the merge order, and what pinning turns off.

Most of these drive `discover()` directly with injected `cwd`/`home`, which is the whole reason
it takes them as parameters — the search path is exercised against a synthetic tree rather than
against whatever the machine running the suite happens to have in `$HOME`.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mom.config.resolve import ConfigError
from mom.runtime.bootstrap import bootstrap
from mom.runtime.discovery import discover, override_for, user_config_dirs


USER_LLMS = dedent("""
    version: 2
    llms:
      a: { model: openai/a }
      doomed: { model: openai/doomed }
""")

PROJECT_ENSEMBLES = dedent("""
    ensembles:
      e:
        members: [{ llm: a }]
        synthesizer: { llm: a }
""")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "proj").mkdir(exist_ok=True)
    return tmp_path


def _discover(tree: Path, **kwargs: object):
    return discover(cwd=tree / "proj", home=tree / "home", **kwargs)  # type: ignore[arg-type]


# ---- the search path ---------------------------------------------------------------------------
def test_user_dot_mom_wins_over_xdg_and_they_do_not_stack(tree: Path):
    """First found wins. Stacking them would make ~/.mom a silent partial override of the XDG
    file, which is a very confusing thing to debug."""
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS)
    _write(tree / "home" / ".config" / "mom" / "config.yaml", "version: 2\nllms: {}\n")
    _write(tree / "proj" / "mom.yaml", PROJECT_ENSEMBLES)

    sources = _discover(tree)
    assert sources.files == (tree / "home" / ".mom" / "config.yaml", tree / "proj" / "mom.yaml")
    skipped = [
        p
        for p in sources.checked
        if p.role == "user config" and p.note and "already matched" in p.note
    ]
    assert [p.path for p in skipped] == [tree / "home" / ".config" / "mom" / "config.yaml"]
    # Both candidates are named config.yaml, so the note has to name the winning *path*.
    assert str(tree / "home" / ".mom" / "config.yaml") in skipped[0].note


def test_xdg_is_used_when_dot_mom_is_absent(tree: Path):
    _write(tree / "home" / ".config" / "mom" / "config.yaml", USER_LLMS)
    assert _discover(tree).files == (tree / "home" / ".config" / "mom" / "config.yaml",)


def test_explicit_xdg_config_home_is_honoured(tree: Path):
    _write(tree / "elsewhere" / "mom" / "config.yaml", USER_LLMS)
    sources = discover(cwd=tree / "proj", home=tree / "home", xdg_config_home=tree / "elsewhere")
    assert sources.files == (tree / "elsewhere" / "mom" / "config.yaml",)


def test_relative_xdg_config_home_is_ignored(tree: Path):
    """The XDG spec says a relative $XDG_CONFIG_HOME must be treated as unset."""
    assert user_config_dirs(tree / "home", Path("relative/path")) == (
        tree / "home" / ".mom",
        tree / "home" / ".config" / "mom",
    )


def test_project_mom_yaml_wins_over_dot_mom_directory(tree: Path):
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    _write(tree / "proj" / ".mom" / "config.yaml", "version: 2\n")
    assert _discover(tree).files == (tree / "proj" / "mom.yaml",)


def test_plain_config_yaml_in_cwd_is_not_a_candidate(tree: Path):
    """`./config.yaml` is too generic a name to claim in an arbitrary directory — this repo has
    one of its own, and so will plenty of unrelated projects."""
    _write(tree / "proj" / "config.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    assert _discover(tree).files == ()


def test_no_upward_walk_from_cwd(tree: Path):
    """A parent's mom.yaml is not inherited; the user level is what covers running from a
    subdirectory."""
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    (tree / "proj" / "nested").mkdir()
    sources = discover(cwd=tree / "proj" / "nested", home=tree / "home")
    assert sources.files == ()


def test_an_absolute_xdg_root_is_usable_without_a_home(tree: Path):
    """No home does not mean no config: a container run as a uid with no passwd entry can still
    be pointed at one with $XDG_CONFIG_HOME, which is why that variable is absolute-only."""
    _write(tree / "elsewhere" / "mom" / "config.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sources = discover(cwd=tree / "proj", home=None, xdg_config_home=tree / "elsewhere")
    assert sources.files == (tree / "elsewhere" / "mom" / "config.yaml",)


def test_missing_home_skips_the_user_level_with_a_note(tree: Path):
    """A container run as a uid with no passwd entry has no home; a project config still serves."""
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sources = discover(cwd=tree / "proj", home=None)
    assert sources.files == (tree / "proj" / "mom.yaml",)
    assert any("no home directory" in note for note in sources.notes)


# ---- sibling overrides -------------------------------------------------------------------------
def test_sibling_override_is_named_from_the_base_stem():
    assert override_for(Path("/x/config.yaml")) == Path("/x/config.override.yaml")
    assert override_for(Path("/x/mom.yaml")) == Path("/x/mom.override.yaml")
    assert override_for(Path("/etc/mom/prod.yaml")) == Path("/etc/mom/prod.override.yaml")


def test_each_level_layers_its_own_override(tree: Path):
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS)
    _write(tree / "home" / ".mom" / "config.override.yaml", "llms: { doomed: null }\n")
    _write(tree / "proj" / "mom.yaml", PROJECT_ENSEMBLES)
    _write(tree / "proj" / "mom.override.yaml", "server: { auth: none }\n")

    sources = _discover(tree)
    assert [p.name for p in sources.files] == [
        "config.yaml",
        "config.override.yaml",
        "mom.yaml",
        "mom.override.yaml",
    ]


# ---- merge semantics ---------------------------------------------------------------------------
def test_a_project_file_may_be_only_ensembles_over_user_llms(tree: Path, monkeypatch):
    """Validation runs once, after the merge — so no single layer has to satisfy the schema.
    This is the machine-wide-catalog case the whole feature exists for."""
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS)
    _write(tree / "proj" / "mom.yaml", "version: 2\n" + PROJECT_ENSEMBLES)
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    catalog = bootstrap().catalog()
    assert sorted(catalog.llms) == ["a", "doomed"]
    assert sorted(catalog.ensembles) == ["e"]


def test_null_in_a_later_layer_masks_an_inherited_key(tree: Path, monkeypatch):
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS)
    _write(tree / "proj" / "mom.yaml", "version: 2\nllms: { doomed: null }\n" + PROJECT_ENSEMBLES)
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    assert sorted(bootstrap().catalog().llms) == ["a"]


def test_an_override_can_drop_one_model_from_a_panel_it_did_not_author(tree: Path, monkeypatch):
    """The reason `members_exclude` exists. `members:` is a LIST, and a list replaces wholesale on
    merge — so an override that wanted this panel minus one model used to have to restate the
    whole roster (and then quietly stop tracking the tracked config it copied). This is the
    machine-local shape: a tracked config with the full panel, an untracked override that thins
    it, and a base file nobody has to edit."""
    _write(
        tree / "home" / ".mom" / "config.yaml",
        USER_LLMS
        + dedent("""
            ensembles:
              panel:
                members: [a, doomed]
                synthesizer: { llm: a }
        """),
    )
    _write(
        tree / "home" / ".mom" / "config.override.yaml",
        "ensembles: { panel: { members_exclude: doomed } }\n",
    )
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    catalog = bootstrap().catalog()
    assert [m.identity for m in catalog.ensembles["panel"].members] == ["a"]
    # The llm itself is untouched — still callable by name, just no longer on this panel.
    assert "doomed" in catalog.llms


def test_config_overlay_merges_last_and_is_deduped(tree: Path):
    """An MOM_CONFIG_OVERLAY that names a file discovery already picked up must not be merged
    twice — harmless for YAML, but a lie in `mom config where`."""
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sibling = _write(tree / "proj" / "mom.override.yaml", "server: { auth: none }\n")
    assert _discover(tree, overlay=sibling).files == (
        tree / "proj" / "mom.yaml",
        sibling,
    )


def test_a_duplicated_overlay_keeps_its_last_place_not_its_first(tree: Path):
    """De-duplication has to drop the *earlier* copy here. `files` runs lowest-precedence-first,
    so keeping the first occurrence would demote the file the operator explicitly asked to apply
    last and let an intervening layer override it."""
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS + "server: { public_url: 'base' }\n")
    user_override = _write(
        tree / "home" / ".mom" / "config.override.yaml", "server: { public_url: 'OVERLAY' }\n"
    )
    _write(tree / "proj" / "mom.yaml", PROJECT_ENSEMBLES + "server: { public_url: 'project' }\n")

    sources = _discover(tree, overlay=user_override)
    assert sources.files[-1] == user_override
    from mom.config.loader import load_layered

    assert load_layered(sources.files).config.server.public_url == "OVERLAY"


def test_a_named_overlay_that_does_not_exist_is_still_merged_and_so_fails(tree: Path):
    """Discovered candidates are optional; a file you *named* is not. A typo'd overlay path
    should say so rather than silently contributing nothing."""
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sources = _discover(tree, overlay=tree / "nope.yaml")
    assert tree / "nope.yaml" in sources.files
    with pytest.raises(FileNotFoundError):
        bootstrap(config=tree / "proj" / "mom.yaml", overlay=tree / "nope.yaml").catalog()


# ---- pinning -----------------------------------------------------------------------------------
def test_pinning_ignores_both_levels_but_keeps_its_own_sibling(tree: Path):
    _write(tree / "home" / ".mom" / "config.yaml", USER_LLMS)
    _write(tree / "proj" / "mom.yaml", PROJECT_ENSEMBLES)
    pinned = _write(tree / "etc" / "prod.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    override = _write(tree / "etc" / "prod.override.yaml", "server: { auth: none }\n")

    sources = _discover(tree, config=pinned)
    assert sources.pinned is True
    assert sources.files == (pinned, override)


def test_pinning_keeps_cwd_secrets_but_drops_the_user_level(tree: Path):
    """ "No stray files from $HOME" is the point of pinning. The working directory is a different
    matter: `MOM_CONFIG=tools/live_config.example.yaml` pins a config in a subdirectory and must
    still read ./.env, which is where the keys are."""
    pinned = _write(tree / "proj" / "tools" / "live.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sources = _discover(tree, config=pinned)
    assert sources.secret_dirs == (tree / "proj" / "tools", tree / "proj")


def test_nothing_found_reports_every_path_it_checked(tree: Path, monkeypatch):
    monkeypatch.chdir(tree / "proj")
    monkeypatch.setenv("HOME", str(tree / "home"))
    with pytest.raises(ConfigError) as excinfo:
        bootstrap().catalog()
    message = str(excinfo.value)
    assert "no config found" in message
    for expected in (".mom/config.yaml", "mom.yaml", "config.yaml"):
        assert expected in message


# ---- secret directories ------------------------------------------------------------------------
def test_cwd_is_always_a_secret_dir_even_when_the_config_lives_in_dot_mom(tree: Path):
    _write(tree / "proj" / ".mom" / "config.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    sources = _discover(tree)
    assert sources.secret_dirs[:2] == (tree / "proj" / ".mom", tree / "proj")


def test_user_secret_dirs_are_searched_without_a_user_config(tree: Path):
    """The likeliest setup is a project mom.yaml plus ~/.mom/.env for keys and no user-level
    YAML at all. Deriving the directory from a config that does not exist would lose those."""
    _write(tree / "proj" / "mom.yaml", USER_LLMS + PROJECT_ENSEMBLES)
    assert _discover(tree).secret_dirs == (
        tree / "proj",
        tree / "home" / ".mom",
        tree / "home" / ".config" / "mom",
    )
