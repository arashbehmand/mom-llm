"""Where the config and the secret files live — a search path, resolved from machine-local facts.

Two levels, deep-merged like git config: a user level (``~/.mom`` or ``$XDG_CONFIG_HOME/mom``)
holding the llms and keys a machine has once, and a project level (``./mom.yaml`` or
``./.mom/config.yaml``) holding only what this directory adds. The project wins. Validation
happens after the merge, so neither level has to be a complete config on its own.

Pinning with ``--config``/``MOM_CONFIG`` turns discovery **off**: a server told exactly which
file to serve must not also pick up whatever happens to sit in ``$HOME``.

This module is pure in the way that matters for testing: every machine fact it depends on —
``cwd``, ``home``, ``$XDG_CONFIG_HOME`` — is a parameter, not a global read, so the search path
can be exercised against a synthetic tree without chdir or ``$HOME`` surgery. It touches the
filesystem only to ask whether a candidate exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


#: Filenames probed at each level, in priority order. First **found** wins; they do not stack.
USER_CANDIDATES = ("config.yaml",)
PROJECT_CANDIDATES = ("mom.yaml", ".mom/config.yaml")


@dataclass(frozen=True, slots=True)
class Probe:
    """One candidate path and what became of it — the raw material for ``mom config where``."""

    path: Path
    role: str
    found: bool
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigSources:
    """The resolved search path: what was checked, what will be merged, where secrets live."""

    pinned: bool
    checked: tuple[Probe, ...] = ()
    #: Merge order, lowest precedence first. Explicitly-named files (a pinned config, an
    #: ``MOM_CONFIG_OVERLAY``) appear even when missing, so naming a file that is not there
    #: fails loudly; discovered candidates appear only when found.
    files: tuple[Path, ...] = ()
    #: Directories to search for ``.env`` / ``auth.json``, highest precedence first.
    secret_dirs: tuple[Path, ...] = ()
    #: Level-wide remarks that belong to no single path (e.g. an unresolvable home directory).
    notes: tuple[str, ...] = field(default_factory=tuple)


def override_for(base: Path) -> Path:
    """The sibling override beside ``base``, named from its stem.

    ``config.yaml`` -> ``config.override.yaml``, and — the reason this is derived rather than a
    fixed filename — ``mom.yaml`` -> ``mom.override.yaml``, so a project that names its config
    ``mom.yaml`` does not grow a surprising ``config.override.yaml``, and an arbitrary
    ``--config /etc/mom/prod.yaml`` still has a well-defined sibling.
    """
    return base.with_name(f"{base.stem}.override{base.suffix}")


def user_config_dirs(home: Path | None, xdg_config_home: Path | None) -> tuple[Path, ...]:
    """The user-level directories, in priority order.

    Deliberately the literal XDG rule rather than ``platformdirs.user_config_dir``: on macOS
    platformdirs answers ``~/Library/Application Support``, and the config a user hand-edits
    belongs where they can find it. (``platformdirs`` still owns the *data* dir, which is
    machine state rather than something anyone edits.) A relative ``$XDG_CONFIG_HOME`` is
    ignored, as the spec requires.
    """
    absolute_xdg = xdg_config_home if xdg_config_home and xdg_config_home.is_absolute() else None
    if home is None:
        # No home does not mean no config: a container run as a uid with no passwd entry can
        # still be pointed at one with $XDG_CONFIG_HOME, which is the whole reason that variable
        # is absolute-only.
        return (absolute_xdg / "mom",) if absolute_xdg else ()
    dirs = [home / ".mom", (absolute_xdg or home / ".config") / "mom"]
    return _dedup(dirs)


def _dedup(paths: Iterable[Path], *, keep: Literal["first", "last"] = "first") -> tuple[Path, ...]:
    """Order-preserving de-duplication by resolved identity.

    Two entries can name the same file through a symlink, a repeated ``$XDG_CONFIG_HOME``, or an
    ``MOM_CONFIG_OVERLAY`` that points at a file discovery already picked up.

    ``keep`` matters because the two lists this serves are ordered in opposite directions.
    ``secret_dirs`` runs highest-precedence-first, so the winner is the *first* occurrence.
    ``files`` runs lowest-precedence-first, so it is the *last* — dropping the later copy would
    demote a file the operator explicitly asked to apply last, and let an intervening layer
    override it.
    """
    ordered = list(paths)
    if keep == "last":
        ordered.reverse()
    seen: set[Path] = set()
    out: list[Path] = []
    for path in ordered:
        try:
            key = path.resolve()
        except OSError:  # pragma: no cover - resolve() only fails on exotic filesystems
            key = path.absolute()
        if key not in seen:
            seen.add(key)
            out.append(path)
    if keep == "last":
        out.reverse()
    return tuple(out)


def _first_found(candidates: Sequence[Path], role: str, probes: list[Probe]) -> Path | None:
    """Probe candidates in order, recording each; return the first that exists.

    Candidates after the winner are still recorded — with a note saying why they were not
    consulted — because "I put it in the other file" is the single most likely way to be
    confused by a search path.
    """
    winner: Path | None = None
    for candidate in candidates:
        if winner is not None:
            probes.append(Probe(candidate, role, False, f"skipped — {winner} already matched"))
        elif candidate.is_file():
            probes.append(Probe(candidate, role, True))
            winner = candidate
        else:
            probes.append(Probe(candidate, role, False))
    return winner


def _with_override(base: Path | None, role: str, probes: list[Probe]) -> tuple[Path, ...]:
    """A found base plus its sibling override, if that exists too."""
    if base is None:
        return ()
    sibling = override_for(base)
    found = sibling.is_file()
    probes.append(Probe(sibling, f"{role} override", found))
    return (base, sibling) if found else (base,)


def _pinned_sources(config: Path, probes: list[Probe]) -> tuple[tuple[Path, ...], Path]:
    """A pinned config contributes itself (present or not) and its sibling override."""
    probes.append(Probe(config, "pinned config", config.is_file()))
    sibling = override_for(config)
    found = sibling.is_file()
    probes.append(Probe(sibling, "pinned override", found))
    return ((config, sibling) if found else (config,)), config.parent


def discover(
    *,
    config: Path | None = None,
    overlay: Path | None = None,
    cwd: Path,
    home: Path | None,
    xdg_config_home: Path | None = None,
) -> ConfigSources:
    """Resolve the config search path and the directories secrets are read from.

    ``home`` is ``None`` when the caller could not determine one — a container run as a uid with
    no passwd entry and no ``$HOME``. That skips the user level rather than failing: a pinned or
    project-level config still serves.
    """
    probes: list[Probe] = []
    notes: list[str] = []
    files: list[Path] = []
    secret_dirs: list[Path] = []

    if config is not None:
        pinned_files, pinned_dir = _pinned_sources(config, probes)
        files.extend(pinned_files)
        # The pinned file's own directory, then cwd. Pinning turns off *config* discovery, not
        # the working directory's .env — `MOM_CONFIG=tools/live_config.example.yaml` pins a
        # config in a subdirectory and must still pick up ./.env. The user level stays out:
        # that is the "no stray files from $HOME" half of pinning.
        secret_dirs.extend((pinned_dir, cwd))
    else:
        user_dirs = user_config_dirs(home, xdg_config_home)
        if not user_dirs:
            notes.append("user level skipped: no home directory could be determined")
        # Probed in merge order — user first, project over it (global defaults, local
        # additions) — so the report reads in the order the files are actually applied.
        user_base = _first_found(
            [d / name for d in user_dirs for name in USER_CANDIDATES], "user config", probes
        )
        files.extend(_with_override(user_base, "user", probes))
        project_base = _first_found(
            [cwd / name for name in PROJECT_CANDIDATES], "project config", probes
        )
        files.extend(_with_override(project_base, "project", probes))
        secret_dirs.extend(_secret_dirs(project_base, cwd, user_dirs))

    if overlay is not None:
        # Named explicitly, so it is merged whether or not it exists — a typo'd overlay path
        # should fail, not silently contribute nothing.
        probes.append(Probe(overlay, "config overlay", overlay.is_file()))
        files.append(overlay)

    return ConfigSources(
        pinned=config is not None,
        checked=tuple(probes),
        files=_dedup(files, keep="last"),
        secret_dirs=_dedup(secret_dirs),
        notes=tuple(notes),
    )


def _secret_dirs(
    project_base: Path | None,
    cwd: Path,
    user_dirs: Sequence[Path],
) -> tuple[Path, ...]:
    """Directories to read ``.env`` / ``auth.json`` from, highest precedence first.

    Two departures from "wherever that level's config.yaml was found", both because the naive
    rule loses files people will certainly have:

    * **cwd is always here**, not only when it happens to be the project level. Otherwise
      ``MOM_CONFIG=tools/live_config.example.yaml`` — pinning a config that lives in a
      subdirectory — would stop reading ``./.env`` and silently lose every key.
    * **the user level contributes its directories even with no config.yaml there.** The most
      likely setup is a project ``mom.yaml`` for ensembles and ``~/.mom/.env`` for keys, with no
      user-level YAML at all; deriving the directory from a config that does not exist would
      make those keys invisible.
    """
    dirs: list[Path] = []
    if project_base is not None:
        dirs.append(project_base.parent)
    dirs.append(cwd)
    # Candidate order, not "wherever the YAML was found". `~/.mom` outranks `~/.config/mom` for
    # config, and secrets must not silently invert that when only the XDG dir happens to hold a
    # config.yaml — "the catalog is in XDG, the keys are in ~/.mom/.env" is a reasonable layout.
    dirs.extend(user_dirs)
    return _dedup(dirs)
