"""Reading ``.env`` / ``auth.json`` off the search path, and putting keys where they are read.

Provider credentials are dereferenced late and by name: ``litellm_client._resolve_api_key`` and
``_proxy_client`` call ``os.getenv`` at *call* time, as does the Langfuse wiring. So making a
discovered key usable means putting it in ``os.environ`` before the first call — there is no
object to thread it through.

Until now nothing in mom did that. Provider keys in a ``.env`` reached providers only because
litellm calls ``load_dotenv()`` when it is imported, and that resolves through ``find_dotenv()``,
which walks up from *litellm's own installed directory* rather than the working directory — so it
finds a repo's ``.env`` when ``.venv/`` happens to sit inside the repo, and finds nothing at all
from a system-wide install. This module makes it deliberate.

Two mechanisms, deliberately, because the two kinds of name are read by different code:

* **Provider and third-party names** (``ANTHROPIC_API_KEY``, a ``proxy_url_env``, ``LANGFUSE_*``)
  go into ``os.environ`` with ``setdefault``, so the process environment always outranks a file.
* **Names that configure mom itself never enter the process environment.** They are handed to
  ``Settings`` directly instead. This is an auth gateway: ``MOM_API_TOKEN`` has no business being
  visible to every subprocess mom spawns, or to anything that dumps the environment — and neither
  has ``API_TOKEN``, the legacy spelling of the same secret, which is why the set comes from
  :func:`~mom.runtime.settings.settings_env_names` rather than a ``MOM_`` prefix test.

An **empty value is treated as absent everywhere** — collected, reported, and applied. That is
the only reading under which "first definition wins" agrees with the consumers: litellm's
``_resolve_api_key`` already skips falsy values, so a ``KEY=`` line that shadowed a real key one
level down would produce a missing-key failure at the provider while this module reported the
shadowing file as the winner.

Warnings are returned as **data**, never logged from here. ``mom mcp`` writes JSON-RPC frames on
stdout, so a warning emitted before ``configure_logging(..., stream=sys.stderr)`` has run is a
protocol violation rather than noise; the caller logs these once logging is pointed somewhere
safe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal

from mom.config.resolve import infer_key_env_candidates
from mom.runtime.discovery import ConfigSources
from mom.runtime.settings import settings_env_names


SecretKind = Literal["dotenv", "auth_json", "opencode"]

#: The same shape ``api_key_env`` / ``proxy_url_env`` accept (``schema.EnvVarName``).
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: opencode's provider ids are mostly litellm's prefixes; this covers the ones that are not.
_OPENCODE_PROVIDER_ALIASES = {"google": "gemini", "google-vertex": "vertex_ai"}


@dataclass(frozen=True, slots=True)
class SecretSource:
    """One file on the secrets search path, and what it contributed.

    ``values`` is excluded from ``repr`` on purpose: a frozen dataclass renders its fields into
    tracebacks and log lines, and this one holds live API keys.
    """

    kind: SecretKind
    path: Path
    found: bool
    values: Mapping[str, str] = field(default_factory=dict, repr=False)
    #: Names this source actually won — those no higher-precedence source had already defined.
    applied: tuple[str, ...] = ()
    #: ``MOM_*`` names this source carries. They never enter ``os.environ`` (they reach
    #: ``Settings`` through ``dotenv_files``), so they cannot appear in ``applied`` — but a file
    #: whose whole contribution is ``MOM_API_TOKEN`` must not report as having contributed
    #: nothing. That file is why the gateway authenticates.
    settings_names: tuple[str, ...] = ()
    #: Names a higher-precedence source (usually the process environment) already defined, so
    #: this file lost. Distinguishes "beaten" from "empty" in ``mom config where``.
    shadowed: tuple[str, ...] = ()
    warning: str | None = None


def _mode_warning(path: Path) -> str | None:
    """Warn when an ``auth.json`` is readable beyond its owner.

    Only ``auth.json``. A ``.env`` is conventionally group-readable and warning about every one
    of them would train the operator to skip the line — which is how the warning that matters
    gets missed. ``auth.json`` is mom's own file, written for credentials, so 0600 is the
    expectation there.

    Never fatal: the operator may have made that choice deliberately, and refusing to start over
    a permission bit would be worse than the exposure it complains about.
    """
    if os.name == "nt":  # pragma: no cover - POSIX mode bits are meaningless on Windows
        return None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:  # pragma: no cover - the file was readable a moment ago
        return None
    if mode & 0o077:
        return f"mode {mode:04o} is wider than 0600"
    return None


def _read_dotenv(path: Path) -> SecretSource:
    from dotenv import dotenv_values

    if not path.is_file():
        return SecretSource("dotenv", path, found=False)
    try:
        parsed = dotenv_values(path, encoding="utf-8")
    except OSError as exc:
        # found=False on purpose: `dotenv_files` feeds found dotenvs to `Settings(_env_file=…)`,
        # where python-dotenv reopens them. Reporting a file we could not read as present would
        # turn a warning here into a raise there, which is not what "secrets are soft" means.
        return SecretSource("dotenv", path, found=False, warning=f"unreadable: {exc}")
    values = {k: v for k, v in parsed.items() if v and _ENV_NAME.match(k)}
    return SecretSource("dotenv", path, found=True, values=values)


def _read_auth_json(path: Path) -> SecretSource:
    """A flat ``{"ANTHROPIC_API_KEY": "sk-…"}`` map — the same vocabulary the config already uses
    when it names an ``api_key_env``.

    A malformed file is a warning, not an error: secrets are best-effort (a missing key surfaces
    as a clear upstream failure on the call that needed it), while a malformed *config* is fatal.
    """
    if not path.is_file():
        return SecretSource("auth_json", path, found=False)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return SecretSource("auth_json", path, found=True, warning=f"ignored — {exc}")
    if not isinstance(loaded, dict):
        return SecretSource(
            "auth_json", path, found=True, warning="ignored — top level must be an object"
        )

    values: dict[str, str] = {}
    rejected = 0
    reserved: list[str] = []
    for name, value in loaded.items():
        if not (
            isinstance(name, str) and _ENV_NAME.match(name) and isinstance(value, str) and value
        ):
            rejected += 1
        elif name.startswith("MOM_"):
            reserved.append(name)
        else:
            values[name] = value

    warnings = [w for w in (_mode_warning(path),) if w]
    if rejected:
        # The count, not the names: a reversed mapping ({"sk-ant-…": "ANTHROPIC_API_KEY"}) puts
        # the credential in the key position, and this module promises never to print one.
        warnings.append(
            f"skipped {rejected} entr{'y' if rejected == 1 else 'ies'} that are not "
            "UPPER_SNAKE_CASE name -> non-empty string"
        )
    if reserved:
        # There is no route from auth.json into Settings, and putting MOM_API_TOKEN into
        # os.environ is exactly what this module avoids. Say so rather than dropping it silently.
        warnings.append(
            f"MOM_* settings are not read from auth.json (put them in .env): "
            f"{', '.join(sorted(reserved))}"
        )
    return SecretSource(
        "auth_json", path, found=True, values=values, warning="; ".join(warnings) or None
    )


def opencode_auth_path(home: Path | None, xdg_data_home: Path | None) -> Path | None:
    """Where opencode keeps its credentials, mirroring its own xdg resolution."""
    if xdg_data_home and xdg_data_home.is_absolute():
        return xdg_data_home / "opencode" / "auth.json"
    if home is None:
        return None
    return home / ".local" / "share" / "opencode" / "auth.json"


def _read_opencode(path: Path) -> SecretSource:
    """Map opencode's ``type: "api"`` entries onto the env vars mom already infers per provider.

    Only ``api`` entries carry a usable key. An ``oauth`` entry holds a refresh/access token pair
    for a session opencode manages and renews — handing litellm a refresh token would fail at the
    provider with an opaque 401, so those are skipped rather than mapped. opencode also
    authenticates providers mom has no model prefix for (subscription plans and bundled
    gateways); those are reported, not treated as an error.
    """
    if not path.is_file():
        return SecretSource("opencode", path, found=False)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return SecretSource("opencode", path, found=True, warning=f"ignored — {exc}")
    if not isinstance(loaded, dict):
        return SecretSource(
            "opencode", path, found=True, warning="ignored — top level must be an object"
        )

    values: dict[str, str] = {}
    skipped: list[str] = []
    not_a_key: list[str] = []
    for provider, entry in loaded.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "api":
            # oauth / wellknown. Worth naming rather than passing over in silence: a provider mom
            # fully supports, authenticated by a route mom cannot use, is exactly the thing an
            # operator wonders about when the bridge "found nothing".
            not_a_key.append(str(provider))
            continue
        key = entry.get("key")
        prefix = _OPENCODE_PROVIDER_ALIASES.get(provider, provider)
        candidates = infer_key_env_candidates(f"{prefix}/x")
        if not isinstance(key, str) or not key or not candidates:
            skipped.append(str(provider))
            continue
        values.setdefault(candidates[0], key)

    notes = []
    if not_a_key:
        notes.append(
            "authenticated in opencode but not an API key (oauth/wellknown): "
            + ", ".join(sorted(not_a_key))
        )
    if skipped:
        notes.append(f"no mom equivalent for: {', '.join(sorted(skipped))}")
    return SecretSource(
        "opencode", path, found=True, values=values, warning="; ".join(notes) or None
    )


def collect_secrets(
    sources: ConfigSources,
    *,
    auth_from_opencode: bool = False,
    home: Path | None = None,
    xdg_data_home: Path | None = None,
) -> tuple[SecretSource, ...]:
    """Read every secret file on the search path, highest precedence first.

    Reading is separated from applying so ``mom config where`` can report exactly what *would*
    be set without touching the process environment.
    """
    collected: list[SecretSource] = []
    for directory in sources.secret_dirs:
        collected.append(_read_dotenv(directory / ".env"))
        collected.append(_read_auth_json(directory / "auth.json"))
    if auth_from_opencode:
        path = opencode_auth_path(home, xdg_data_home)
        if path is not None:
            collected.append(_read_opencode(path))
    return tuple(collected)


def resolve_secrets(
    collected: Sequence[SecretSource],
    *,
    environ: Mapping[str, str] | None = None,
    apply: bool = True,
) -> tuple[SecretSource, ...]:
    """Fill in each source's ``applied`` names, first definition wins; optionally set them.

    Names belonging to ``Settings`` are skipped here by design — they reach it through
    :func:`settings_values` instead, so they never become process-wide state. With
    ``apply=False`` nothing is mutated and the result is a pure preview.
    """
    env = os.environ if environ is None else environ
    taken = {name for name in env if env.get(name)}
    resolved: list[SecretSource] = []
    for source in collected:
        won: list[str] = []
        settings_names: list[str] = []
        shadowed: list[str] = []
        for name, value in source.values.items():
            if _is_settings_name(name):
                settings_names.append(name)
                continue
            if name in taken:
                shadowed.append(name)
                continue
            taken.add(name)
            won.append(name)
            if apply:
                # Not `setdefault`: that keys on presence while `taken` keys on truthiness, so an
                # empty variable already in the environment would be reported as overridden and
                # then left empty. An empty value is not a definition — override it.
                os.environ[name] = value
        resolved.append(
            replace(
                source,
                applied=tuple(won),
                settings_names=tuple(settings_names),
                shadowed=tuple(shadowed),
            )
        )
    return tuple(resolved)


def _is_settings_name(name: str) -> bool:
    """Whether this name configures mom rather than a provider.

    The ``MOM_`` prefix test stays alongside the derived set so a future setting is kept out of
    the environment from the moment it is added, not from the moment someone remembers to.
    """
    return name.startswith("MOM_") or name in settings_env_names()


def settings_values(collected: Sequence[SecretSource]) -> dict[str, str]:
    """The settings names the files define, merged highest-precedence-first.

    Handing ``Settings`` the *values* rather than the file paths is what makes the empty-is-absent
    rule apply to mom's own settings too. ``Settings(_env_file=…)`` re-reads and re-parses the raw
    files, so it never saw the filtering this module does: a project ``.env`` with a bare
    ``MOM_API_TOKEN=`` would beat a real token in ``~/.mom/.env`` and leave the gateway unable to
    authenticate anyone. It also drops a dependency on ``pydantic-settings`` layering a sequence
    of dotenv files later-wins, which was load-bearing for precedence and invisible in the code.
    """
    merged: dict[str, str] = {}
    for source in collected:
        if source.kind != "dotenv":
            continue
        for name in source.settings_names:
            merged.setdefault(name, source.values[name])
    return merged
