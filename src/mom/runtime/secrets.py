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
* **``MOM_*`` names never enter the process environment.** ``Settings`` reads them from the
  discovered ``.env`` files directly (``pydantic-settings`` takes a *sequence* of dotenv files,
  later overriding earlier, with the real environment still outranking all of them). This is an
  auth gateway: ``MOM_API_TOKEN`` has no business being visible to every subprocess mom spawns,
  or to anything that dumps the environment.

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
        return SecretSource("dotenv", path, found=True, warning=f"unreadable: {exc}")
    values = {k: v for k, v in parsed.items() if v is not None and _ENV_NAME.match(k)}
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
    rejected: list[str] = []
    reserved: list[str] = []
    for name, value in loaded.items():
        if not (isinstance(name, str) and _ENV_NAME.match(name) and isinstance(value, str)):
            rejected.append(str(name))
        elif name.startswith("MOM_"):
            reserved.append(name)
        else:
            values[name] = value

    warnings = [w for w in (_mode_warning(path),) if w]
    if rejected:
        warnings.append(f"skipped non-string or malformed keys: {', '.join(sorted(rejected))}")
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
    for provider, entry in loaded.items():
        if not isinstance(entry, dict) or entry.get("type") != "api":
            continue  # oauth / wellknown: not an API key
        key = entry.get("key")
        prefix = _OPENCODE_PROVIDER_ALIASES.get(provider, provider)
        candidates = infer_key_env_candidates(f"{prefix}/x")
        if not isinstance(key, str) or not key or not candidates:
            skipped.append(str(provider))
            continue
        values.setdefault(candidates[0], key)

    warning = f"no mom equivalent for: {', '.join(sorted(skipped))}" if skipped else None
    return SecretSource("opencode", path, found=True, values=values, warning=warning)


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

    ``MOM_*`` is skipped here by design — those reach ``Settings`` through ``dotenv_files``
    instead, so they never become process-wide state. With ``apply=False`` nothing is mutated
    and the result is a pure preview.
    """
    env = os.environ if environ is None else environ
    taken = {name for name in env if env.get(name)}
    resolved: list[SecretSource] = []
    for source in collected:
        won: list[str] = []
        for name, value in source.values.items():
            if name.startswith("MOM_") or name in taken:
                continue
            taken.add(name)
            won.append(name)
            if apply:
                os.environ.setdefault(name, value)
        resolved.append(replace(source, applied=tuple(won)))
    return tuple(resolved)


def dotenv_files(collected: Sequence[SecretSource]) -> tuple[Path, ...]:
    """The discovered ``.env`` files, ordered **lowest** precedence first for ``pydantic-settings``.

    ``Settings(_env_file=…)`` layers a sequence with later files overriding earlier ones, which
    is the reverse of the order secrets are collected in — hence the flip.
    """
    return tuple(
        source.path for source in reversed(collected) if source.kind == "dotenv" and source.found
    )
