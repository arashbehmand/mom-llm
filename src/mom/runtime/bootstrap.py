"""The one place config and secrets are resolved, for every entry point.

``mom serve``, ``mom mcp``, ``mom config``, ``mom cache`` and ``mom metrics`` all start here, so
they cannot disagree about which files are in play. Before this existed there were four
independent resolution sites with three different policies, two of which silently dropped
``MOM_CONFIG_OVERLAY``.

**``MOM_CONFIG`` is read from the process environment and the working directory's ``.env`` only,
never from a discovered one.** That rule is what breaks the obvious circularity — discovery needs
to know whether it is pinned, and a pin that could arrive from a file discovery has not located
yet is not resolvable. Stating it makes ``settings.config_file`` and ``sources.files`` agree by
construction; the alternative (resolve, then re-read settings, then notice the pin changed) would
leave the two describing different files, which logs and ``/health`` would then report as fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path

from mom.config.loader import load_layered
from mom.config.resolve import ConfigError, ResolvedCatalog
from mom.runtime.discovery import ConfigSources, discover
from mom.runtime.secrets import SecretSource, collect_secrets, dotenv_files, resolve_secrets
from mom.runtime.settings import Settings


@dataclass(frozen=True, slots=True)
class Bootstrapped:
    """Everything an entry point needs, resolved once."""

    settings: Settings
    sources: ConfigSources
    secrets: tuple[SecretSource, ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        """Operator-facing remarks, for the caller to log *after* logging is configured."""
        return tuple(self.sources.notes) + tuple(
            f"{s.path}: {s.warning}" for s in self.secrets if s.warning
        )

    def catalog(self) -> ResolvedCatalog:
        """Merge and validate the resolved stack. Raises ``ConfigError`` when there is nothing."""
        if not self.sources.files:
            raise ConfigError(_nothing_found(self.sources))
        return load_layered(self.sources.files)


def _nothing_found(sources: ConfigSources) -> str:
    """A miss should say where it looked — the whole point of a search path is that the answer
    is no longer obvious from one env var."""
    lines = [f"  {probe.path}" for probe in sources.checked]
    return (
        "no config found. Checked:\n"
        + "\n".join(lines or ["  (nothing — no home directory and no ./mom.yaml)"])
        + "\n\nCreate ~/.mom/config.yaml or ./mom.yaml, or pass --config / set MOM_CONFIG."
        + "\nRun `mom config where` for the full search path."
    )


def _home() -> Path | None:
    """The user's home, or ``None`` where there is not one.

    ``Path.home()`` raises on a container run as a uid with no passwd entry and no ``$HOME`` —
    OpenShift's default, and the image here runs ``mom serve`` directly. Skipping the user level
    is the right answer there; a pinned or project-level config still serves.
    """
    try:
        return Path.home()
    except RuntimeError:
        return None


def _path_from_env(env: Mapping[str, str], *names: str) -> Path | None:
    for name in names:
        value = env.get(name)
        if value:
            return Path(value)
    return None


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def bootstrap(
    *,
    config: Path | None = None,
    overlay: Path | None = None,
    data_dir: Path | None = None,
    auth_from_opencode: bool = False,
    cwd: Path | None = None,
    apply: bool = True,
) -> Bootstrapped:
    """Resolve the search path, materialize secrets, and build ``Settings`` — exactly once.

    Ordering, and why each step has to be where it is:

    1. Read (without applying) the working directory's ``.env``.
    2. Take ``MOM_CONFIG`` / ``MOM_CONFIG_OVERLAY`` / ``MOM_AUTH_FROM_OPENCODE`` from the process
       environment layered over that; explicit arguments win over both.
    3. Discover, using the pin from step 2.
    4. Publish every level's secrets, cwd included, in one pass so precedence is decided once.
    5. Build ``Settings`` once, pointed at every discovered ``.env`` so ``MOM_*`` reaches it
       without going through ``os.environ``.

    ``apply=False`` computes the identical result without touching ``os.environ`` — that is what
    ``mom config where`` reports, so asking where a key would come from never changes where it
    comes from.
    """
    here = Path.cwd() if cwd is None else cwd
    home = _home()

    # Step 1 + 2: read — do not yet apply — the cwd .env. It is the only file allowed to
    # influence where discovery looks, and applying it here would double-count it: cwd is always
    # on the secrets search path below, and a name already in os.environ would then be reported
    # as contributed by nothing.
    env: dict[str, str] = dict(os.environ)
    for source in collect_secrets(ConfigSources(pinned=False, secret_dirs=(here,))):
        if source.kind == "dotenv":
            env = {**source.values, **env}  # process env still wins

    pinned = config or _path_from_env(env, "MOM_CONFIG", "MOM_CONFIG_PATH")
    resolved_overlay = overlay or _path_from_env(env, "MOM_CONFIG_OVERLAY")
    use_opencode = auth_from_opencode or _truthy(env.get("MOM_AUTH_FROM_OPENCODE"))

    # Step 3.
    sources = discover(
        config=pinned,
        overlay=resolved_overlay,
        cwd=here,
        home=home,
        xdg_config_home=_path_from_env(env, "XDG_CONFIG_HOME"),
    )

    # Step 4.
    secrets = resolve_secrets(
        collect_secrets(
            sources,
            auth_from_opencode=use_opencode,
            home=home,
            xdg_data_home=_path_from_env(env, "XDG_DATA_HOME"),
        ),
        apply=apply,
    )

    # Step 5.
    settings = _settings(
        dotenv_files(secrets),
        config=pinned,
        overlay=resolved_overlay,
        data_dir=data_dir,
        auth_from_opencode=use_opencode,
    )
    return Bootstrapped(settings=settings, sources=sources, secrets=secrets)


def _settings(
    env_files: Sequence[Path],
    *,
    config: Path | None,
    overlay: Path | None,
    data_dir: Path | None,
    auth_from_opencode: bool,
) -> Settings:
    """Build ``Settings`` with the already-resolved paths applied.

    ``model_copy`` rather than keyword construction for the path fields: they carry validation
    aliases (``MOM_CONFIG`` / ``MOM_CONFIG_PATH``), so passing them by field name would silently
    not bind. Values go in as ``Path``, not ``str`` — the field is typed ``Path | None`` and
    ``model_copy`` does not validate, so a string here would be a type the rest of the code has
    to keep defensively re-wrapping.
    """
    settings = Settings(_env_file=tuple(env_files) or None)
    overrides: dict[str, object] = {}
    if config is not None:
        overrides["config_file"] = Path(config)
    if overlay is not None:
        overrides["config_overlay"] = Path(overlay)
    if data_dir is not None:
        overrides["data_dir"] = Path(data_dir)
    if auth_from_opencode:
        # One-directional: an omitted flag must not clobber MOM_AUTH_FROM_OPENCODE.
        overrides["auth_from_opencode"] = True
    return settings.model_copy(update=overrides) if overrides else settings
