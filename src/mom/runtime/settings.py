"""Process settings: environment and secrets only (never the YAML model config).

``Settings`` is the single typed source for machine-local facts and secrets. It reads ``MOM_``
prefixed env vars, and — via :class:`~pydantic.AliasChoices` — also accepts the legacy v1 names
(``API_TOKEN``, ``REDIS_URL``, ``MOM_CONFIG_PATH``, ``LITELLM_VERBOSE``) so existing deployments
keep working; the ``MOM_`` name wins when both are set.

There is no static ``env_file`` here. A bare ``".env"`` resolves against the working directory at
construction time, which is exactly the cwd-dependence config discovery exists to remove — and it
is invisible to the search path, so a ``~/.mom/.env`` would never be seen. Instead
:func:`mom.runtime.bootstrap.bootstrap` passes the discovered dotenv files as ``_env_file``
(``pydantic-settings`` layers a sequence, later files overriding earlier, with the real process
environment still outranking all of them). That is also what keeps ``MOM_API_TOKEN`` out of
``os.environ`` — see :mod:`mom.runtime.secrets`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Machine-local configuration resolved from the environment (and ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="MOM_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- auth (secret) ---
    api_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MOM_API_TOKEN", "API_TOKEN"),
    )

    # --- paths ---
    config_file: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("MOM_CONFIG", "MOM_CONFIG_PATH"),
    )
    # Deep-merged over config_file (e.g. server.public_url, or any deployment-local value that
    # shouldn't sit in the tracked config) — the same layering `mom config validate --overlay`
    # already offered, now also available to `mom serve`.
    config_overlay: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("MOM_CONFIG_OVERLAY"),
    )
    data_dir: Path | None = None  # None -> platform default at resolution time

    # --- optional services ---
    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MOM_REDIS_URL", "REDIS_URL"),
    )

    # --- logging ---
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    litellm_debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("MOM_LITELLM_DEBUG", "LITELLM_VERBOSE"),
    )

    # --- credential bridges ---
    # Both a flag and an env var, because `mom serve --reload` reaches its uvicorn child only
    # through the environment: the child re-imports the app factory and resolves everything
    # itself, so a value that lives solely in the parent's argv would be lost.
    auth_from_opencode: bool = Field(
        default=False,
        validation_alias=AliasChoices("MOM_AUTH_FROM_OPENCODE"),
    )


def settings_env_names() -> frozenset[str]:
    """Every environment variable name that configures mom itself.

    Derived from the model rather than hand-listed, so it cannot drift when a field or a legacy
    alias is added. It covers both spellings of the aliased fields — ``MOM_API_TOKEN`` *and*
    ``API_TOKEN`` — which matters because the legacy names carry exactly the same secrets as the
    prefixed ones and must be kept out of the process environment for the same reason.
    """
    names: set[str] = set()
    prefix = Settings.model_config.get("env_prefix", "")
    for field_name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            names.update(str(choice) for choice in alias.choices)
        else:
            names.add(f"{prefix}{field_name}".upper())
    return frozenset(names)
