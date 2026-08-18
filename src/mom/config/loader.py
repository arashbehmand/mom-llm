"""Load and resolve configuration from YAML.

Supports config layering: a tracked, public base file (``models.yaml``) deep-merged with an
optional local override (gitignored) so the shareable example and the owner's real setup live in
separate files. Loading has no global side effects — it returns a fresh ``ResolvedCatalog``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from mom.config.resolve import ConfigError, ResolvedCatalog, resolve_catalog
from mom.config.schema import Config


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def load_raw(path: str | Path, *, overlay: str | Path | None = None) -> dict[str, Any]:
    """Read (and optionally layer) YAML into a raw dict, before schema validation."""
    raw = _read_yaml(Path(path))
    if overlay is not None:
        raw = _deep_merge(raw, _read_yaml(Path(overlay)))
    return raw


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Validate a raw mapping into a :class:`Config` (raises ``ConfigError`` on failure)."""
    try:
        return Config.model_validate(dict(raw))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


def load_config(path: str | Path, *, overlay: str | Path | None = None) -> ResolvedCatalog:
    """Load, validate, and resolve a config file into an immutable catalog."""
    return resolve_catalog(parse_config(load_raw(path, overlay=overlay)))
