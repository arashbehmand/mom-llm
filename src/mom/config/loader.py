"""Load and resolve configuration from YAML.

Supports config layering: an ordered stack of files deep-merged lowest-first, then validated
**once** at the end. That ordering is what lets a project-level file be nothing but
``version: 2`` plus ``ensembles:`` referencing ``llms`` defined at the user level — no single
layer has to satisfy the schema on its own, only their merge does.

The stack is built by :mod:`mom.runtime.discovery`; this module only reads and merges what it is
handed. Loading has no global side effects — it returns a fresh ``ResolvedCatalog``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from mom.config.merge import deep_merge
from mom.config.resolve import ConfigError, ResolvedCatalog, resolve_catalog
from mom.config.schema import Config


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
        raw = deep_merge(raw, _read_yaml(Path(overlay)))
    return raw


def load_layered_raw(paths: Sequence[Path]) -> dict[str, Any]:
    """Read every path and deep-merge them lowest-precedence-first, before schema validation.

    Every path must exist: discovery already dropped the candidates that did not, so a missing
    file here is a caller bug (or a file deleted underneath us), not an absent optional layer.
    """
    raw: dict[str, Any] = {}
    for path in paths:
        raw = deep_merge(raw, _read_yaml(Path(path)))
    return raw


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Validate a raw mapping into a :class:`Config` (raises ``ConfigError`` on failure)."""
    try:
        return Config.model_validate(dict(raw))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


def load_layered(paths: Sequence[Path]) -> ResolvedCatalog:
    """Load, merge, validate, and resolve an ordered stack of config files into one catalog."""
    return resolve_catalog(parse_config(load_layered_raw(paths)))


def load_config(path: str | Path, *, overlay: str | Path | None = None) -> ResolvedCatalog:
    """Load, validate, and resolve a config file into an immutable catalog."""
    return resolve_catalog(parse_config(load_raw(path, overlay=overlay)))
