"""Structural dict merge shared by config layering (``loader``), ``extends``/``variants`` param
resolution (``resolve``), and multi-level discovery. A ``null`` in ``over`` deletes an inherited
key — this is what ``docs/CONFIGURATION.md`` documents for every merge site in the config pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``over`` onto ``base``. A ``null`` value in ``over`` deletes the inherited key;
    nested maps merge key-by-key; anything else in ``over`` replaces the base value outright."""
    result: dict[str, Any] = dict(base)
    for key, value in over.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
