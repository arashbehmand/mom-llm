"""Typed config primitives: durations, byte sizes, and the effort ladder.

These give the YAML a human-friendly surface (``20m``, ``1GB``, ``high``) while resolving to
ordinary typed Python values, so the rest of the codebase never parses strings.
"""

from __future__ import annotations

from datetime import timedelta
from enum import IntEnum
import re
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer


# ----------------------------------------------------------------------------------------------
# Duration: "500ms" | "2s" | "20m" | "1h" | "14d"
# ----------------------------------------------------------------------------------------------
_DURATION_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h|d)\s*$")
_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_duration(value: object) -> timedelta:
    """Parse a duration string (or pass through a timedelta/number-of-seconds)."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    if not isinstance(value, str):
        raise TypeError(f"duration must be a string like '20m', got {type(value).__name__}")
    match = _DURATION_RE.match(value)
    if not match:
        raise ValueError(f"invalid duration {value!r}; use forms like '500ms', '2s', '20m', '14d'")
    return timedelta(seconds=float(match["value"]) * _DURATION_UNITS[match["unit"]])


def _format_duration(value: timedelta) -> str:
    seconds = value.total_seconds()
    for unit, factor in (("d", 86400.0), ("h", 3600.0), ("m", 60.0), ("s", 1.0)):
        if seconds and seconds % factor == 0:
            return f"{int(seconds // factor)}{unit}"
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    return f"{seconds:g}s"


Duration = Annotated[
    timedelta,
    BeforeValidator(parse_duration),
    PlainSerializer(_format_duration, return_type=str),
]

# ----------------------------------------------------------------------------------------------
# Byte size: "512KB" | "1GB" (base-1024)
# ----------------------------------------------------------------------------------------------
_SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB|TB)\s*$", re.IGNORECASE)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def parse_bytes(value: object) -> int:
    """Parse a byte-size string (or pass through an int)."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"size must be a string like '1GB', got {type(value).__name__}")
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid size {value!r}; use forms like '512KB', '1GB'")
    return int(float(match["value"]) * _SIZE_UNITS[match["unit"].upper()])


def _format_bytes(value: int) -> str:
    for unit, factor in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if value and value % factor == 0:
            return f"{value // factor}{unit}"
    return f"{value}B"


ByteSize = Annotated[
    int,
    BeforeValidator(parse_bytes),
    PlainSerializer(_format_bytes, return_type=str),
]


# ----------------------------------------------------------------------------------------------
# Effort ladder
# ----------------------------------------------------------------------------------------------
class EffortLevel(IntEnum):
    """Ordered reasoning-effort levels. Integer values give a total order for nearest-tier math."""

    NONE = 0
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    XHIGH = 5
    MAX = 6

    @property
    def label(self) -> str:
        return self.name.lower()


# Canonical names + terse aliases accepted in effort cells / tier lists.
_EFFORT_ALIASES: dict[str, EffortLevel] = {
    "none": EffortLevel.NONE,
    "minimal": EffortLevel.MINIMAL,
    "min": EffortLevel.MINIMAL,
    "low": EffortLevel.LOW,
    "l": EffortLevel.LOW,
    "medium": EffortLevel.MEDIUM,
    "med": EffortLevel.MEDIUM,
    "m": EffortLevel.MEDIUM,
    "high": EffortLevel.HIGH,
    "h": EffortLevel.HIGH,
    "xhigh": EffortLevel.XHIGH,
    "xh": EffortLevel.XHIGH,
    "max": EffortLevel.MAX,
}

# Sentinels usable in a per-member effort cell (not a level).
EFFORT_PASSTHROUGH = "pass"  # relay the client's requested effort to this member
EFFORT_OFF = "off"  # send no reasoning param (model default / non-reasoning)
EFFORT_SKIP = "skip"  # exclude this member at this tier
_EFFORT_SENTINELS = frozenset({EFFORT_PASSTHROUGH, EFFORT_OFF, EFFORT_SKIP})


def parse_effort_level(value: object) -> EffortLevel:
    """Parse a single effort level (canonical name or terse alias)."""
    if isinstance(value, EffortLevel):
        return value
    if isinstance(value, str):
        level = _EFFORT_ALIASES.get(value.strip().lower())
        if level is not None:
            return level
    raise ValueError(
        f"invalid effort level {value!r}; expected one of "
        f"{', '.join(sorted({level.label for level in EffortLevel}))}"
    )


def normalize_effort_cell(value: object) -> str:
    """Normalize a per-member effort cell to a canonical token.

    Returns a level label (``"high"``) or a sentinel (``"pass"``/``"off"``/``"skip"``).
    """
    if isinstance(value, EffortLevel):
        return value.label
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _EFFORT_SENTINELS:
            return token
        if token in _EFFORT_ALIASES:
            return _EFFORT_ALIASES[token].label
    raise ValueError(
        f"invalid effort cell {value!r}; expected a level (none/minimal/low/medium/high/xhigh/max, "
        f"or l/m/h/xh) or a sentinel (pass/off/skip)"
    )


def is_effort_sentinel(token: str) -> bool:
    return token in _EFFORT_SENTINELS


def nearest_tier(requested: EffortLevel, available: list[EffortLevel]) -> EffortLevel:
    """Return the defined tier nearest to ``requested`` by ordinal distance, ties rounding up."""
    if not available:
        raise ValueError("no tiers defined")
    if requested in available:
        return requested
    # Ties round up: prefer the higher tier when two are equidistant (quality bias).
    return min(available, key=lambda tier: (abs(tier - requested), -tier))
