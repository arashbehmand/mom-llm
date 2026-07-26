"""Progress events: a lightweight, JSON-serializable view of a request's lifecycle.

These are deliberately distinct from the engine's rich ``StreamEvent`` union. A progress event is
a coarse milestone — fan-out started, each member completed, synthesis started, completed/failed —
meant to be published on the :class:`~mom.domain.ports.EventBus` and streamed to a
``GET /v1/progress/{id}`` observer. It carries only small, scalar fields so a Redis-backed bus can
ship it across processes as one JSON line.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


ProgressKind = Literal[
    "fanout_started",
    "member_completed",
    "synthesis_started",
    "completed",
    "failed",
]

# The two milestones after which no further events arrive for a request; a subscriber closes.
_TERMINAL_KINDS: frozenset[str] = frozenset({"completed", "failed"})

# Optional fields, in wire order. Kept as a constant so serialization stays declarative.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "member",
    "model",
    "status",
    "detail",
    "members_total",
    "completed",
    "duration_ms",
)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One coarse milestone in a request's lifecycle (bus- and SSE-serializable)."""

    kind: ProgressKind
    ensemble: str = ""
    member: str | None = None  # member/synthesizer identity
    model: str | None = None  # provider model id
    status: str | None = None  # member outcome status, or finish_reason on completion
    detail: str | None = None  # human-readable message (failures)
    members_total: int | None = None  # members fanned out (fanout_started / member_completed)
    completed: int | None = None  # members completed so far (member_completed)
    duration_ms: float | None = None

    @property
    def terminal(self) -> bool:
        """Whether this is the last event a subscriber will see for the request."""
        return self.kind in _TERMINAL_KINDS

    def to_dict(self) -> dict[str, Any]:
        """A compact dict dropping unset optionals (``kind``/``ensemble`` are always present)."""
        data: dict[str, Any] = {"kind": self.kind, "ensemble": self.ensemble}
        for name in _OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressEvent:
        return cls(
            kind=data["kind"],
            ensemble=data.get("ensemble", ""),
            member=data.get("member"),
            model=data.get("model"),
            status=data.get("status"),
            detail=data.get("detail"),
            members_total=data.get("members_total"),
            completed=data.get("completed"),
            duration_ms=data.get("duration_ms"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> ProgressEvent:
        return cls.from_dict(json.loads(raw))
