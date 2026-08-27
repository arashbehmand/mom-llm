"""Progress events: a lightweight, JSON-serializable view of a request's lifecycle.

These are deliberately distinct from the engine's rich ``StreamEvent`` union. A progress event is
a coarse milestone — fan-out started, each member completed, synthesis started, completed/failed —
meant to be published on the :class:`~mom.domain.ports.EventBus` and streamed to a
``GET /v1/progress/{id}`` observer. It carries only small, bounded fields (``preview`` is
truncated at publish time — see ``PREVIEW_CHARS``) so a Redis-backed bus can ship it across
processes as one JSON line.
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

# `preview` is truncated to this many characters before publishing (see engine/pipeline.py) — a
# glimpse of the actual output, not the full text, to keep each event a small JSON line.
PREVIEW_CHARS = 280

# Optional fields, in wire order. Kept as a constant so serialization stays declarative.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "member",
    "model",
    "status",
    "detail",
    "members_total",
    "members",
    "completed",
    "duration_ms",
    "preview",
    "cost_usd",
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
    # (identity, model) for every fanned-out member, known upfront (fanout_started only) — lets an
    # observer label a still-pending member instead of showing a bare "waiting" placeholder.
    members: tuple[tuple[str, str], ...] | None = None
    completed: int | None = None  # members completed so far (member_completed)
    duration_ms: float | None = None
    preview: str | None = None  # truncated glimpse of the actual output (see PREVIEW_CHARS)
    # This member's cost (member_completed) or the run total (completed). Optional because a
    # progress event is a milestone, not an accounting record — metrics.db remains the ledger;
    # this is what makes a still-running fan-out's spend visible (MCP `runs`, progress page).
    cost_usd: float | None = None

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
        members = data.get("members")
        return cls(
            kind=data["kind"],
            ensemble=data.get("ensemble", ""),
            member=data.get("member"),
            model=data.get("model"),
            status=data.get("status"),
            detail=data.get("detail"),
            members_total=data.get("members_total"),
            members=tuple((m, mo) for m, mo in members) if members else None,
            completed=data.get("completed"),
            duration_ms=data.get("duration_ms"),
            preview=data.get("preview"),
            cost_usd=data.get("cost_usd"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> ProgressEvent:
        return cls.from_dict(json.loads(raw))
