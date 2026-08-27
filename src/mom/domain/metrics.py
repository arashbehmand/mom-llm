"""The metric record type (pure) and the MetricsSink port.

Lives in the domain so the engine can build and emit metrics without importing the SQLite store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from mom.domain.errors import ErrorKind
from mom.domain.results import OutcomeStatus


Role = Literal["fanout", "synthesis"]
TurnType = Literal["ensemble", "relay"]


@dataclass(frozen=True, slots=True)
class CallMetric:
    request_id: str
    ts: float
    ensemble: str
    llm: str
    model: str | None
    role: Role
    # The full ModelOutcome status vocabulary (schema v2) — previously collapsed to 'ok'/'error'
    # at the call site, which hid 'empty'/'timeout'/'aborted' calls inside a single opaque bucket.
    status: OutcomeStatus
    cache_hit: bool = False
    turn_type: TurnType = "ensemble"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: float | None = None
    error: str | None = None
    # Appended last (schema v2) — MUST stay last: MetricsStore.insert_many does astuple(m) against
    # a hand-written column list in the same order, so a field inserted out of order between
    # same-typed columns would silently swap data rather than error.
    finish_reason: str | None = None
    error_kind: ErrorKind | None = None
    error_detail: str | None = None
    attempts: int = 1


class MetricsSink(Protocol):
    def record(self, metric: CallMetric) -> None:
        """Enqueue a metric off the hot path (never awaits)."""
        ...


class MetricsReader(Protocol):
    async def aggregate(
        self, *, start: float | None = None, end: float | None = None, ensemble: str | None = None
    ) -> dict[str, object]:
        """Aggregate usage/cost over an optional time/ensemble window."""
        ...

    async def aggregate_by(
        self,
        dimension: str,
        *,
        start: float | None = None,
        end: float | None = None,
        ensemble: str | None = None,
    ) -> list[dict[str, object]]:
        """Aggregate usage/cost grouped by a dimension (member / turn_type / day)."""
        ...

    async def estimated_cache_savings(
        self, *, start: float | None = None, end: float | None = None, ensemble: str | None = None
    ) -> float:
        """What the cache hits in the window would have cost at their llm's average price."""
        ...

    async def recent_runs(self, *, limit: int = 20) -> list[dict[str, object]]:
        """The most recently active runs, one row each (newest first)."""
        ...

    async def run_calls(self, request_id: str) -> list[dict[str, object]]:
        """Every recorded call of one run, oldest first (members then synthesis)."""
        ...
