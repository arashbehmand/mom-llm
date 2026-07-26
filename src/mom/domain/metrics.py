"""The metric record type (pure) and the MetricsSink port.

Lives in the domain so the engine can build and emit metrics without importing the SQLite store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


Role = Literal["fanout", "synthesis"]
Status = Literal["ok", "error"]
TurnType = Literal["ensemble", "relay"]


@dataclass(frozen=True, slots=True)
class CallMetric:
    request_id: str
    ts: float
    ensemble: str
    llm: str
    model: str | None
    role: Role
    status: Status
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
