"""The composition container — everything the app needs, built once at startup.

It lives in the runtime layer (not ``api``) so the composition root (``wiring``) can build it
without the runtime importing the api layer. ``api.deps`` re-exports it and adds the FastAPI
dependency that reads it from ``app.state``.
"""

from __future__ import annotations

from dataclasses import dataclass

from mom.config.resolve import ResolvedCatalog
from mom.domain.metrics import MetricsReader, MetricsSink
from mom.domain.ports import Clock, IdFactory, LLMClient, Tracer
from mom.runtime.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    """Everything the app needs, constructed once at startup (or injected in tests)."""

    settings: Settings
    catalog: ResolvedCatalog
    client: LLMClient
    clock: Clock
    ids: IdFactory
    metrics: MetricsSink | None = None
    metrics_reader: MetricsReader | None = None
    tracer: Tracer | None = None
