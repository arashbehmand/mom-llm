"""The composition container: everything the app needs, built once at startup.

This lives in ``runtime`` (the composition layer) rather than ``api`` so the wiring/composition
root can construct it without ``runtime`` importing ``api`` — which the layered-architecture
import contract forbids. The FastAPI-specific accessors that read it off ``app.state`` stay in
``mom.api.deps``.
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
