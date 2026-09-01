"""The composition container — everything the app needs, built once at startup.

It lives in the runtime layer (not ``api``) so the composition root (``wiring``) can build it
without the runtime importing the api layer. ``api.deps`` re-exports it and adds the FastAPI
dependency that reads it from ``app.state``.
"""

from __future__ import annotations

from dataclasses import dataclass

from mom.config.resolve import ResolvedCatalog
from mom.domain.metrics import MetricsReader, MetricsSink
from mom.domain.ports import (
    CacheStore,
    Clock,
    EventBus,
    IdFactory,
    LLMClient,
    TokenEstimator,
    ToolCallCustody,
    Tracer,
)
from mom.engine.coalesce import CoalesceRegistry
from mom.runtime.discovery import ConfigSources
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
    token_estimator: TokenEstimator | None = None
    # Process-local custody of provider-native tool-call ids behind minted client ids.
    custody: ToolCallCustody | None = None
    bus: EventBus | None = None
    # In-flight request coalescing (`server.dedupe`); None when disabled.
    coalesce: CoalesceRegistry | None = None
    # The response cache the client already wraps, kept here purely so a reader (`mom mcp`'s
    # cache_stats) can inspect it without opening a second connection to the same file.
    cache_store: CacheStore | None = None
    # Which config files the catalog was merged from, and what else was probed. `catalog` alone
    # cannot answer "where did this come from" once resolution is a search path, and
    # `settings.config_file` is only ever the pin — None whenever discovery did the work.
    sources: ConfigSources | None = None
