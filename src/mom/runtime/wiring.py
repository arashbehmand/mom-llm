"""Composition root: build the container (and its async cleanup) from settings + config."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from pathlib import Path

import platformdirs

from mom.adapters.caching import CachingClient
from mom.adapters.eventbus import InMemoryEventBus, RedisEventBus
from mom.adapters.litellm_client import LiteLLMClient, LiteLLMTokenEstimator
from mom.adapters.observability import (
    CompositeTracer,
    LangfuseTracer,
    NoopTracer,
    OtelTracer,
)
from mom.config.loader import load_config
from mom.config.resolve import ResolvedCatalog
from mom.domain.ports import EventBus, LLMClient, Tracer
from mom.engine.coalesce import CoalesceRegistry
from mom.runtime.clock import SystemClock, UuidIds
from mom.runtime.container import Container
from mom.runtime.custody import InMemoryToolCallCustody
from mom.runtime.logging import get_logger
from mom.runtime.settings import Settings
from mom.store.cache import SqliteCacheStore
from mom.store.metrics import MetricsRecorder, MetricsStore


logger = get_logger("mom.wiring")


def _langfuse_tracer(catalog: ResolvedCatalog) -> Tracer | None:
    lf = catalog.config.observability.langfuse
    if not lf.enabled:
        return None
    public_key = os.getenv(lf.public_key_env)
    secret_key = os.getenv(lf.secret_key_env)
    host = os.getenv(lf.host_env) or "https://cloud.langfuse.com"
    if not (public_key and secret_key):
        return None
    return LangfuseTracer.create(public_key=public_key, secret_key=secret_key, host=host)


def _otel_tracer(catalog: ResolvedCatalog) -> Tracer | None:
    otel = catalog.config.observability.otel
    if not otel.enabled:
        return None
    return OtelTracer.create(
        endpoint=otel.endpoint, protocol=otel.protocol, service_name=otel.service_name
    )


def build_tracer(catalog: ResolvedCatalog) -> Tracer:
    """Compose the enabled tracers (Langfuse and/or OTel); ``NoopTracer`` when none are on."""
    tracers = [t for t in (_langfuse_tracer(catalog), _otel_tracer(catalog)) if t is not None]
    if not tracers:
        return NoopTracer()
    if len(tracers) == 1:
        return tracers[0]
    return CompositeTracer(tracers)


def _bus_ttl_seconds(catalog: ResolvedCatalog) -> float:
    """The in-memory bus's history/idle TTL: long enough to outlive the longest silent gap in a
    request's lifecycle.

    That gap is NOT the fan-out deadline (bounded, typically minutes) — it's
    ``synthesis_started -> completed``, during which nothing is published and which is bounded only
    by the synthesizer's call timeout (20m by default; a per-llm override can be longer still). A
    fixed default (previously a flat 300s) could be — and in production, with a 10m-deadline/20m-
    timeout config, was — shorter than a single slow call, silently evicting the channel and, by
    the bus's own lazy-sweep design, sentinel-closing any subscriber still watching mid-request.
    """
    defaults = catalog.config.defaults
    longest_timeout = defaults.call.timeout.total_seconds()
    for llm in catalog.llms.values():
        if llm.timeout is not None:
            longest_timeout = max(longest_timeout, llm.timeout.total_seconds())
    if defaults.fanout.deadline is not None:
        longest_timeout = max(longest_timeout, defaults.fanout.deadline.total_seconds())
    return longest_timeout + 300.0  # a generous replay window for a late-joining subscriber


def build_bus(
    settings: Settings, catalog: ResolvedCatalog
) -> tuple[EventBus, Callable[[], Awaitable[None]]]:
    """Build the progress event bus (Redis when ``redis_url`` is set, else in-memory)."""
    if settings.redis_url:
        try:
            redis_bus = RedisEventBus.from_url(settings.redis_url)
        except Exception:
            logger.warning("redis event bus init failed; using in-memory", exc_info=True)
        else:
            return redis_bus, redis_bus.aclose
    memory_bus = InMemoryEventBus(ttl_seconds=_bus_ttl_seconds(catalog))
    return memory_bus, memory_bus.aclose


def build_coalesce(catalog: ResolvedCatalog) -> CoalesceRegistry | None:
    """Build the in-flight request coalescing registry, or ``None`` when ``server.dedupe`` is
    off (the default) — a disabled feature must cost nothing and change nothing."""
    dedupe = catalog.config.server.dedupe
    if not dedupe.enabled:
        return None
    return CoalesceRegistry(
        orphan_grace_seconds=dedupe.orphan_grace.total_seconds(),
        max_buffer_chars=dedupe.max_buffer,
    )


def resolve_data_dir(settings: Settings, catalog: ResolvedCatalog) -> Path:
    if settings.data_dir is not None:
        return Path(settings.data_dir)
    if catalog.config.storage.data_dir is not None:
        return Path(catalog.config.storage.data_dir)
    return Path(platformdirs.user_data_dir("mom-llm"))


async def build_container(settings: Settings) -> tuple[Container, Callable[[], Awaitable[None]]]:
    """Load config, open stores, wire adapters. Returns the container and an async cleanup."""
    if settings.config_file is None:
        raise RuntimeError("MOM_CONFIG must point to a config file to serve")
    catalog = load_config(settings.config_file, overlay=settings.config_overlay)
    clock = SystemClock()
    data_dir = resolve_data_dir(settings, catalog)

    closers: list[Callable[[], Awaitable[None]]] = []
    client: LLMClient = LiteLLMClient()
    if catalog.config.cache.enabled:
        cache = await SqliteCacheStore.open(
            data_dir / "cache.db",
            ttl_seconds=catalog.config.cache.ttl.total_seconds(),
            max_bytes=catalog.config.cache.max_size,
        )
        client = CachingClient(client, cache, clock, coalesce=catalog.config.cache.coalesce)
        closers.append(cache.close)

    metrics_store = await MetricsStore.open(data_dir / "metrics.db")
    recorder = MetricsRecorder(metrics_store)
    await recorder.start()
    closers.append(recorder.stop)
    closers.append(metrics_store.close)

    tracer = build_tracer(catalog)
    bus, bus_close = build_bus(settings, catalog)
    closers.append(bus_close)

    coalesce = build_coalesce(catalog)

    container = Container(
        settings=settings,
        catalog=catalog,
        client=client,
        clock=clock,
        ids=UuidIds(),
        metrics=recorder,
        metrics_reader=metrics_store,
        tracer=tracer,
        token_estimator=LiteLLMTokenEstimator(),
        custody=InMemoryToolCallCustody(),
        bus=bus,
        coalesce=coalesce,
    )

    async def cleanup() -> None:
        tracer.flush()
        for close in closers:
            await close()

    return container, cleanup
