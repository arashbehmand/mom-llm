"""Composition root: build the container (and its async cleanup) from settings + config."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from pathlib import Path

import platformdirs

from mom.adapters.caching import CachingClient
from mom.adapters.litellm_client import LiteLLMClient, LiteLLMTokenEstimator
from mom.adapters.observability import LangfuseTracer, NoopTracer
from mom.config.loader import load_config
from mom.config.resolve import ResolvedCatalog
from mom.domain.ports import LLMClient, Tracer
from mom.runtime.clock import SystemClock, UuidIds
from mom.runtime.container import Container
from mom.runtime.settings import Settings
from mom.store.cache import SqliteCacheStore
from mom.store.metrics import MetricsRecorder, MetricsStore


def build_tracer(catalog: ResolvedCatalog) -> Tracer:
    lf = catalog.config.observability.langfuse
    if not lf.enabled:
        return NoopTracer()
    public_key = os.getenv(lf.public_key_env)
    secret_key = os.getenv(lf.secret_key_env)
    host = os.getenv(lf.host_env) or "https://cloud.langfuse.com"
    if not (public_key and secret_key):
        return NoopTracer()
    return (
        LangfuseTracer.create(public_key=public_key, secret_key=secret_key, host=host)
        or NoopTracer()
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
    catalog = load_config(settings.config_file)
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
    )

    async def cleanup() -> None:
        tracer.flush()
        for close in closers:
            await close()

    return container, cleanup
