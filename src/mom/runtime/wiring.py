"""Composition root: build the container (and its async cleanup) from settings + config."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from pathlib import Path

import platformdirs

from mom.adapters.caching import CachingClient
from mom.adapters.eventbus import ClosableBus, InMemoryEventBus, RedisEventBus, RunIndexBus
from mom.adapters.litellm_client import (
    LiteLLMClient,
    LiteLLMTokenEstimator,
    uncatalogued_models,
)
from mom.adapters.observability import (
    CompositeTracer,
    LangfuseTracer,
    NoopTracer,
    OtelTracer,
)
from mom.config.loader import load_config
from mom.config.resolve import ResolvedCatalog
from mom.domain.ports import CacheStore, EventBus, LLMClient, Tracer
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
    """Build the progress event bus (Redis when ``redis_url`` is set, else in-memory).

    Whichever is chosen is wrapped in a ``RunIndexBus`` so "what is running right now" is
    answerable on both — the Redis bus is pure pub/sub with nothing to enumerate, and the
    in-memory one only retains history for ids the caller already knows.
    """
    ttl = _bus_ttl_seconds(catalog)
    inner: ClosableBus
    if settings.redis_url:
        try:
            inner = RedisEventBus.from_url(settings.redis_url)
        except Exception:
            logger.warning("redis event bus init failed; using in-memory", exc_info=True)
            inner = InMemoryEventBus(ttl_seconds=ttl)
    else:
        inner = InMemoryEventBus(ttl_seconds=ttl)
    bus = RunIndexBus(inner, ttl_seconds=ttl)
    return bus, bus.aclose


def build_coalesce(catalog: ResolvedCatalog) -> CoalesceRegistry:
    """Build the in-flight request coalescing registry.

    Always built, even when ``server.dedupe.enabled`` is false — the config flag is now the
    *default* policy rather than the on/off switch, since a `<<SYSTEM>> dedupe: on` directive
    has to be able to coalesce one turn on a deployment that leaves it off. An unused registry
    is an empty dict and a lock: a disabled feature still costs nothing, because whether a given
    run attaches is decided per request by ``plan.dedupe``.
    """
    dedupe = catalog.config.server.dedupe
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


def _warn_stale_model_catalog(catalog: ResolvedCatalog) -> None:
    """Report configured models the pinned litellm has no catalog entry for.

    Reported for every provider, because the gap costs something everywhere (see
    ``uncatalogued_models``) — but at two volumes, because only one of the two is knowable here.

    Anthropic's missing entry is *predictably* fatal: litellm substitutes a 4096-token cap and a
    thinking model returns empty. That is a warning, before a single call is made.

    Everywhere else the consequence is a $0 price, and whether it actually lands depends on
    something startup cannot see — whether the provider reports its own cost. OpenRouter does,
    which is why 26 of this config's 29 uncatalogued models are genuinely fine; Gemini and xAI
    do not, which is why three of them had billed to $0.00 across hundreds of live calls. So the
    prediction goes out at info, and ``_warn_once_if_free`` raises the warning from the observed
    call, naming the models that really are free rather than the ones that might be. A startup
    list of 29 names, mostly healthy, teaches the operator to skip the line — which is how the
    Opus 5 breakage stayed invisible for a day in the first place.

    Never a startup failure: one uncatalogued model must not take the gateway down, and the rest
    of the panel is unaffected. Config ``pricing:`` covers the cost half for a model; the sizing
    half needs a newer litellm.
    """
    gaps = uncatalogued_models(llm.model for llm in catalog.llms.values())
    if not gaps:
        return
    capped = gaps.pop("anthropic", [])
    if capped:
        logger.warning(
            "anthropic models missing from litellm's model catalog: their answers will be "
            "capped at litellm's 4096-token default (thinking models return empty), they may "
            "forward sampling params the model rejects, and they price at $0 "
            "— raise the litellm floor in pyproject.toml",
            models=capped,
        )
    unpriced = {
        provider: models
        for provider, models in gaps.items()
        if not all(_has_declared_pricing(catalog, model) for model in models)
    }
    if unpriced:
        logger.info(
            "models missing from litellm's model catalog: they price at $0 unless the provider "
            "reports a cost of its own (OpenRouter does), and litellm cannot tell whether they "
            "still accept sampling params. Any that really do go free are named in a warning on "
            "their first call — declare `pricing:` in config or raise the litellm floor",
            by_provider=unpriced,
        )


def _has_declared_pricing(catalog: ResolvedCatalog, model: str) -> bool:
    """Whether config prices this model itself, which is the supported way to cover a model
    litellm's catalog does not carry (``compute_cost`` prefers it over litellm's map anyway)."""
    return any(llm.pricing is not None for llm in catalog.llms.values() if llm.model == model)


async def build_container(settings: Settings) -> tuple[Container, Callable[[], Awaitable[None]]]:
    """Load config, open stores, wire adapters. Returns the container and an async cleanup."""
    if settings.config_file is None:
        raise RuntimeError("MOM_CONFIG must point to a config file to serve")
    catalog = load_config(settings.config_file, overlay=settings.config_overlay)
    _warn_stale_model_catalog(catalog)
    clock = SystemClock()
    data_dir = resolve_data_dir(settings, catalog)

    closers: list[Callable[[], Awaitable[None]]] = []
    client: LLMClient = LiteLLMClient()
    cache_store: CacheStore | None = None
    if catalog.config.cache.enabled:
        cache = await SqliteCacheStore.open(
            data_dir / "cache.db",
            ttl_seconds=catalog.config.cache.ttl.total_seconds(),
            max_bytes=catalog.config.cache.max_size,
        )
        client = CachingClient(client, cache, clock, coalesce=catalog.config.cache.coalesce)
        cache_store = cache
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
        cache_store=cache_store,
    )

    async def cleanup() -> None:
        tracer.flush()
        for close in closers:
            await close()

    return container, cleanup
