"""Composition root selects the right tracer and event bus from config/settings."""

from __future__ import annotations

from textwrap import dedent

import pytest
import yaml

from mom.adapters.eventbus import InMemoryEventBus, RedisEventBus
from mom.adapters.observability import CompositeTracer, NoopTracer, OtelTracer
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime import wiring
from mom.runtime.settings import Settings
from mom.testing import RecordingTracer


def _catalog(observability: str = "") -> object:
    text = dedent(f"""
        version: 2
        {observability}
        llms:
          a: {{ model: openai/a }}
        ensembles:
          e:
            members: [{{ llm: a }}]
            synthesizer: {{ llm: a }}
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def test_build_tracer_is_noop_by_default():
    assert isinstance(wiring.build_tracer(_catalog()), NoopTracer)


def test_build_tracer_selects_otel_when_enabled():
    catalog = _catalog("observability: { otel: { enabled: true } }")
    assert isinstance(wiring.build_tracer(catalog), OtelTracer)


def test_build_tracer_composes_when_multiple_enabled(monkeypatch: pytest.MonkeyPatch):
    # Avoid constructing a real Langfuse client; just prove the composition path.
    monkeypatch.setattr(wiring, "_langfuse_tracer", lambda _catalog: RecordingTracer())
    catalog = _catalog("observability: { otel: { enabled: true } }")
    assert isinstance(wiring.build_tracer(catalog), CompositeTracer)


def test_build_bus_is_in_memory_by_default():
    bus, _close = wiring.build_bus(Settings(_env_file=None))
    assert isinstance(bus, InMemoryEventBus)


async def test_build_bus_is_redis_when_url_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOM_REDIS_URL", "redis://localhost:6379/0")
    bus, close = wiring.build_bus(Settings(_env_file=None))
    assert isinstance(bus, RedisEventBus)
    await close()  # tears down the (unconnected) client's pool


async def test_build_bus_falls_back_to_memory_when_redis_init_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def _boom(_url: str) -> RedisEventBus:
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(RedisEventBus, "from_url", staticmethod(_boom))
    monkeypatch.setenv("MOM_REDIS_URL", "redis://localhost:6379/0")
    bus, _close = wiring.build_bus(Settings(_env_file=None))
    assert isinstance(bus, InMemoryEventBus)
