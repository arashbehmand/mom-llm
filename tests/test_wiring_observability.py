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
    bus, _close = wiring.build_bus(Settings(_env_file=None), _catalog())
    assert isinstance(bus, InMemoryEventBus)


async def test_build_bus_is_redis_when_url_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOM_REDIS_URL", "redis://localhost:6379/0")
    bus, close = wiring.build_bus(Settings(_env_file=None), _catalog())
    assert isinstance(bus, RedisEventBus)
    await close()  # tears down the (unconnected) client's pool


async def test_build_bus_falls_back_to_memory_when_redis_init_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def _boom(_url: str) -> RedisEventBus:
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(RedisEventBus, "from_url", staticmethod(_boom))
    monkeypatch.setenv("MOM_REDIS_URL", "redis://localhost:6379/0")
    bus, _close = wiring.build_bus(Settings(_env_file=None), _catalog())
    assert isinstance(bus, InMemoryEventBus)


def test_build_bus_ttl_defaults_to_call_timeout_plus_replay_window():
    # defaults.call.timeout defaults to 20m = 1200s; no fanout.deadline, no per-llm override.
    bus, _close = wiring.build_bus(Settings(_env_file=None), _catalog())
    assert bus._ttl == 1200.0 + 300.0


def test_build_bus_ttl_tracks_a_configured_fanout_deadline_past_call_timeout():
    catalog = _catalog("defaults: { call: { timeout: 5m }, fanout: { deadline: 10m } }")
    bus, _close = wiring.build_bus(Settings(_env_file=None), catalog)
    assert bus._ttl == 600.0 + 300.0  # the deadline (10m), not the shorter call timeout (5m)


def test_build_bus_ttl_tracks_the_longest_per_llm_timeout_override():
    text = dedent("""
        version: 2
        defaults:
          call: { timeout: 5m }
        llms:
          a: { model: openai/a, timeout: 45m }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: a }
    """)
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(text)))
    bus, _close = wiring.build_bus(Settings(_env_file=None), catalog)
    assert bus._ttl == 2700.0 + 300.0  # 45m override beats the 5m default


def _catalog_with_server(server: str) -> object:
    text = dedent(f"""
        version: 2
        server: {server}
        llms:
          a: {{ model: openai/a }}
        ensembles:
          e:
            members: [{{ llm: a }}]
            synthesizer: {{ llm: a }}
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def test_build_coalesce_is_none_by_default():
    assert wiring.build_coalesce(_catalog()) is None


def test_build_coalesce_builds_a_registry_when_enabled():
    catalog = _catalog_with_server("{ dedupe: { enabled: true } }")
    registry = wiring.build_coalesce(catalog)
    assert registry is not None
    assert registry._grace_seconds == 90.0
    assert registry._max_buffer_chars == 8 * 1024 * 1024


def test_build_coalesce_honors_configured_grace_and_buffer():
    catalog = _catalog_with_server(
        "{ dedupe: { enabled: true, orphan_grace: 30s, max_buffer: 1MB } }"
    )
    registry = wiring.build_coalesce(catalog)
    assert registry is not None
    assert registry._grace_seconds == 30.0
    assert registry._max_buffer_chars == 1024 * 1024
