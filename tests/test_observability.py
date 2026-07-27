"""Hermetic unit tests for the tracing adapters.

``LangfuseTracer`` wraps a Langfuse client but is constructed with an injected client, so we drive
it with an in-memory stub — no ``langfuse`` install, no network. The contract under test is that
tracing is *fire-and-forget*: ``observe`` and ``flush`` must never raise into the request path,
even when the underlying client misbehaves. ``create`` is checked against a fake ``langfuse``
module (success) and against the real (absent) import (failure -> ``None``).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from mom.adapters.observability import LangfuseTracer, NoopTracer
from mom.domain.results import Usage


# --------------------------------------------------------------------------------------------
# Stub Langfuse client.
# --------------------------------------------------------------------------------------------


class _Generation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.ended = False
        self.start_calls: list[dict[str, Any]] = []
        self.children: list[_Generation] = []

    def start_observation(self, **kwargs: Any) -> _Generation:
        self.start_calls.append(kwargs)
        child = _Generation()
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class _StubClient:
    def __init__(
        self,
        *,
        trace_id: str = "trace-123",
        fail_trace_id: bool = False,
        fail_start: bool = False,
        fail_flush: bool = False,
    ) -> None:
        self.generation = _Generation()
        self.start_calls: list[dict[str, Any]] = []
        self.flush_count = 0
        self._trace_id = trace_id
        self._fail_trace_id = fail_trace_id
        self._fail_start = fail_start
        self._fail_flush = fail_flush

    def create_trace_id(self, *, seed: str) -> str:
        if self._fail_trace_id:
            raise RuntimeError("no trace id for you")
        return self._trace_id

    def start_observation(self, **kwargs: Any) -> _Generation:
        self.start_calls.append(kwargs)
        if self._fail_start:
            raise RuntimeError("start_observation blew up")
        return self.generation

    def flush(self) -> None:
        if self._fail_flush:
            raise RuntimeError("flush blew up")
        self.flush_count += 1


def _observe(tracer: LangfuseTracer, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "request_id": "req-1",
        "ensemble": "panel",
        "role": "fanout",
        "llm": "member-a",
        "model": "openai/gpt-x",
        "messages": [{"role": "user", "content": "hi"}],
        "output": "the answer",
        "usage": Usage(prompt_tokens=10, completion_tokens=5, cached_prompt_tokens=3),
        "duration_ms": 42.0,
    }
    kwargs.update(overrides)
    tracer.observe(**kwargs)


# --------------------------------------------------------------------------------------------
# NoopTracer.
# --------------------------------------------------------------------------------------------


def test_noop_tracer_is_inert() -> None:
    tracer = NoopTracer()
    assert tracer.observe(request_id="x", output="y") is None
    assert tracer.flush() is None


# --------------------------------------------------------------------------------------------
# LangfuseTracer.observe / flush — the fire-and-forget contract.
# --------------------------------------------------------------------------------------------


def test_observe_records_generation_with_usage_details() -> None:
    client = _StubClient()
    tracer = LangfuseTracer(client)
    _observe(tracer)  # role="fanout"

    # the client creates ONE root span named after the ensemble (this becomes the trace name)
    (root_start,) = client.start_calls
    assert root_start["as_type"] == "span"
    assert root_start["name"] == "panel"
    assert root_start["trace_context"] == {"trace_id": "trace-123"}

    # the generation is nested UNDER the root span
    root = client.generation
    (gen_start,) = root.start_calls
    assert gen_start["as_type"] == "generation"
    assert gen_start["name"] == "fanout:member-a"
    assert gen_start["model"] == "openai/gpt-x"
    assert gen_start["metadata"]["ensemble"] == "panel"
    assert gen_start["metadata"]["cached"] is False

    gen = root.children[0]
    assert gen.ended is True
    output_update = gen.updates[0]
    assert output_update["output"] == "the answer"
    assert output_update["usage_details"] == {"input": 10, "output": 5, "cache_read": 3}
    assert root.ended is False  # a fan-out call does NOT close the ensemble root


def test_synthesis_closes_ensemble_root_span() -> None:
    client = _StubClient()
    tracer = LangfuseTracer(client)
    _observe(tracer, role="synthesis", llm="concluder")

    root = client.generation
    assert client.start_calls[0]["name"] == "panel"  # trace named after the ensemble, not the synth
    assert root.start_calls[0]["name"] == "synthesis:concluder"  # synth generation nests under it
    assert root.ended is True  # the concluding call closes the root
    assert root.updates[-1]["output"] == "the answer"  # root output = the final answer


def test_observe_marks_error_level_when_error_present() -> None:
    client = _StubClient()
    tracer = LangfuseTracer(client)
    _observe(tracer, error="boom", cached=True)

    gen = client.generation.children[0]
    assert client.generation.start_calls[0]["metadata"]["cached"] is True
    error_update = next(u for u in gen.updates if u.get("level") == "ERROR")
    assert error_update["status_message"] == "boom"
    assert gen.ended is True


def test_observe_never_raises_when_client_fails() -> None:
    tracer = LangfuseTracer(_StubClient(fail_start=True))
    # Must be swallowed — tracing cannot break the request path.
    _observe(tracer)


def test_flush_delegates_to_client() -> None:
    client = _StubClient()
    LangfuseTracer(client).flush()
    assert client.flush_count == 1


def test_flush_never_raises_when_client_fails() -> None:
    LangfuseTracer(_StubClient(fail_flush=True)).flush()  # swallowed, no exception


# --------------------------------------------------------------------------------------------
# _trace_id: deterministic seed, with a hash fallback when the client can't mint one.
# --------------------------------------------------------------------------------------------


def test_trace_id_uses_client_seed() -> None:
    tracer = LangfuseTracer(_StubClient(trace_id="seeded-id"))
    assert tracer._trace_id("req-1") == "seeded-id"


def test_trace_id_falls_back_to_sha256() -> None:
    tracer = LangfuseTracer(_StubClient(fail_trace_id=True))
    fallback = tracer._trace_id("req-1")
    assert len(fallback) == 32
    assert all(ch in "0123456789abcdef" for ch in fallback)
    # Deterministic in the request id.
    assert fallback == tracer._trace_id("req-1")


# --------------------------------------------------------------------------------------------
# LangfuseTracer.create: fake langfuse (success) vs absent/broken import (None).
# --------------------------------------------------------------------------------------------


def test_create_returns_tracer_with_fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeLangfuse:
        def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
            captured.update(public_key=public_key, secret_key=secret_key, host=host)

    module = types.ModuleType("langfuse")
    module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    tracer = LangfuseTracer.create(public_key="pk", secret_key="sk", host="https://lf.local")

    assert isinstance(tracer, LangfuseTracer)
    assert captured == {"public_key": "pk", "secret_key": "sk", "host": "https://lf.local"}


def test_create_returns_none_when_client_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadLangfuse:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("bad credentials")

    module = types.ModuleType("langfuse")
    module.Langfuse = _BadLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    assert LangfuseTracer.create(public_key="pk", secret_key="sk", host="h") is None


def test_create_returns_none_when_langfuse_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the import to fail even if langfuse becomes installed later.
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert LangfuseTracer.create(public_key="pk", secret_key="sk", host="h") is None
