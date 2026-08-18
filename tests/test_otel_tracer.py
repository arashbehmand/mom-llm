"""OtelTracer span construction (GenAI conventions) via an in-memory exporter, plus CompositeTracer.

The OTLP network export is not exercised (it needs a live collector and sockets are disabled in the
suite); instead the tracer emits into an in-memory span exporter so span timing, name, and
attributes are asserted directly. ``OtelTracer.create`` — the real OTLP wiring — is smoke-tested to
confirm the import paths and processor attach.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mom.adapters.observability import CompositeTracer, OtelTracer
from mom.domain.results import Usage
from mom.testing import RecordingTracer


def _tracer_with_exporter() -> tuple[OtelTracer, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelTracer(provider.get_tracer("test"), provider), exporter


def _observe(tracer: OtelTracer, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "request_id": "req-1",
        "ensemble": "bmom",
        "role": "fanout",
        "llm": "a",
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "output": "hello",
        "usage": Usage(prompt_tokens=10, completion_tokens=5),
        "duration_ms": 12.0,
    }
    kwargs.update(overrides)
    tracer.observe(**kwargs)


def test_span_carries_genai_and_mom_attributes():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer)
    (span,) = exporter.get_finished_spans()
    assert span.name == "fanout a"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["mom.request.id"] == "req-1"
    assert attrs["mom.ensemble"] == "bmom"
    assert attrs["mom.role"] == "fanout"


def test_span_duration_matches_reported_call_time():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, duration_ms=50.0)
    (span,) = exporter.get_finished_spans()
    elapsed_ms = (span.end_time - span.start_time) / 1_000_000
    assert abs(elapsed_ms - 50.0) < 5.0  # start is back-dated by the reported duration


def test_error_sets_error_status():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, error="upstream boom")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


def test_missing_model_falls_back_to_unknown_system():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, model=None)
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.system"] == "unknown"
    assert "gen_ai.request.model" not in attrs


def test_cost_usd_recorded_as_gen_ai_usage_cost():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, cost_usd=0.0123)
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes or {})["gen_ai.usage.cost"] == 0.0123


def test_cost_usd_omitted_when_not_given():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer)
    (span,) = exporter.get_finished_spans()
    assert "gen_ai.usage.cost" not in dict(span.attributes or {})


def test_status_and_finish_reason_recorded_as_mom_attributes():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, status="empty", finish_reason="length")
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["mom.status"] == "empty"
    assert attrs["mom.finish_reason"] == "length"


def test_error_adds_exception_event_with_kind_and_scrubbed_detail():
    tracer, exporter = _tracer_with_exporter()
    _observe(
        tracer,
        error="call failed",
        error_kind="rate_limit",
        error_detail="429 from provider, retry-after 2s",
    )
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "rate_limit"
    assert event.attributes["exception.message"] == "429 from provider, retry-after 2s"


def test_error_without_kind_defaults_exception_type_to_unknown():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer, error="call failed")
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.attributes["exception.type"] == "unknown"


def test_no_error_means_no_exception_event():
    tracer, exporter = _tracer_with_exporter()
    _observe(tracer)
    (span,) = exporter.get_finished_spans()
    assert span.events == ()


def test_observe_never_raises_from_a_broken_tracer():
    class _Boom:
        def start_span(self, **_: Any) -> Any:
            raise RuntimeError("otel exploded")

    # Fire-and-forget: a backend failure must not surface into the request path.
    _observe(OtelTracer(_Boom()))


def test_create_builds_an_otlp_tracer():
    tracer = OtelTracer.create(endpoint="http://localhost:4318", protocol="http")
    assert tracer is not None  # import paths + processor wiring are valid


def test_composite_tracer_fans_out_to_each_backend():
    a, b = RecordingTracer(), RecordingTracer()
    composite = CompositeTracer([a, b])
    composite.observe(request_id="r", role="fanout", llm="a")
    composite.flush()
    assert len(a.observations) == len(b.observations) == 1
    assert a.flushed == b.flushed == 1
