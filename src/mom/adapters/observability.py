"""Tracing adapters: NoopTracer, a best-effort LangfuseTracer, and an OTel-based OtelTracer.

Tracing is fire-and-forget: it never raises into the request path, and it is a no-op when the
backend is disabled or misconfigured. Langfuse groups observations per request via a deterministic
trace id derived from the request id, and nests each call's generation under one root span named
after the ensemble (so the trace reads ``bmom``, not ``synthesis:...``); OTel emits one span per
LLM call following the GenAI semantic conventions. ``langfuse`` and ``opentelemetry`` are optional
dependencies imported lazily, so this module imports (and ``NoopTracer`` runs) without either.
"""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import hashlib
import time
from typing import Any

from mom.domain.results import Usage
from mom.runtime.logging import get_logger


logger = get_logger("mom.tracing")


class NoopTracer:
    def observe(self, **_: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class CompositeTracer:
    """Fan an observation out to several tracers (e.g. Langfuse *and* OTel)."""

    def __init__(self, tracers: Iterable[Any]) -> None:
        self._tracers = tuple(tracers)

    def observe(self, **kwargs: Any) -> None:
        for tracer in self._tracers:
            tracer.observe(**kwargs)

    def flush(self) -> None:
        for tracer in self._tracers:
            tracer.flush()


class LangfuseTracer:
    """Records each LLM call as a Langfuse generation, grouped by request id."""

    def __init__(self, client: Any) -> None:
        self._client = client
        # trace_id -> the ensemble-named root span each request's generations nest under. Created
        # on a request's first observed call, closed when its synthesis call is observed.
        self._roots: dict[str, Any] = {}

    @classmethod
    def create(cls, *, public_key: str, secret_key: str, host: str) -> LangfuseTracer | None:
        try:
            from langfuse import Langfuse

            client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        except Exception:
            logger.warning("langfuse init failed; tracing disabled", exc_info=True)
            return None
        return cls(client)

    def _trace_id(self, request_id: str) -> str:
        try:
            return str(self._client.create_trace_id(seed=request_id))
        except Exception:
            return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:32]

    def _root_span(self, trace_id: str, ensemble: str, messages: list[dict[str, Any]]) -> Any:
        """The ensemble-named root span for this trace; its name becomes the trace name."""
        root = self._roots.get(trace_id)
        if root is None:
            self._roots[trace_id] = root = self._start_root(trace_id, ensemble, messages)
        return root

    def _start_root(self, trace_id: str, ensemble: str, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "trace_context": {"trace_id": trace_id},
            "as_type": "span",
            "name": ensemble,
            "input": messages,
        }
        # Sever any ambient OTel span (left current by a prior request's generation) so the root
        # has no stale cross-request parent — a true root of its own trace.
        try:
            from opentelemetry import context as otel_context
        except ImportError:
            return self._client.start_observation(**kwargs)
        token = otel_context.attach(otel_context.Context())
        try:
            return self._client.start_observation(**kwargs)
        finally:
            otel_context.detach(token)

    def observe(
        self,
        *,
        request_id: str,
        ensemble: str,
        role: str,
        llm: str,
        model: str | None,
        messages: list[dict[str, Any]],
        output: str,
        usage: Usage,
        duration_ms: float,
        cached: bool = False,
        error: str | None = None,
    ) -> None:
        try:
            trace_id = self._trace_id(request_id)
            root = self._root_span(trace_id, ensemble, messages)
            generation = root.start_observation(  # nest under the ensemble root, not the trace
                as_type="generation",
                name=f"{role}:{llm}",
                model=model,
                input=messages,
                metadata={
                    "ensemble": ensemble,
                    "role": role,
                    "cached": cached,
                    "duration_ms": duration_ms,
                },
            )
            generation.update(
                output=output,
                usage_details={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "cache_read": usage.cached_prompt_tokens,
                },
            )
            if error:
                generation.update(level="ERROR", status_message=error)
            generation.end()
            if role == "synthesis":  # the concluding call closes the ensemble root span
                closing = self._roots.pop(trace_id, None)
                if closing is not None:
                    closing.update(output=output)
                    closing.end()
        except Exception:
            logger.debug("langfuse observe failed", exc_info=True)

    def flush(self) -> None:
        try:
            # close any request whose synthesis was never observed (disconnect / all-failed)
            for root in list(self._roots.values()):
                with contextlib.suppress(Exception):
                    root.end()
            self._roots.clear()
            self._client.flush()
        except Exception:
            logger.debug("langfuse flush failed", exc_info=True)


def _gen_ai_system(model: str | None) -> str:
    """The provider portion of a ``provider/model`` string (GenAI ``gen_ai.system``)."""
    if not model:
        return "unknown"
    return model.split("/", 1)[0] if "/" in model else model


class OtelTracer:
    """Emits one OTLP span per LLM call using the GenAI semantic conventions.

    The span is timed to the call (``duration_ms`` back-dates its start) and carries GenAI
    attributes (``gen_ai.system`` / ``gen_ai.request.model`` / ``gen_ai.usage.*``) plus a few
    mom-specific ones (request id, ensemble, role) so a whole request's calls group in a trace UI.
    """

    def __init__(self, tracer: Any, provider: Any = None) -> None:
        self._tracer = tracer
        self._provider = provider  # retained for force_flush on shutdown

    @classmethod
    def create(
        cls, *, endpoint: str, protocol: str = "http", service_name: str = "mom-llm"
    ) -> OtelTracer | None:
        """Build an OTLP-exporting tracer, or ``None`` if OTel isn't installed/initializable."""
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            if protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            tracer = provider.get_tracer("mom")
        except Exception:
            logger.warning("otel init failed; tracing disabled", exc_info=True)
            return None
        return cls(tracer, provider)

    def observe(
        self,
        *,
        request_id: str,
        ensemble: str,
        role: str,
        llm: str,
        model: str | None,
        messages: list[dict[str, Any]],  # noqa: ARG002 (part of the Tracer contract)
        output: str,  # noqa: ARG002
        usage: Usage,
        duration_ms: float,
        cached: bool = False,
        error: str | None = None,
    ) -> None:
        try:
            from opentelemetry.trace import SpanKind, Status, StatusCode

            end_ns = time.time_ns()
            start_ns = end_ns - int(max(duration_ms, 0.0) * 1_000_000)
            span = self._tracer.start_span(
                name=f"{role} {llm}", kind=SpanKind.CLIENT, start_time=start_ns
            )
            try:
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.system", _gen_ai_system(model))
                if model:
                    span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
                span.set_attribute("mom.request.id", request_id)
                span.set_attribute("mom.ensemble", ensemble)
                span.set_attribute("mom.role", role)
                span.set_attribute("mom.llm", llm)
                span.set_attribute("mom.cached", cached)
                if error:
                    span.set_status(Status(StatusCode.ERROR, error))
            finally:
                span.end(end_time=end_ns)
        except Exception:
            logger.debug("otel observe failed", exc_info=True)

    def flush(self) -> None:
        try:
            if self._provider is not None:
                self._provider.force_flush()
        except Exception:
            logger.debug("otel flush failed", exc_info=True)
