"""Tracing adapters: NoopTracer and a best-effort LangfuseTracer.

Tracing is fire-and-forget: it never raises into the request path, and it is a no-op when
Langfuse is disabled or its credentials are missing. Observations are grouped per request via a
deterministic trace id derived from the request id.
"""

from __future__ import annotations

import hashlib
from typing import Any

from mom.domain.results import Usage
from mom.runtime.logging import get_logger


logger = get_logger("mom.tracing")


class NoopTracer:
    def observe(self, **_: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    """Records each LLM call as a Langfuse generation, grouped by request id."""

    def __init__(self, client: Any) -> None:
        self._client = client

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
            generation = self._client.start_observation(
                trace_context={"trace_id": self._trace_id(request_id)},
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
        except Exception:
            logger.debug("langfuse observe failed", exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.debug("langfuse flush failed", exc_info=True)
