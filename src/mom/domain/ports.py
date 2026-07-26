"""Ports: the Protocols the engine depends on. Implementations live in ``adapters``.

Keeping these here (in the pure domain) inverts the dependency — the engine talks to
``LLMClient``/``Clock``/``IdFactory``, never to LiteLLM or aiosqlite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from mom.domain.progress import ProgressEvent
from mom.domain.results import Usage


@dataclass(frozen=True, slots=True)
class CallSpec:
    """Everything one upstream LLM call needs, provider-neutral."""

    llm_name: str
    model: str
    messages: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)
    api: str = "chat"
    proxy_url_env: str | None = None
    key_env_candidates: tuple[str, ...] = ()
    retries: int = 0
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """A non-streaming completion, normalized from any provider."""

    content: str
    reasoning: str | None
    finish_reason: str
    usage: Usage
    tool_calls: tuple[dict[str, Any], ...] = ()
    cached: bool = False
    cost_usd: float | None = None  # provider cost from the adapter's cost map (None if unknown)


@dataclass(frozen=True, slots=True)
class CompletionChunk:
    """A single streamed delta, normalized from any provider (the ONE chunk shape)."""

    content: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_call: dict[str, Any] | None = None
    cost_usd: float | None = None


class LLMClient(Protocol):
    """Transport to a single upstream model."""

    async def complete(self, spec: CallSpec) -> Completion:
        """Non-streaming call (used for fan-out members)."""
        ...

    def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        """Streaming call (used for the synthesizer)."""
        ...


class CacheStore(Protocol):
    async def get(self, key: str, *, now: float) -> str | None: ...
    async def put(self, key: str, llm: str, body: str, *, now: float) -> None: ...


class Tracer(Protocol):
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
        """Record one LLM call as an observation (grouped by request_id). Fire-and-forget."""
        ...

    def flush(self) -> None:
        """Flush buffered observations (called on shutdown)."""
        ...


class TokenEstimator(Protocol):
    def count(self, *, model: str, messages: list[dict[str, Any]]) -> int:
        """Best-effort input-token count for ``messages`` against ``model``. Never raises."""
        ...


class EventBus(Protocol):
    """Publish/subscribe for per-request progress events, keyed by ``request_id``.

    ``publish`` is fire-and-forget: it must never block the request path nor raise. ``subscribe``
    returns an async iterator that yields a request's progress events — any buffered history first,
    then live events — and completes once a terminal event (``completed``/``failed``) is seen.
    """

    def publish(self, request_id: str, event: ProgressEvent) -> None:
        """Publish one progress event for a request (fire-and-forget)."""
        ...

    def subscribe(self, request_id: str) -> AsyncIterator[ProgressEvent]:
        """Stream a request's progress events (history first, then live, until terminal)."""
        ...


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock seconds since the epoch (injected for deterministic tests)."""
        ...


class IdFactory(Protocol):
    def new_id(self, prefix: str) -> str:
        """A fresh unique id with the given prefix."""
        ...


class ToolCallCustody(Protocol):
    """Custody of provider-native tool-call ids behind MoM-minted, client-facing ids.

    The provider's raw id (e.g. Gemini's ``call_..__thought__..`` signature) never reaches the
    client; it is stashed here under the minted id so a later relay continuation can restore it
    for the *same* owner (the synthesizer that emitted the call). Best-effort and in-memory: a
    miss (a restart, another worker) simply relays the minted id, which providers still accept.
    """

    def remember(self, client_id: str, provider_id: str, owner: str) -> None:
        """Stash ``provider_id`` (owned by ``owner``) under the minted ``client_id``."""
        ...

    def provider_id(self, client_id: str, owner: str) -> str | None:
        """The stored provider id for ``client_id`` iff it was minted by ``owner`` (else None)."""
        ...
