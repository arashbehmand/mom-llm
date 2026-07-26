"""Ports: the Protocols the engine depends on. Implementations live in ``adapters``.

Keeping these here (in the pure domain) inverts the dependency — the engine talks to
``LLMClient``/``Clock``/``IdFactory``, never to LiteLLM or aiosqlite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass(frozen=True, slots=True)
class CompletionChunk:
    """A single streamed delta, normalized from any provider (the ONE chunk shape)."""

    content: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_call: dict[str, Any] | None = None


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


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock seconds since the epoch (injected for deterministic tests)."""
        ...


class IdFactory(Protocol):
    def new_id(self, prefix: str) -> str:
        """A fresh unique id with the given prefix."""
        ...
