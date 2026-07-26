"""Fan-out outcomes and token usage — frozen value types, the pipeline's currency.

An ``Opinion`` is a sum type (``Answer`` | ``Failure`` | ``Skipped``); success/failure is never a
string prefix on the content (v1's brittle ``content.startswith("Error:")`` convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


OutcomeStatus = Literal["ok", "empty", "error", "timeout", "skipped", "aborted"]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage for one call. Cached/written tokens are first-class for cache-aware cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    """The result of asking one member for its perspective."""

    identity: str
    llm: str
    model: str
    status: OutcomeStatus
    content: str = ""
    reasoning: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    cached: bool = False
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """The collected (non-streaming) result of a whole ensemble request."""

    text: str
    outcomes: tuple[ModelOutcome, ...]
    usage: Usage
    total_cost_usd: float
    finish_reason: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    reasoning: str = ""

    @property
    def successful(self) -> tuple[ModelOutcome, ...]:
        return tuple(o for o in self.outcomes if o.ok)
