"""The typed event stream the engine emits. Every renderer folds over this — nothing else.

Adding a variant makes ``mypy --strict`` fail every unhandled ``match`` in the encoders, so the
streaming and non-streaming paths cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mom.domain.results import ModelOutcome, Usage


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


@dataclass(frozen=True, slots=True)
class FanoutStarted:
    identity: str
    model: str


@dataclass(frozen=True, slots=True)
class MemberCompleted:
    outcome: ModelOutcome


@dataclass(frozen=True, slots=True)
class FanoutSkipped:
    reason: Literal["tool_continuation", "passthrough"]


@dataclass(frozen=True, slots=True)
class SynthesisStarted:
    llm: str
    model: str


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    content: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    index: int
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    arguments_fragment: str


@dataclass(frozen=True, slots=True)
class Completed:
    finish_reason: FinishReason
    usage: Usage
    total_cost_usd: float


@dataclass(frozen=True, slots=True)
class PipelineFailed:
    code: str
    message: str
    http_status: int = 502


StreamEvent = (
    FanoutStarted
    | MemberCompleted
    | FanoutSkipped
    | SynthesisStarted
    | AnswerDelta
    | ToolCallStarted
    | ToolCallDelta
    | Completed
    | PipelineFailed
)
