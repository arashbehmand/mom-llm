"""The typed event stream the engine emits. Every renderer folds over this — nothing else.

Renderers fold over this with an ``if isinstance(...)`` chain, not an exhaustive ``match`` — mypy
does NOT fail on an unhandled variant, so adding one (e.g. ``MemberAbandoned``) is safe by default:
an encoder that doesn't recognize it just ignores it. That also means a renderer that SHOULD react
to a new variant won't get a type-checker nudge to do so — grep for ``isinstance(event,`` across
``api/encoders/`` when adding one.
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
class MemberAbandoned:
    """A member whose result won't arrive in time — the fan-out deadline elapsed while it was
    still running. It is NOT a ``MemberCompleted`` (there is no ``ModelOutcome`` yet, and its
    eventual result is never recorded here — the caller detaches/cancels it separately and, on
    detach, records its metric once it actually finishes)."""

    identity: str
    model: str


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
    | MemberAbandoned
    | FanoutSkipped
    | SynthesisStarted
    | AnswerDelta
    | ToolCallStarted
    | ToolCallDelta
    | Completed
    | PipelineFailed
)
