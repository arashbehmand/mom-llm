"""OpenAI Chat Completions encoder: StreamEvent -> SSE, and -> a JSON response.

The single chunk shape lives here. Rules enforced: the first delta carries the assistant role;
a terminal ``finish_reason`` chunk is always emitted (synthesized if the provider omits one);
``[DONE]`` is always last; ``show_work: inline`` renders member perspectives as a ``<think>``
block in the content stream (v1-compatible).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import html
import json
from typing import Any

from mom.api.schemas.openai_chat import (
    ChatCompletionResponse,
    ChatMessageOut,
    Choice,
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from mom.domain.events import (
    AnswerDelta,
    Completed,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
)
from mom.domain.results import EnsembleResult, ModelOutcome


@dataclass(frozen=True, slots=True)
class ChatFrame:
    id: str
    created: int
    model: str


def _member_line(outcome: ModelOutcome) -> str:
    body = outcome.content if outcome.ok else (outcome.error or outcome.status)
    return f"Model: {html.escape(outcome.model)}\nContent: {html.escape(body)}\n---\n"


def render_think_block(outcomes: tuple[ModelOutcome, ...]) -> str:
    if not outcomes:
        return ""
    lines = "".join(_member_line(o) for o in outcomes)
    return f"<think>\n{lines}</think>\n\n"


def _chunk(frame: ChatFrame, delta: dict[str, Any], finish_reason: str | None) -> bytes:
    payload = {
        "id": frame.id,
        "object": "chat.completion.chunk",
        "created": frame.created,
        "model": frame.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _usage_chunk(frame: ChatFrame, usage: dict[str, Any]) -> bytes:
    payload = {
        "id": frame.id,
        "object": "chat.completion.chunk",
        "created": frame.created,
        "model": frame.model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _usage_dict(result_usage: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": result_usage.prompt_tokens,
        "completion_tokens": result_usage.completion_tokens,
        "total_tokens": result_usage.total_tokens,
        "prompt_tokens_details": {"cached_tokens": result_usage.cached_prompt_tokens},
        "completion_tokens_details": {"reasoning_tokens": result_usage.reasoning_tokens},
    }


async def encode_sse(
    events: AsyncIterator[StreamEvent],
    frame: ChatFrame,
    *,
    show_work: str,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    """Fold the event stream into an OpenAI SSE byte stream."""
    role_sent = False
    think_open = False

    def open_role() -> bytes | None:
        nonlocal role_sent
        if role_sent:
            return None
        role_sent = True
        return _chunk(frame, {"role": "assistant", "content": ""}, None)

    async for event in events:
        if isinstance(event, MemberCompleted) and show_work == "inline":
            first = open_role()
            if first:
                yield first
            if not think_open:
                think_open = True
                yield _chunk(frame, {"content": "<think>\n"}, None)
            yield _chunk(frame, {"content": _member_line(event.outcome)}, None)
        elif isinstance(event, SynthesisStarted):
            if think_open:
                think_open = False
                yield _chunk(frame, {"content": "</think>\n\n"}, None)
        elif isinstance(event, AnswerDelta):
            first = open_role()
            if first:
                yield first
            if event.content:
                yield _chunk(frame, {"content": event.content}, None)
        elif isinstance(event, Completed):
            if think_open:
                yield _chunk(frame, {"content": "</think>\n\n"}, None)
                think_open = False
            first = open_role()
            if first:
                yield first
            yield _chunk(frame, {}, event.finish_reason)
            if include_usage:
                yield _usage_chunk(frame, _usage_dict(event.usage))
            yield b"data: [DONE]\n\n"
            return
        elif isinstance(event, PipelineFailed):
            error = {"message": event.message, "type": "upstream_error", "code": event.code}
            yield f"data: {json.dumps({'error': error})}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return
    # Stream ended without a terminal event — synthesize one.
    yield _chunk(frame, {}, "stop")
    yield b"data: [DONE]\n\n"


def build_completion(
    result: EnsembleResult, frame: ChatFrame, *, show_work: str
) -> ChatCompletionResponse:
    """Build a non-streaming Chat Completions response from a collected result."""
    content = result.text
    if show_work == "inline":
        content = render_think_block(result.outcomes) + content
    usage = result.usage
    return ChatCompletionResponse(
        id=frame.id,
        created=frame.created,
        model=frame.model,
        choices=[
            Choice(
                message=ChatMessageOut(content=content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=usage.cached_prompt_tokens),
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=usage.reasoning_tokens
            ),
        ),
    )
