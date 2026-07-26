"""Anthropic Messages encoder: StreamEvent -> Anthropic SSE, and -> a message object.

Folds the same domain event stream into Anthropic's content-block event sequence
(message_start / content_block_* / message_delta / message_stop).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

from mom.domain.events import (
    AnswerDelta,
    Completed,
    PipelineFailed,
    StreamEvent,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.results import EnsembleResult


_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "error": "end_turn",
}


def _sse(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def stop_reason(finish: str) -> str:
    return _STOP_REASON.get(finish, "end_turn")


def build_message(result: EnsembleResult, *, message_id: str, model: str) -> dict[str, Any]:
    """Non-streaming Anthropic message object."""
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": args,
            }
        )
    usage = result.usage
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(result.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


async def encode_sse(
    events: AsyncIterator[StreamEvent], *, message_id: str, model: str
) -> AsyncIterator[bytes]:
    """Fold the event stream into an Anthropic Messages SSE byte stream."""
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    next_index = 0
    open_block: int | None = None
    text_index: int | None = None
    tool_blocks: dict[int, int] = {}
    finish = "stop"
    output_tokens = 0

    def close_open() -> bytes | None:
        nonlocal open_block
        if open_block is None:
            return None
        index = open_block
        open_block = None
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    async for event in events:
        if isinstance(event, AnswerDelta):
            if not event.content:
                continue
            if text_index is None:
                closed = close_open()
                if closed:
                    yield closed
                text_index = next_index
                next_index += 1
                open_block = text_index
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": text_index,
                    "delta": {"type": "text_delta", "text": event.content},
                },
            )
        elif isinstance(event, ToolCallStarted):
            closed = close_open()
            if closed:
                yield closed
            text_index = None
            index = next_index
            next_index += 1
            open_block = index
            tool_blocks[event.index] = index
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": event.call_id,
                        "name": event.name,
                        "input": {},
                    },
                },
            )
        elif isinstance(event, ToolCallDelta):
            block_index = tool_blocks.get(event.index)
            if block_index is not None:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": event.arguments_fragment,
                        },
                    },
                )
        elif isinstance(event, Completed):
            finish = event.finish_reason
            output_tokens = event.usage.completion_tokens
            break
        elif isinstance(event, PipelineFailed):
            closed = close_open()
            if closed:
                yield closed
            yield _sse(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": event.message}},
            )
            return

    closed = close_open()
    if closed:
        yield closed
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason(finish), "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})
