"""OpenAI Responses encoder: StreamEvent -> response.* SSE events, and -> a response object.

Stateless Open Responses subset: one message item for the answer, a function_call item per tool
call, a single centralized ``sequence_number`` counter, and both ``event:`` and ``data:`` lines.

``show_work: inline`` surfaces member perspectives as a reasoning item (this surface's native
"thinking" convention — no ``<think>`` tags, unlike the Chat Completions encoder), closed off
before any genuine synthesizer reasoning opens its own reasoning item.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

from mom.domain.events import (
    AnswerDelta,
    Completed,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.results import EnsembleResult, ModelOutcome


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def _summary_part(text: str) -> dict[str, Any]:
    return {"type": "summary_text", "text": text}


def _member_line(outcome: ModelOutcome) -> str:
    body = outcome.content if outcome.ok else (outcome.error or outcome.status)
    return f"Model: {outcome.model}\nContent: {body}\n---\n"


def _response_obj(
    *, response_id: str, model: str, created: int, status: str, output: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "store": False,
    }


def build_response(
    result: EnsembleResult, *, response_id: str, model: str, created: int, show_work: str = "off"
) -> dict[str, Any]:
    """Non-streaming Responses object."""
    output: list[dict[str, Any]] = []
    if show_work == "inline" and result.outcomes:
        member_text = "".join(_member_line(o) for o in result.outcomes)
        output.append(
            {
                "type": "reasoning",
                "id": f"{response_id}-rs-members",
                "summary": [{"type": "summary_text", "text": member_text}],
            }
        )
    if result.reasoning:  # a reasoning item precedes the answer message, mirroring the stream
        output.append(
            {
                "type": "reasoning",
                "id": f"{response_id}-rs",
                "summary": [{"type": "summary_text", "text": result.reasoning}],
            }
        )
    if result.text:
        output.append(
            {
                "type": "message",
                "id": f"{response_id}-msg",
                "role": "assistant",
                "status": "completed",
                "content": [_text_part(result.text)],
            }
        )
    for i, call in enumerate(result.tool_calls):
        fn = call.get("function", {})
        output.append(
            {
                "type": "function_call",
                "id": f"{response_id}-fc{i}",
                "call_id": call.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
                "status": "completed",
            }
        )
    obj = _response_obj(
        response_id=response_id, model=model, created=created, status="completed", output=output
    )
    usage = result.usage
    obj["usage"] = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    return obj


async def encode_sse(
    events: AsyncIterator[StreamEvent],
    *,
    response_id: str,
    model: str,
    created: int,
    show_work: str = "off",
) -> AsyncIterator[bytes]:
    """Fold the event stream into a Responses SSE byte stream."""
    seq = 0

    def evt(event_type: str, data: dict[str, Any]) -> bytes:
        nonlocal seq
        payload = {"type": event_type, "sequence_number": seq, **data}
        seq += 1
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

    def skeleton(status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
        return _response_obj(
            response_id=response_id, model=model, created=created, status=status, output=output
        )

    yield evt("response.created", {"response": skeleton("in_progress", [])})
    yield evt("response.in_progress", {"response": skeleton("in_progress", [])})

    next_index = 0
    reasoning: tuple[int, str] | None = None  # (output_index, item_id)
    reasoning_buf: list[str] = []
    text: tuple[int, str] | None = None  # (output_index, item_id)
    text_buf: list[str] = []
    tools: dict[int, list[Any]] = {}  # engine index -> [output_index, item_id, name, call_id, buf]
    output: list[dict[str, Any]] = []

    def close_reasoning() -> list[bytes]:
        nonlocal reasoning
        if reasoning is None:
            return []
        index, item_id = reasoning
        reasoning = None
        joined = "".join(reasoning_buf)
        reasoning_buf.clear()  # a later reasoning item must not re-emit this one's summary
        part = _summary_part(joined)
        item = {"type": "reasoning", "id": item_id, "summary": [part]}
        output.append(item)
        return [
            evt(
                "response.reasoning_summary_text.done",
                {"item_id": item_id, "output_index": index, "summary_index": 0, "text": joined},
            ),
            evt(
                "response.reasoning_summary_part.done",
                {"item_id": item_id, "output_index": index, "summary_index": 0, "part": part},
            ),
            evt("response.output_item.done", {"output_index": index, "item": item}),
        ]

    def close_text() -> list[bytes]:
        nonlocal text
        if text is None:
            return []
        index, item_id = text
        text = None
        joined = "".join(text_buf)
        text_buf.clear()  # a later text item must not re-emit this one's text
        item = {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "completed",
            "content": [_text_part(joined)],
        }
        output.append(item)
        return [
            evt(
                "response.output_text.done",
                {"item_id": item_id, "output_index": index, "content_index": 0, "text": joined},
            ),
            evt(
                "response.content_part.done",
                {
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "part": _text_part(joined),
                },
            ),
            evt("response.output_item.done", {"output_index": index, "item": item}),
        ]

    def finalize_tools() -> list[bytes]:
        chunks: list[bytes] = []
        for entry in tools.values():
            index, item_id, name, call_id, buf = entry
            args = "".join(buf)
            chunks.append(
                evt(
                    "response.function_call_arguments.done",
                    {"item_id": item_id, "output_index": index, "arguments": args},
                )
            )
            item = {
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": args,
                "status": "completed",
            }
            output.append(item)
            chunks.append(evt("response.output_item.done", {"output_index": index, "item": item}))
        tools.clear()
        return chunks

    def reasoning_delta(delta: str) -> list[bytes]:
        nonlocal reasoning, next_index
        chunks: list[bytes] = []
        if reasoning is None:
            index = next_index
            next_index += 1
            item_id = f"{response_id}-rs{index}"
            reasoning = (index, item_id)
            chunks.append(
                evt(
                    "response.output_item.added",
                    {
                        "output_index": index,
                        "item": {"type": "reasoning", "id": item_id, "summary": []},
                    },
                )
            )
            chunks.append(
                evt(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index": 0,
                        "part": _summary_part(""),
                    },
                )
            )
        reasoning_buf.append(delta)
        chunks.append(
            evt(
                "response.reasoning_summary_text.delta",
                {
                    "item_id": reasoning[1],
                    "output_index": reasoning[0],
                    "summary_index": 0,
                    "delta": delta,
                },
            )
        )
        return chunks

    usage: dict[str, int] | None = None
    async for event in events:
        if isinstance(event, MemberCompleted) and show_work == "inline":
            for chunk in reasoning_delta(_member_line(event.outcome)):
                yield chunk
        elif isinstance(event, SynthesisStarted):
            # Member-dump reasoning ends here; genuine synthesizer reasoning opens its own item.
            for chunk in close_reasoning():
                yield chunk
        elif isinstance(event, AnswerDelta):
            if event.reasoning:
                for chunk in reasoning_delta(event.reasoning):
                    yield chunk
            if event.content:
                for chunk in close_reasoning():  # the reasoning summary precedes the answer text
                    yield chunk
                if text is None:
                    index = next_index
                    next_index += 1
                    item_id = f"{response_id}-msg{index}"
                    text = (index, item_id)
                    yield evt(
                        "response.output_item.added",
                        {
                            "output_index": index,
                            "item": {
                                "type": "message",
                                "id": item_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    )
                    yield evt(
                        "response.content_part.added",
                        {
                            "item_id": item_id,
                            "output_index": index,
                            "content_index": 0,
                            "part": _text_part(""),
                        },
                    )
                text_buf.append(event.content)
                yield evt(
                    "response.output_text.delta",
                    {
                        "item_id": text[1],
                        "output_index": text[0],
                        "content_index": 0,
                        "delta": event.content,
                    },
                )
        elif isinstance(event, ToolCallStarted):
            for chunk in close_reasoning():
                yield chunk
            for chunk in close_text():
                yield chunk
            index = next_index
            next_index += 1
            item_id = f"{response_id}-fc{event.index}"
            tools[event.index] = [index, item_id, event.name, event.call_id, []]
            yield evt(
                "response.output_item.added",
                {
                    "output_index": index,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        elif isinstance(event, ToolCallDelta):
            entry = tools.get(event.index)
            if entry is not None:
                entry[4].append(event.arguments_fragment)
                yield evt(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": entry[1],
                        "output_index": entry[0],
                        "delta": event.arguments_fragment,
                    },
                )
        elif isinstance(event, Completed):
            usage = {
                "input_tokens": event.usage.prompt_tokens,
                "output_tokens": event.usage.completion_tokens,
                "total_tokens": event.usage.total_tokens,
            }
            break
        elif isinstance(event, PipelineFailed):
            for chunk in close_reasoning():
                yield chunk
            for chunk in close_text():
                yield chunk
            for chunk in finalize_tools():  # finalize any opened tool items — no danglers
                yield chunk
            failed = skeleton("failed", output)
            failed["error"] = {"code": event.code, "message": event.message}
            if usage is not None:
                failed["usage"] = usage
            yield evt("response.failed", {"response": failed})
            return

    for chunk in close_reasoning():
        yield chunk
    for chunk in close_text():
        yield chunk
    for chunk in finalize_tools():
        yield chunk

    completed = skeleton("completed", output)
    if usage is not None:
        completed["usage"] = usage
    yield evt("response.completed", {"response": completed})
