"""OpenAI Chat Completions encoder: StreamEvent -> SSE, and -> a JSON response.

The single chunk shape lives here. Rules enforced: the first delta carries the assistant role;
a terminal ``finish_reason`` chunk is always emitted (synthesized if the provider omits one);
``[DONE]`` is always last; ``show_work: inline`` renders member perspectives as a ``<think>``
block in the content stream (v1-compatible). A plan's ``notices`` (a ``<<SYSTEM>>`` directive MoM
couldn't honor) head that same block whatever ``show_work`` says — opening it for them alone if
need be, since a client that never sees the panel still has to be told its directive was dropped.
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
    FanoutStarted,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.results import EnsembleResult, ModelOutcome


@dataclass(frozen=True, slots=True)
class ChatFrame:
    id: str
    created: int
    model: str


# User-Agent substrings for clients that need the `compat` tool-call shape (id/type/name on every
# delta). Vercel's AI SDK is the canonical example; matched case-insensitively.
_COMPAT_USER_AGENTS = ("ai-sdk", "vercel-ai", "vercel/ai")


def resolve_stream_profile(configured: str, user_agent: str | None) -> str:
    """The effective streaming profile: a recognized AI-SDK ``User-Agent`` forces ``compat``.

    User-Agent can only *upgrade* strict→compat (the safe direction) for known-fragile clients; it
    never downgrades an ensemble that deliberately configured ``compat``.
    """
    if user_agent and any(marker in user_agent.lower() for marker in _COMPAT_USER_AGENTS):
        return "compat"
    return configured if configured in ("compat", "strict") else "compat"


def _member_line(outcome: ModelOutcome) -> str:
    body = outcome.content if outcome.ok else (outcome.error or outcome.status)
    return f"Model: {html.escape(outcome.model)}\nContent: {html.escape(body)}\n---\n"


def _notice_lines(notices: tuple[str, ...]) -> str:
    """Escaped like member content is: a name quoted back from a directive is user text, and the
    block it lands in is parsed as markup by every client that renders `<think>`. Quotes are left
    alone (`quote=False`) — nothing here becomes an attribute value, and a notice is mostly
    quoted names, which `&#x27;` would turn into noise for a client that doesn't render markup."""
    if not notices:
        return ""
    return "".join(f"{html.escape(notice, quote=False)}\n" for notice in notices) + "\n"


def render_think_block(
    outcomes: tuple[ModelOutcome, ...],
    *,
    progress_url: str | None = None,
    notices: tuple[str, ...] = (),
) -> str:
    if not outcomes and not notices:
        return ""
    header = f"Progress: {progress_url}\n\n" if progress_url else ""
    lines = "".join(_member_line(o) for o in outcomes)
    return f"<think>\n{header}{_notice_lines(notices)}{lines}</think>\n\n"


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
    stream_profile: str = "compat",
    progress_url: str | None = None,
    notices: tuple[str, ...] = (),
) -> AsyncIterator[bytes]:
    """Fold the event stream into an OpenAI SSE byte stream.

    ``stream_profile`` controls the tool-call delta shape: ``compat`` (default) re-emits
    ``id``/``type``/``function.name`` on every delta (safe for AI-SDK-style clients that only read
    the header once); ``strict`` sends them on the first delta only. Both keep ``index`` present as
    an int and ``arguments`` a (possibly empty) string, never null; parallel calls stay serialized
    by their distinct ``index``.
    """
    role_sent = False
    think_open = False
    compat = stream_profile != "strict"
    tool_headers: dict[int, tuple[str, str]] = {}  # index -> (minted id, name), for compat re-emit

    def open_role() -> bytes | None:
        nonlocal role_sent
        if role_sent:
            return None
        role_sent = True
        return _chunk(frame, {"role": "assistant", "content": ""}, None)

    def open_think() -> list[bytes]:
        """The `<think>` + `Progress:` preamble, once. Idempotent — safe to call from every event
        that might be the first thing seen (previously this only ran on the first
        ``MemberCompleted``, so the progress link stayed invisible until the first fan-out member
        finished, which can be minutes; ``FanoutStarted`` fires immediately instead)."""
        nonlocal think_open
        if think_open:
            return []
        think_open = True
        chunks = [_chunk(frame, {"content": "<think>\n"}, None)]
        if progress_url:
            chunks.append(_chunk(frame, {"content": f"Progress: {progress_url}\n\n"}, None))
        if notices:
            chunks.append(_chunk(frame, {"content": _notice_lines(notices)}, None))
        return chunks

    # Notices open the block immediately, before the first event and whatever `show_work` says:
    # an ignored directive is news the human needs *before* the answer it shaped, and on a
    # passthrough or relay turn there is no FanoutStarted coming to open it later.
    if notices:
        first = open_role()
        if first:
            yield first
        for chunk in open_think():
            yield chunk

    async for event in events:
        if isinstance(event, FanoutStarted) and show_work == "inline":
            first = open_role()
            if first:
                yield first
            for chunk in open_think():
                yield chunk
        elif isinstance(event, MemberCompleted) and show_work == "inline":
            first = open_role()
            if first:
                yield first
            for chunk in open_think():  # fallback opener if FanoutStarted never arrived
                yield chunk
            yield _chunk(frame, {"content": _member_line(event.outcome)}, None)
        elif isinstance(event, SynthesisStarted):
            if think_open:
                think_open = False
                yield _chunk(frame, {"content": "</think>\n\n"}, None)
        elif isinstance(event, AnswerDelta):
            first = open_role()
            if first:
                yield first
            if event.reasoning:
                yield _chunk(frame, {"reasoning_content": event.reasoning}, None)
            if event.content:
                yield _chunk(frame, {"content": event.content}, None)
        elif isinstance(event, ToolCallStarted):
            first = open_role()
            if first:
                yield first
            index = int(event.index)
            tool_headers[index] = (event.call_id, event.name)
            delta = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": event.call_id,
                        "type": "function",
                        "function": {"name": event.name, "arguments": ""},
                    }
                ]
            }
            yield _chunk(frame, delta, None)
        elif isinstance(event, ToolCallDelta):
            index = int(event.index)
            call: dict[str, Any] = {
                "index": index,
                "function": {"arguments": event.arguments_fragment},
            }
            if compat and index in tool_headers:
                call_id, name = tool_headers[index]
                call["id"] = call_id
                call["type"] = "function"
                call["function"] = {"name": name, "arguments": event.arguments_fragment}
            yield _chunk(frame, {"tool_calls": [call]}, None)
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
            # Close a `<think>` block opened by an early FanoutStarted/MemberCompleted before a
            # failure (e.g. a quorum miss after the deadline) cuts the fan-out short — previously
            # a pre-existing gap here, now user-visible now that think can open well before any
            # answer content.
            if think_open:
                yield _chunk(frame, {"content": "</think>\n\n"}, None)
                think_open = False
            error = {"message": event.message, "type": "upstream_error", "code": event.code}
            yield f"data: {json.dumps({'error': error})}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return
    # Stream ended without a terminal event — synthesize one.
    if think_open:
        yield _chunk(frame, {"content": "</think>\n\n"}, None)
    yield _chunk(frame, {}, "stop")
    yield b"data: [DONE]\n\n"


def build_completion(
    result: EnsembleResult,
    frame: ChatFrame,
    *,
    show_work: str,
    progress_url: str | None = None,
    notices: tuple[str, ...] = (),
) -> ChatCompletionResponse:
    """Build a non-streaming Chat Completions response from a collected result."""
    content = result.text
    outcomes = result.outcomes if show_work == "inline" else ()
    if outcomes or notices:
        content = render_think_block(outcomes, progress_url=progress_url, notices=notices) + content
    usage = result.usage
    message = ChatMessageOut(
        content=content or None,
        reasoning_content=result.reasoning or None,
        tool_calls=list(result.tool_calls) or None,
    )
    return ChatCompletionResponse(
        id=frame.id,
        created=frame.created,
        model=frame.model,
        choices=[Choice(message=message, finish_reason=result.finish_reason)],
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
