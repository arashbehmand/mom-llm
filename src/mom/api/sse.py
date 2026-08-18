"""Shared Server-Sent-Events helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse


# `Cache-Control: no-cache` stops an intermediary from caching/buffering the stream waiting for it
# to "complete"; `X-Accel-Buffering: no` is nginx's own opt-out of response buffering (harmless on
# any proxy that doesn't understand it). Neither was set anywhere before this — a buffering
# intermediary between mom and a client can swallow `with_heartbeat`'s keepalive comments entirely,
# defeating the one thing that's supposed to stop a slow fan-out from tripping a client's idle
# read-timeout (the observed trigger for lobe-chat sending duplicate full-turn retries).
SSE_HEADERS: dict[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def with_heartbeat(
    stream: AsyncIterator[bytes], interval: float, comment: bytes = b": keepalive\n\n"
) -> AsyncIterator[bytes]:
    """Relay an SSE byte stream, injecting a keepalive ``comment`` whenever it idles ``interval``s.

    An SSE comment (a line beginning with ``:``) carries no event or data, so clients ignore it,
    but it keeps the connection and the client's *idle* read-timeout alive through a slow fan-out
    that would otherwise emit nothing for minutes. On close the pending pull is cancelled, which
    propagates into the underlying pipeline (so ``detach_on_disconnect`` still fires).
    """
    ait = stream.__aiter__()
    pull: asyncio.Task[bytes] | None = None
    try:
        while True:
            if pull is None:
                pull = asyncio.ensure_future(ait.__anext__())
            done, _ = await asyncio.wait({pull}, timeout=interval)
            if not done:  # idle for `interval`s — keep the connection warm
                yield comment
                continue
            try:
                item = pull.result()
            except StopAsyncIteration:
                return
            pull = None
            yield item
    finally:
        if pull is not None and not pull.done():
            pull.cancel()


def sse_response(
    stream: AsyncIterator[bytes], *, headers: dict[str, str], heartbeat_seconds: float | None
) -> StreamingResponse:
    """Build the ``StreamingResponse`` for an SSE stream — the ONE place all three streaming
    routers (chat completions, Responses, Anthropic messages) do this, so a fourth surface can't
    forget the anti-buffering headers or the heartbeat wrapping the way two of the three did
    before this fix (``chat.py`` had the heartbeat but no surface had the headers).

    ``heartbeat_seconds`` is ``None`` when ``server.stream_heartbeat`` is unset — no wrapping, so
    the stream is untouched (matches ``with_heartbeat`` being opt-in today).
    """
    if heartbeat_seconds is not None:
        stream = with_heartbeat(stream, heartbeat_seconds)
    return StreamingResponse(
        stream, media_type="text/event-stream", headers={**headers, **SSE_HEADERS}
    )
