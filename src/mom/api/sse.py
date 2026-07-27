"""Shared Server-Sent-Events helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


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
