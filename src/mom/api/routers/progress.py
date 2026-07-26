"""``GET /v1/progress/{request_id}`` — server-sent progress events for one request.

Streams the coarse lifecycle milestones the pipeline publishes on the event bus (fan-out started,
each member completed, synthesis started, completed/failed) as SSE. The stream closes on the
terminal event; a periodic heartbeat comment keeps an otherwise-idle connection alive.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.domain.progress import ProgressEvent


router = APIRouter(dependencies=[Depends(require_api_key)])

_HEARTBEAT_SECONDS = 15.0


def _frame(event: ProgressEvent) -> bytes:
    return f"event: {event.kind}\ndata: {event.to_json()}\n\n".encode()


async def _aclose(iterator: AsyncIterator[ProgressEvent]) -> None:
    aclose: Any = getattr(iterator, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()


@router.get("/progress/{request_id}")
async def progress(request_id: str, container: ContainerDep) -> StreamingResponse:
    bus = container.bus

    async def event_stream() -> AsyncIterator[bytes]:
        if bus is None:
            return
        events = bus.subscribe(request_id)
        # A background pump drives the subscription so the outer loop can time out on a plain
        # queue.get() for heartbeats — cancelling queue.get() (unlike cancelling the generator's
        # __anext__) never tears down the live subscription.
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for event in events:
                    await queue.put(event)
                    if event.terminal:
                        break
            finally:
                await queue.put(None)

        task = asyncio.create_task(pump())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield b": ping\n\n"
                    continue
                if item is None:
                    return
                yield _frame(item)
                if item.terminal:
                    return
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await _aclose(events)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
