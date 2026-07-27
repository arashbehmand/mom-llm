"""``GET /v1/progress/{request_id}`` — server-sent progress events for one request.

Streams the coarse lifecycle milestones the pipeline publishes on the event bus (fan-out started,
each member completed, synthesis started, completed/failed) as SSE. The stream closes on the
terminal event; a periodic heartbeat comment keeps an otherwise-idle connection alive.

A browser navigating here (``Accept: text/html``) gets a small self-contained page that opens an
``EventSource`` against this same URL and renders the milestones live; any other ``Accept`` gets
the raw SSE feed. Since a plain link can't carry an ``Authorization`` header, auth also accepts the
API token as a ``?token=`` query param (see ``X-MoM-Progress-Url`` in :mod:`mom.api.reqid`).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
import html
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from mom.api.auth import check_token
from mom.api.deps import ContainerDep
from mom.domain.progress import ProgressEvent


router = APIRouter()

_HEARTBEAT_SECONDS = 15.0


def _presented_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key") or request.query_params.get("token")


async def _progress_auth(request: Request, container: ContainerDep) -> None:
    check_token(container, _presented_token(request))


def _frame(event: ProgressEvent) -> bytes:
    return f"event: {event.kind}\ndata: {event.to_json()}\n\n".encode()


async def _aclose(iterator: AsyncIterator[ProgressEvent]) -> None:
    aclose: Any = getattr(iterator, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/event-stream" not in accept and "text/html" in accept


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>mom &middot; progress &middot; {request_id}</title>
<style>
  body {{ font: 14px/1.5 ui-monospace, monospace; max-width: 40rem; margin: 2rem auto;
         padding: 0 1rem; color: #1a1a1a; background: #fff; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #ddd; background: #111; }} }}
  h1 {{ font-size: 1rem; font-weight: 600; }}
  #status {{ display: inline-block; padding: .15rem .5rem; border-radius: .3rem;
            background: #eee; }}
  @media (prefers-color-scheme: dark) {{ #status {{ background: #333; }} }}
  ol {{ list-style: none; margin: 1rem 0 0; padding: 0; }}
  li {{ padding: .3rem 0; border-top: 1px solid #eee; }}
  @media (prefers-color-scheme: dark) {{ li {{ border-color: #333; }} }}
  .detail {{ opacity: .65; }}
</style>
</head>
<body>
<h1>progress &middot; <code>{request_id}</code></h1>
<p id="status">connecting&hellip;</p>
<ol id="log"></ol>
<script>
  const log = document.getElementById('log');
  const status = document.getElementById('status');
  const LABELS = {{
    fanout_started: 'fan-out started',
    member_completed: 'member completed',
    synthesis_started: 'synthesis started',
    completed: 'completed',
    failed: 'failed',
  }};
  function line(kind, data) {{
    const li = document.createElement('li');
    const parts = [LABELS[kind] || kind];
    if (data.member) parts.push(data.member);
    if (data.model) parts.push('(' + data.model + ')');
    if (data.status) parts.push('&mdash; ' + data.status);
    if (typeof data.completed === 'number' && typeof data.members_total === 'number') {{
      parts.push('[' + data.completed + '/' + data.members_total + ']');
    }}
    li.textContent = parts.join(' ');
    if (data.detail) {{
      const d = document.createElement('div');
      d.className = 'detail';
      d.textContent = data.detail;
      li.appendChild(d);
    }}
    log.appendChild(li);
  }}
  const es = new EventSource(location.href);
  for (const kind of Object.keys(LABELS)) {{
    es.addEventListener(kind, (e) => {{
      const data = JSON.parse(e.data);
      line(kind, data);
      if (kind === 'fanout_started') status.textContent = 'running';
      if (kind === 'completed' || kind === 'failed') {{
        status.textContent = kind;
        es.close();
      }}
    }});
  }}
  es.onerror = () => {{
    if (es.readyState === EventSource.CLOSED && status.textContent === 'connecting&hellip;') {{
      status.textContent = 'no live progress for this request (already finished, or unknown id)';
    }}
  }};
</script>
</body>
</html>
"""


def _render_page(request_id: str) -> str:
    return _PAGE.format(request_id=html.escape(request_id))


@router.get("/progress/{request_id}", dependencies=[Depends(_progress_auth)])
async def progress(request_id: str, container: ContainerDep, request: Request) -> Response:
    if _wants_html(request):
        return HTMLResponse(_render_page(request_id))

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
