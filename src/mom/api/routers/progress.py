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
from mom.api.sse import SSE_HEADERS
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mom &middot; progress &middot; {request_id}</title>
<style>
  :root {{
    --bg: #fff; --fg: #1a1a1a; --muted: #6b6b6b; --border: #e2e2e2; --card: #f7f7f8;
    --pending: #9ca3af; --ok: #16a34a; --err: #dc2626; --err-bg: #fef2f2; --err-fg: #7f1d1d;
    --warn: #d97706; --warn-bg: #fffbeb; --warn-fg: #78350f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0d0d0e; --fg: #e6e6e6; --muted: #9a9a9a; --border: #2b2b2d; --card: #18181a;
      --pending: #6b7280; --ok: #4ade80; --err: #f87171; --err-bg: #2a1414; --err-fg: #fca5a5;
      --warn: #fbbf24; --warn-bg: #2a2110; --warn-fg: #fde68a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font: 14px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    max-width: 64rem; margin: 0 auto; padding: 1.5rem 1rem 4rem;
    color: var(--fg); background: var(--bg);
  }}
  h1 {{ font-size: 1.05rem; font-weight: 600; margin: 0 0 .15rem; }}
  h2 {{ font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
       color: var(--muted); margin: 1.75rem 0 .6rem; }}
  code {{ font: inherit; }}
  .sub {{ color: var(--muted); font-size: .8rem; margin-bottom: .75rem; word-break: break-all; }}
  .badge {{
    display: inline-block; padding: .2rem .6rem; border-radius: 1rem; font-size: .75rem;
    font-weight: 600; background: var(--card); border: 1px solid var(--border);
  }}
  .badge.ok {{ color: var(--ok); border-color: var(--ok); }}
  .badge.err {{ color: var(--err); border-color: var(--err); }}
  .error-banner {{
    margin-top: 1rem; padding: .75rem 1rem; border-radius: .5rem; background: var(--err-bg);
    color: var(--err-fg); border: 1px solid var(--err); white-space: pre-wrap;
    word-break: break-word;
  }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: .75rem;
  }}
  .card {{
    border: 1px solid var(--border); border-radius: .5rem; padding: .7rem .8rem;
    background: var(--card);
  }}
  .card-head {{ display: flex; align-items: center; gap: .45rem; min-width: 0; }}
  .dot {{
    flex: none; width: .55rem; height: .55rem; border-radius: 50%; background: var(--pending);
  }}
  .dot.pending {{ animation: pulse 1.3s ease-in-out infinite; }}
  .dot.ok {{ background: var(--ok); }}
  .dot.err {{ background: var(--err); }}
  .dot.warn {{ background: var(--warn); }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .35; }} }}
  .name {{ font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .meta {{ color: var(--muted); font-size: .75rem; margin-top: .15rem; }}
  .preview {{
    margin-top: .5rem; padding-top: .5rem; border-top: 1px solid var(--border);
    font-size: .8rem; white-space: pre-wrap; word-break: break-word;
    max-height: 12rem; overflow-y: auto; color: var(--fg);
  }}
  .preview.err {{ color: var(--err-fg); }}
  .preview.warn {{ color: var(--warn-fg); }}
  .placeholder {{ color: var(--muted); }}
</style>
</head>
<body>
<h1>progress &middot; <code>{request_id}</code></h1>
<div class="sub" id="ensemble"></div>
<span id="status" class="badge pending">connecting&hellip;</span>
<div id="error-banner"></div>

<h2>Members <span id="member-count"></span></h2>
<div id="members" class="grid"></div>

<h2>Synthesis</h2>
<div id="synth" class="card">
  <div class="card-head">
    <span class="dot pending"></span>
    <span class="name placeholder">waiting for fan-out&hellip;</span>
  </div>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const status = $('status');
  const errorBanner = $('error-banner');
  const members = $('members');
  const memberCount = $('member-count');
  const synth = $('synth');
  let ensembleShown = false;
  let total = 0;
  let filled = 0;
  let fanoutStartMs = null;
  let ticker = null;
  let slotByIdentity = {{}};  // identity -> still-pending card element

  function formatElapsed(ms) {{
    const s = Math.floor(ms / 1000);
    return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's';
  }}

  // A completed member's own duration_ms — distinct from formatElapsed's live ticker text.
  // Previously rendered as a raw millisecond count (e.g. "151000ms" for a 151s member).
  function formatDuration(ms) {{
    if (ms < 1000) return Math.round(ms) + 'ms';
    const s = ms / 1000;
    if (s < 60) return (Math.round(s * 10) / 10) + 's';
    return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
  }}

  function tick() {{
    if (fanoutStartMs === null) return;
    const elapsed = formatElapsed(Date.now() - fanoutStartMs);
    for (const identity in slotByIdentity) {{
      const timer = slotByIdentity[identity].querySelector('.timer');
      if (timer) timer.textContent = 'waiting\\u2026 ' + elapsed;
    }}
  }}

  function stopTicker() {{
    if (ticker !== null) {{ clearInterval(ticker); ticker = null; }}
  }}

  const ESCAPES = {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}};
  function escapeHtml(s) {{
    return s.replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }}

  function showEnsemble(data) {{
    if (ensembleShown || !data.ensemble) return;
    ensembleShown = true;
    $('ensemble').textContent = data.ensemble;
  }}

  function cardHead(dotClass, name, meta) {{
    let html = '<div class="card-head"><span class="dot ' + dotClass + '"></span>' +
      '<span class="name">' + escapeHtml(name) + '</span></div>';
    if (meta) html += '<div class="meta">' + escapeHtml(meta) + '</div>';
    return html;
  }}

  function makePendingCards(memberList, detail) {{
    total = memberList.length;
    filled = 0;
    slotByIdentity = {{}};
    if (total === 0) {{
      // A passthrough/relay turn: no fan-out at all, not a stuck/broken request. Previously
      // nothing was published for this case, so the dashboard just showed a blank page.
      const reason = detail ? ' \\u00b7 ' + escapeHtml(detail) : '';
      members.innerHTML = '<div class="placeholder">fan-out skipped' + reason + '</div>';
      memberCount.textContent = '';
      return;
    }}
    members.innerHTML = '';
    memberList.forEach(([identity, model]) => {{
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = cardHead('pending', identity, model) +
        '<div class="meta timer">waiting\\u2026</div>';
      members.appendChild(div);
      slotByIdentity[identity] = div;
    }});
    memberCount.textContent = '(0/' + total + ')';
  }}

  // Statuses that reached a definite outcome but aren't a clean success and aren't a hard failure
  // either — a member abandoned past the fan-out deadline (still running/cached in the
  // background, or cancelled), or one that answered with nothing. Rendered amber, not red.
  const WARN_STATUSES = new Set(['detached', 'aborted', 'empty']);

  function fillCard(data) {{
    const identity = data.member;
    let slot = identity ? slotByIdentity[identity] : null;
    if (slot) {{
      delete slotByIdentity[identity];
    }} else {{
      // Unknown identity (older server with no `members` list, or a mismatch) — append
      // defensively instead of dropping the update.
      slot = document.createElement('div');
      slot.className = 'card';
      members.appendChild(slot);
    }}
    filled += 1;
    const ok = data.status === 'ok';
    const warn = !ok && WARN_STATUSES.has(data.status);
    const dotClass = ok ? 'ok' : (warn ? 'warn' : 'err');
    const meta = (data.model || '') + (typeof data.duration_ms === 'number'
      ? '  \\u00b7  ' + formatDuration(data.duration_ms) : '');
    slot.className = 'card';
    slot.innerHTML = cardHead(dotClass, data.member || '?', meta);
    if (data.preview) {{
      const p = document.createElement('div');
      p.className = 'preview' + (ok ? '' : ' ' + dotClass);
      p.textContent = data.preview;
      slot.appendChild(p);
    }}
    const shownTotal = typeof data.members_total === 'number' ? data.members_total : total;
    memberCount.textContent = '(' + filled + '/' + shownTotal + ')';
  }}

  // Belt-and-braces: resolve any card that never got its own member_completed event (the server
  // now publishes one for every announced member, including abandoned ones, but a missed/
  // undecoded SSE frame is always possible) instead of leaving it pulsing "waiting..." forever
  // after the request has already ended.
  function sweepPending(note) {{
    for (const identity in slotByIdentity) {{
      const slot = slotByIdentity[identity];
      slot.className = 'card';
      slot.innerHTML = cardHead('warn', identity, note);
    }}
    slotByIdentity = {{}};
  }}

  function fillSynth(dotClass, name, meta, preview, previewErr) {{
    synth.innerHTML = cardHead(dotClass, name, meta);
    if (preview) {{
      const p = document.createElement('div');
      p.className = 'preview' + (previewErr ? ' err' : '');
      p.textContent = preview;
      synth.appendChild(p);
    }}
  }}

  const es = new EventSource(location.href);
  const KINDS = ['fanout_started', 'member_completed', 'synthesis_started', 'completed', 'failed'];
  for (const kind of KINDS) {{
    es.addEventListener(kind, (e) => {{
      const data = JSON.parse(e.data);
      showEnsemble(data);
      if (kind === 'fanout_started') {{
        status.textContent = 'running';
        makePendingCards(Array.isArray(data.members) ? data.members : [], data.detail);
        stopTicker();
        if (total > 0) {{
          fanoutStartMs = Date.now();
          ticker = setInterval(tick, 1000);
          tick();
        }}
      }} else if (kind === 'member_completed') {{
        fillCard(data);
      }} else if (kind === 'synthesis_started') {{
        stopTicker();
        status.textContent = 'synthesizing';
        fillSynth('pending', data.model || data.member || 'synthesizing\\u2026', null);
      }} else if (kind === 'completed') {{
        stopTicker();
        sweepPending('no result recorded');
        status.textContent = 'completed';
        status.className = 'badge ok';
        fillSynth('ok', data.status || 'completed', null, data.preview, false);
        es.close();
      }} else if (kind === 'failed') {{
        stopTicker();
        sweepPending('no result recorded');
        status.textContent = 'failed';
        status.className = 'badge err';
        if (data.detail) {{
          errorBanner.innerHTML = '<div class="error-banner">' + escapeHtml(data.detail) + '</div>';
        }}
        es.close();
      }}
    }});
  }}
  es.onerror = () => {{
    if (es.readyState === EventSource.CLOSED && status.textContent === 'connecting\\u2026') {{
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

    # This route has its own heartbeat built into `event_stream()` (the `: ping` comment on an
    # idle `queue.get()`), so it doesn't use `sse_response`'s `with_heartbeat` wrapping — just the
    # same anti-buffering headers the other three SSE surfaces now carry.
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
