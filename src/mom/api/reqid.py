"""Resolve the request id used to key metrics, tracing, and the progress event bus.

A client that wants to watch progress needs to know the request id *before* the answer streams
back: it opens ``GET /v1/progress/{id}`` and issues the chat call with the same id via the
``X-Request-Id`` header. So a well-formed client-supplied id is honored; otherwise a fresh id is
minted. The id is echoed back in the ``X-Request-Id`` response header either way.
"""

from __future__ import annotations

import re

from fastapi import Request

from mom.api.auth import link_token
from mom.api.deps import Container
from mom.domain.ports import IdFactory


REQUEST_ID_HEADER = "x-request-id"
PROGRESS_URL_HEADER = "X-MoM-Progress-Url"

# Conservative: safe as a URL path segment and a Redis channel suffix, and bounded in length.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def resolve_request_id(supplied: str | None, ids: IdFactory) -> str:
    """Honor a well-formed client-supplied id, else mint a fresh ``req-…`` id."""
    if supplied is not None and _SAFE_REQUEST_ID.match(supplied):
        return supplied
    return ids.new_id("req")


def progress_url_from_base(
    base: str | None, request_id: str, container: Container, *, with_token: bool = True
) -> str | None:
    """The progress URL against an explicit base, or ``None`` when no base is knowable.

    ``server.public_url`` wins over the caller's base when configured; ``None`` comes back only
    when neither exists — the stdio MCP server, which has no request to derive a host from and so
    genuinely cannot name a reachable link rather than guessing at one.

    ``with_token`` is what makes the link openable by a browser. What rides in the query string
    is a **link token** scoped to this one request id (see ``auth.link_token``), never the gateway
    credential — the URL is printed in think blocks, response headers and saved transcripts, and a
    credential has no business in any of them. It stays off where the link is *data* rather than a
    response header: the MCP surface returns it inside a tool result, and a run's progress feed is
    not something an agent's transcript should hand out either. Over stdio the caller never
    presented a token in the first place.
    """
    public_url = container.catalog.config.server.public_url
    resolved = f"{public_url.rstrip('/')}/" if public_url else base
    if resolved is None:
        return None
    url = f"{resolved.rstrip('/')}/v1/progress/{request_id}"
    token = link_token(container, request_id) if with_token else None
    if token is not None:
        url = f"{url}?token={token}"  # hex, nothing to quote
    return url


def progress_url(http_request: Request, request_id: str, container: Container) -> str:
    """The browser-openable URL for this request's live progress feed.

    Carries a per-request link token as a query param when auth is enabled: a plain link that a
    browser opens directly (navigation, ``EventSource``) can't attach an ``Authorization`` header,
    and the gateway's own token must not be what fills that gap.

    Prefers ``server.public_url`` over the request's own ``Host`` when configured: behind a
    reverse proxy the request typically arrives over an internal network (its ``base_url`` is an
    internal-only hostname), so without an explicit public URL the generated link would be
    unreachable from outside that network.
    """
    # Never None here: an HTTP request always supplies a base.
    url = progress_url_from_base(str(http_request.base_url), request_id, container)
    assert url is not None  # noqa: S101
    return url


def response_headers(
    http_request: Request, request_id: str, container: Container
) -> dict[str, str]:
    """The ``X-Request-Id`` / ``X-MoM-Progress-Url`` pair every chat-surface response carries."""
    return {
        "X-Request-Id": request_id,
        PROGRESS_URL_HEADER: progress_url(http_request, request_id, container),
    }
