"""Resolve the request id used to key metrics, tracing, and the progress event bus.

A client that wants to watch progress needs to know the request id *before* the answer streams
back: it opens ``GET /v1/progress/{id}`` and issues the chat call with the same id via the
``X-Request-Id`` header. So a well-formed client-supplied id is honored; otherwise a fresh id is
minted. The id is echoed back in the ``X-Request-Id`` response header either way.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from fastapi import Request

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


def progress_url(http_request: Request, request_id: str, container: Container) -> str:
    """The browser-openable URL for this request's live progress feed.

    Carries the API token as a query param when auth is enabled: a plain link that a browser
    opens directly (navigation, ``EventSource``) can't attach an ``Authorization`` header.
    """
    url = f"{http_request.base_url}v1/progress/{request_id}"
    token = container.settings.api_token
    if container.catalog.config.server.auth != "none" and token is not None:
        url = f"{url}?token={quote(token.get_secret_value())}"
    return url


def response_headers(
    http_request: Request, request_id: str, container: Container
) -> dict[str, str]:
    """The ``X-Request-Id`` / ``X-MoM-Progress-Url`` pair every chat-surface response carries."""
    return {
        "X-Request-Id": request_id,
        PROGRESS_URL_HEADER: progress_url(http_request, request_id, container),
    }
