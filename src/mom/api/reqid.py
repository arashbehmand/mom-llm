"""Resolve the request id used to key metrics, tracing, and the progress event bus.

A client that wants to watch progress needs to know the request id *before* the answer streams
back: it opens ``GET /v1/progress/{id}`` and issues the chat call with the same id via the
``X-Request-Id`` header. So a well-formed client-supplied id is honored; otherwise a fresh id is
minted. The id is echoed back in the ``X-Request-Id`` response header either way.
"""

from __future__ import annotations

import re

from mom.domain.ports import IdFactory


REQUEST_ID_HEADER = "x-request-id"

# Conservative: safe as a URL path segment and a Redis channel suffix, and bounded in length.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def resolve_request_id(supplied: str | None, ids: IdFactory) -> str:
    """Honor a well-formed client-supplied id, else mint a fresh ``req-…`` id."""
    if supplied is not None and _SAFE_REQUEST_ID.match(supplied):
        return supplied
    return ids.new_id("req")
