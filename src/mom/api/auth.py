"""Bearer / x-api-key authentication as a router dependency (timing-safe), plus the scoped
link token the progress URL carries instead of the gateway credential."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import secrets

from fastapi import Request

from mom.api.deps import Container, get_container
from mom.domain.errors import AuthError, ConfigError


def present_token(headers: Mapping[str, str]) -> str | None:
    """The token a caller presented, from ``Authorization: Bearer`` or ``x-api-key``.

    Takes headers rather than a ``Request`` so the mounted MCP sub-app — which sees a raw ASGI
    scope, never a FastAPI request — reads them the same way every router does.
    """
    header = headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return headers.get("x-api-key")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def check_token(container: Container, presented: str | None) -> None:
    """Raise if ``presented`` doesn't match the configured API token. Shared by every auth path."""
    if container.catalog.config.server.auth == "none":
        return
    if container.settings.api_token is None:
        raise ConfigError("server auth is enabled but no API token is configured")
    if not presented:
        raise AuthError("missing bearer token")
    expected = container.settings.api_token.get_secret_value()
    if not secrets.compare_digest(_digest(presented), _digest(expected)):
        raise AuthError("invalid bearer token")


async def require_api_key(request: Request) -> None:
    """Enforce the configured auth policy. No-op when the ensemble config sets auth: none."""
    container = get_container(request)
    check_token(container, present_token(request.headers))


def link_token(container: Container, request_id: str) -> str | None:
    """A capability token for ONE request's progress feed: HMAC(api_token, request_id).

    A progress link has to open in a browser — navigation and ``EventSource`` cannot attach an
    ``Authorization`` header — so something authenticating must ride in the query string. It used
    to be the gateway's own API token, which put a credential into a URL that MoM then printed in
    the think block, the ``X-MoM-Progress-Url`` header, and every transcript a client kept.

    This is derived from that token instead of being it: unforgeable without the token, scoped to
    the one request id it was minted for, and worth nothing anywhere else on the gateway. No new
    configuration — the key is the API token itself, so nothing to rotate separately.

    ``None`` when auth is off or no token is configured: there is nothing to authenticate against
    and the link needs no credential.
    """
    token = container.settings.api_token
    if container.catalog.config.server.auth == "none" or token is None:
        return None
    signature = hmac.new(
        token.get_secret_value().encode("utf-8"),
        f"progress:{request_id}".encode(),
        hashlib.sha256,
    )
    return signature.hexdigest()[:32]


def check_link_access(container: Container, request_id: str, presented: str | None) -> None:
    """Authorize a progress request: this id's link token, or the API token itself.

    The API token still works (a client that has it can watch any run, and links minted before
    this existed keep opening); the link token only ever unlocks the id it names.
    """
    expected = link_token(container, request_id)
    if (
        expected is not None
        and presented is not None
        and secrets.compare_digest(_digest(presented), _digest(expected))
    ):
        return
    check_token(container, presented)
