"""Bearer / x-api-key authentication as a router dependency (timing-safe)."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
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
