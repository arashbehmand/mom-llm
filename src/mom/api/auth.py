"""Bearer / x-api-key authentication as a router dependency (timing-safe)."""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

from mom.api.deps import get_container
from mom.domain.errors import AuthError, ConfigError


def _present_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


async def require_api_key(request: Request) -> None:
    """Enforce the configured auth policy. No-op when the ensemble config sets auth: none."""
    container = get_container(request)
    if container.catalog.config.server.auth == "none":
        return
    if container.settings.api_token is None:
        raise ConfigError("server auth is enabled but no API token is configured")
    presented = _present_token(request)
    if not presented:
        raise AuthError("missing bearer token")
    expected = container.settings.api_token.get_secret_value()
    if not secrets.compare_digest(_digest(presented), _digest(expected)):
        raise AuthError("invalid bearer token")
