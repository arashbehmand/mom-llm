"""Response-cache key derivation — a faithful, bit-compatible port of the v1 algorithm.

This is a crown-jewel function: its SHA256 output is the primary key of the on-disk response
cache, so any change silently invalidates every entry and re-spends the cache against paid
providers. It is ported verbatim from v1 (``mom_service.llm_calls``) and pinned by the
``cache_keys`` golden. Behaviour: canonical JSON over ``{llm_name, model, messages, params}``
with sensitive keys dropped and volatile S3 presigned-URL query params stripped.

The only intentional v2 change (see ``tests/golden/DEVIATIONS.md``) is that ``stream`` /
``stream_options`` are excluded upstream so streaming and non-streaming share an entry — that
happens at the call site, not here, so this module stays bit-identical to v1.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


REDACTED = "[REDACTED]"

# Runtime/volatile params never part of the cache identity.
_IGNORED_PARAMS = frozenset({"api_key", "messages", "num_retries", "timeout"})

_SENSITIVE_EXACT = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "token",
        "secret",
        "secret_key",
        "password",
        "credential",
        "credentials",
        "client_secret",
    }
)
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_token",
    "_secret",
    "_password",
    "_credential",
    "_credentials",
)
_SENSITIVE_PREFIXES = (
    "api_key_",
    "apikey_",
    "token_",
    "secret_",
    "password_",
    "auth_",
    "authorization_",
)

# S3 presigned-URL query params that change on every request.
_S3_PRESIGNED_PARAMS = frozenset(
    {
        "X-Amz-Algorithm",
        "X-Amz-Credential",
        "X-Amz-Date",
        "X-Amz-Expires",
        "X-Amz-Security-Token",
        "X-Amz-Signature",
        "X-Amz-SignedHeaders",
    }
)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _strip_presigned_urls(text: str) -> str:
    if "X-Amz-" not in text:
        return text

    def _strip(match: re.Match[str]) -> str:
        url = match.group(0)
        parsed = urlparse(url)
        if not parsed.query:
            return url
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k not in _S3_PRESIGNED_PARAMS}
        return urlunparse(parsed._replace(query=urlencode(filtered, doseq=True)))

    return _URL_RE.sub(_strip, text)


def is_sensitive_key(key: str) -> bool:
    """Return True when a key likely contains credentials or auth material."""
    lower = key.lower()
    if lower in _SENSITIVE_EXACT:
        return True
    if "authorization" in lower:
        return True
    if lower.endswith(_SENSITIVE_SUFFIXES):
        return True
    return lower.startswith(_SENSITIVE_PREFIXES)


def normalize(value: Any, *, redact_sensitive: bool) -> Any:
    """Normalize nested data for deterministic serialization and optional redaction."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _strip_presigned_urls(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            if is_sensitive_key(key):
                if redact_sensitive:
                    out[key] = REDACTED
                continue
            out[key] = normalize(raw_value, redact_sensitive=redact_sensitive)
        return out
    if isinstance(value, (list, tuple)):
        return [normalize(item, redact_sensitive=redact_sensitive) for item in value]
    return str(value)


def normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Drop volatile/runtime and sensitive params from the cache identity."""
    out: dict[str, Any] = {}
    for raw_key, raw_value in sorted((params or {}).items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if key in _IGNORED_PARAMS or is_sensitive_key(key):
            continue
        out[key] = normalize(raw_value, redact_sensitive=False)
    return out


def cache_key(
    *,
    llm_name: str,
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any] | None,
) -> str:
    """Derive the response-cache key (SHA256 hex) for a single LLM call."""
    key_data = {
        "llm_name": llm_name,
        "model": model,
        "messages": normalize(messages, redact_sensitive=False),
        "params": normalize_params(params),
    }
    canonical = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
