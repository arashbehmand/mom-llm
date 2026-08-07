"""Typed error hierarchy. Only ``safe_message`` is ever serialized to a client.

Third-party/provider exception text never reaches a caller (v1 leaked ``str(exc)``); the generic
handler emits a constant message plus a request id for correlation.
"""

from __future__ import annotations

from typing import Literal


# A mom-owned classification of *why* a call failed — never provider text itself. Safe to log,
# store in metrics, and send to Langfuse; used to decide what's worth retrying (see the adapter's
# retry loop) and to group failures without leaking any provider-specific message.
ErrorKind = Literal[
    "timeout",
    "rate_limit",
    "auth",
    "bad_request",
    "context_length",
    "content_filter",
    "connection",
    "server_error",
    "proxy",
    "unknown",
]


class MomError(Exception):
    """Base error. Subclasses set the HTTP status, OpenAI error ``type``, and short ``code``.

    ``kind``/``detail`` are operator-facing only (logs, metrics, tracing) — never part of
    ``safe_message``, which is the sole thing ever serialized to a client. Class-level defaults
    mean every existing ``MomError(message)`` construction keeps working untouched.
    """

    http_status: int = 500
    error_type: str = "api_error"
    code: str = "internal_error"
    kind: ErrorKind = "unknown"
    detail: str | None = None
    attempts: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.safe_message = message


class ConfigError(MomError):
    http_status = 500
    error_type = "api_error"
    code = "config_error"


class AuthError(MomError):
    http_status = 401
    error_type = "authentication_error"
    code = "invalid_api_key"


class UnknownModelError(MomError):
    http_status = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class InvalidRequestError(MomError):
    http_status = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class UpstreamTimeout(MomError):
    http_status = 504
    error_type = "upstream_error"
    code = "timeout"
    kind: ErrorKind = "timeout"


class UpstreamError(MomError):
    """The one exception the litellm adapter raises for every provider-call failure.

    ``kind``/``detail``/``attempts`` are keyword-only and optional so every pre-existing
    ``UpstreamError(message)`` call site is untouched; new call sites that know *why* (a classified
    provider exception, and how many attempts mom's own retry loop made) pass all three.
    """

    http_status = 502
    error_type = "upstream_error"
    code = "upstream_error"

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = "unknown",
        detail: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail
        self.attempts = attempts


class AllModelsFailed(MomError):
    http_status = 502
    error_type = "upstream_error"
    code = "all_models_failed"


class QuorumNotMet(MomError):
    http_status = 502
    error_type = "upstream_error"
    code = "quorum_not_met"
