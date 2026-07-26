"""Typed error hierarchy. Only ``safe_message`` is ever serialized to a client.

Third-party/provider exception text never reaches a caller (v1 leaked ``str(exc)``); the generic
handler emits a constant message plus a request id for correlation.
"""

from __future__ import annotations


class MomError(Exception):
    """Base error. Subclasses set the HTTP status, OpenAI error ``type``, and short ``code``."""

    http_status: int = 500
    error_type: str = "api_error"
    code: str = "internal_error"

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


class UpstreamError(MomError):
    http_status = 502
    error_type = "upstream_error"
    code = "upstream_error"


class AllModelsFailed(MomError):
    http_status = 502
    error_type = "upstream_error"
    code = "all_models_failed"


class QuorumNotMet(MomError):
    http_status = 502
    error_type = "upstream_error"
    code = "quorum_not_met"
