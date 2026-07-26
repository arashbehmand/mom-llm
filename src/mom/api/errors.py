"""Exception handlers: MomError -> OpenAI-shaped error JSON; anything else -> opaque 500."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mom.domain.errors import MomError
from mom.runtime.logging import get_logger


logger = get_logger("mom.api")


def _error_body(message: str, error_type: str, code: str) -> dict[str, object]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MomError)
    async def _handle_mom_error(_request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, MomError):  # narrowing; the handler is only registered for MomError
            body = _error_body("Internal server error", "api_error", "internal_error")
            return JSONResponse(status_code=500, content=body)
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_body(exc.safe_message, exc.error_type, exc.code),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=_error_body("Internal server error", "api_error", "internal_error"),
        )
