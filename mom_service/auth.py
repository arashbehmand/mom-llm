"""
Authentication dependency for API endpoints.

Provides a reusable FastAPI dependency function that validates Bearer token authentication
for all protected endpoints in the MoM service.
"""

import os

from fastapi import HTTPException, Request


def verify_bearer_token(request: Request) -> None:
    """
    FastAPI dependency that verifies Bearer token authentication.

    This function checks:
    1. API_TOKEN environment variable is configured
    2. Authorization header is present and uses Bearer scheme
    3. The provided token matches the configured API_TOKEN

    Args:
        request: FastAPI Request object

    Raises:
        HTTPException: 503 if API_TOKEN not configured, 401 if authentication fails

    Usage:
        @router.get("/endpoint", dependencies=[Depends(verify_bearer_token)])
        async def protected_endpoint():
            ...
    """
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service API token not configured by administrator.",
                "type": "service_unavailable",
            },
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid authentication scheme. Use Bearer token.",
                "type": "authentication_error",
            },
        )

    token = auth_header.split(" ", 1)[1]
    if token != api_token:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid or missing API token.",
                "type": "authentication_error",
            },
        )
