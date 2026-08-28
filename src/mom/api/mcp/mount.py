"""Serving the MCP surface on the gateway: routing, the enable gate, and authentication.

The MCP endpoint is an ASGI app, not a router — FastAPI dependencies never run for it, and neither
do the exception handlers installed on the parent. So the two things every other surface gets for
free are done here explicitly, in front of it, reusing the same ``check_token`` the routers use
rather than growing a second notion of who may call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from mom.api.auth import check_token, present_token
from mom.api.errors import error_body
from mom.api.mcp.server import build_mcp_server
from mom.domain.errors import MomError
from mom.runtime.container import Container


MCP_PATH = "/mcp"

# Every method the route claims. Broader than MCP itself uses (GET streams, POST calls, DELETE
# ends a session) so that a *disabled* surface answers 404 to all of them: a route that omitted a
# method would have the outer router answer 405 with an `Allow` header instead — confirming the
# endpoint exists to exactly the unauthenticated caller the 404 is meant to tell nothing.
_METHODS = ["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"]

# The SDK defaults to rejecting any Host header that is not localhost (DNS-rebinding protection
# for a server a browser might reach). Bearer auth is this surface's gate and the gateway is
# routinely reached at whatever hostname a deployment gives it, so that check would reject
# ordinary production traffic while adding nothing here.
_TRANSPORT_SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)


class McpGate:
    """Enable-gate + auth in front of the MCP endpoint.

    Everything it needs is read off the parent app's state per request rather than captured at
    build time: the container and the MCP app are both created by the lifespan (and recreated if
    it runs again), so anything closed over earlier could disagree with what is actually serving.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        state = getattr(scope.get("app"), "state", None)
        container: Container | None = getattr(state, "container", None)
        app: ASGIApp | None = getattr(state, "mcp_app", None)
        if container is None or app is None:
            await _deny(scope, receive, send, 503, "server_error", "gateway is not ready")
            return
        if not container.catalog.config.server.mcp.enabled:
            # 404, not 403: a surface that is switched off should look absent rather than
            # announce itself to anyone who probes the path.
            await _deny(scope, receive, send, 404, "invalid_request_error", "Not Found")
            return
        try:
            check_token(container, present_token(Headers(scope=scope)))
        except MomError as exc:
            await _deny(
                scope,
                receive,
                send,
                exc.http_status,
                exc.error_type,
                exc.safe_message,
                code=exc.code,
            )
            return
        # Both spellings reach the endpoint's own single route, so `/mcp/` is served rather than
        # bounced back to `/mcp` by a redirect the caller would have to follow.
        await app({**scope, "path": MCP_PATH}, receive, send)


async def _deny(
    scope: Scope,
    receive: Receive,
    send: Send,
    status: int,
    error_type: str,
    message: str,
    *,
    code: str | None = None,
) -> None:
    response = JSONResponse(
        status_code=status,
        content=error_body(message, error_type, code or error_type),
    )
    await response(scope, receive, send)


def mount_mcp(app: FastAPI) -> None:
    """Route ``/mcp`` (and ``/mcp/``) to the gate.

    Routes rather than ``app.mount``: a mount matches ``/mcp/`` and answers bare ``/mcp`` with a
    307 issued by the *outer* router — before the gate runs. That redirect would confirm the
    endpoint exists to an unauthenticated caller on a gateway with the surface switched off, which
    is what the gate's 404 exists to prevent. Starlette treats a callable instance as an ASGI app,
    so the gate is routed to directly and sees every request to either spelling.

    Registration is unconditional and reads no config, so building the app stays pure; the
    endpoint itself is attached by the lifespan (``serve_mcp``) and ``server.mcp.enabled`` is
    enforced per request against the container actually serving.
    """
    gate = McpGate()
    for path in (MCP_PATH, f"{MCP_PATH}/"):
        app.router.routes.append(Route(path, gate, methods=_METHODS, name=f"mcp[{path}]"))


@asynccontextmanager
async def serve_mcp(app: FastAPI) -> AsyncIterator[MCPServer[Any]]:
    """Build the MCP server, run its session manager, and publish its ASGI app on ``app.state``.

    Stateless: each POST is self-contained, so nothing is kept per client and a multi-worker
    deployment needs no sticky routing. Progress notifications still work — they ride the response
    stream of the request that triggered them.
    """
    mcp = build_mcp_server(lambda: getattr(app.state, "container", None))
    app.state.mcp_app = mcp.streamable_http_app(
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        transport_security=_TRANSPORT_SECURITY,
    )
    try:
        async with mcp.session_manager.run():
            yield mcp
    finally:
        app.state.mcp_app = None
