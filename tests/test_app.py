"""App-factory smoke tests over the in-process ASGI transport (no network)."""

from __future__ import annotations

import httpx

from mom import __version__
from mom.api.app import create_app


async def test_health_endpoint():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_create_app_is_pure_construction():
    # Building the app must be pure construction (the container is built in the lifespan).
    app = create_app()
    assert app.title == "MoM — Mixture of Models"
    # Recent Starlette nests included routers (`_IncludedRouter`) instead of flattening them into
    # app.routes, so assert route existence via the generated OpenAPI schema (version-stable).
    assert "/v1/chat/completions" in app.openapi()["paths"]
