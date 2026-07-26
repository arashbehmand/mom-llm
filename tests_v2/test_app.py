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


def test_create_app_has_no_import_time_side_effects():
    # Building the app must be pure construction; nothing observable beyond returning the app.
    app = create_app()
    assert app.title == "MoM — Mixture of Models"
    assert app.state.settings is not None
