"""CORS middleware wiring over the in-process ASGI transport (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import httpx
import pytest
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


def _catalog(*, origins: list[str], allow_credentials: bool = False):
    text = dedent(f"""
        version: 2
        server:
          cors:
            origins: {json.dumps(origins)}
            allow_credentials: {json.dumps(allow_credentials)}
        llms:
          a: {{ model: openai/a }}
        ensembles:
          e:
            members: [{{ llm: a }}]
            synthesizer: {{ llm: a }}
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def _container(catalog, *, settings: Settings | None = None) -> Container:
    return Container(
        settings=settings or Settings(_env_file=None),
        catalog=catalog,
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
    )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_cors_header_present_for_allowed_origin():
    app = create_app(container=_container(_catalog(origins=["https://example.com"])))
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://example.com"})
    assert resp.status_code == 200
    # A specific (non-wildcard) allowed origin is echoed back on the response.
    assert resp.headers["access-control-allow-origin"] == "https://example.com"


async def test_no_cors_header_when_origins_empty():
    # No configured origins -> no middleware installed -> browsers get no CORS grant.
    app = create_app(container=_container(_catalog(origins=[])))
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


async def test_allow_credentials_propagates():
    catalog = _catalog(origins=["https://example.com"], allow_credentials=True)
    app = create_app(container=_container(catalog))
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://example.com"})
    assert resp.headers["access-control-allow-origin"] == "https://example.com"
    assert resp.headers["access-control-allow-credentials"] == "true"


async def test_create_app_with_no_container_and_no_config_skips_cors():
    # The pure construction path (no container, no config file) must still build a working app.
    app = create_app()
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


async def test_cors_loaded_from_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Production path: no container is passed, so create_app best-effort loads settings.config_file.
    config = tmp_path / "models.yaml"
    config.write_text(
        dedent("""
            version: 2
            server:
              cors:
                origins: ["https://prod.example.com"]
            llms:
              a: { model: openai/a }
            ensembles:
              e:
                members: [{ llm: a }]
                synthesizer: { llm: a }
        """),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOM_CONFIG", str(config))
    app = create_app(Settings(_env_file=None))
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://prod.example.com"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://prod.example.com"
