"""The MCP surface mounted on the gateway: routing, the enable gate, and authentication.

Unlike the other API tests these enter the app's lifespan. ``ASGITransport`` does not run one,
and the MCP endpoint needs its session manager started — which is exactly the wiring under test
here. A fresh app per test, because a session manager can only be run once.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
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


CONFIG = """
    version: 2
    server: {{ auth: {auth}, mcp: {{ enabled: {enabled} }} }}
    llms:
      a: {{ model: openai/a }}
    ensembles:
      e:
        members: [{{ llm: a }}]
        synthesizer: {{ llm: a, prompt: p }}
    prompts:
      p: "synthesize"
"""

_JSON_RPC_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


def _container(
    *, auth: str = "bearer", enabled: bool = True, token: str | None = "secret"
) -> Container:
    text = dedent(CONFIG.format(auth=auth, enabled=str(enabled).lower()))
    return Container(
        settings=Settings(_env_file=None, **({"MOM_API_TOKEN": token} if token else {})),
        catalog=resolve_catalog(Config.model_validate(yaml.safe_load(text))),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
    )


@asynccontextmanager
async def _client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container=container)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


async def _post(client: httpx.AsyncClient, *, token: str | None = None, **kwargs):
    headers = dict(_JSON_RPC_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(kwargs.pop("headers", {}))
    return await client.post("/mcp", json=kwargs.pop("body", _INITIALIZE), headers=headers)


def _frames(body: str) -> list[dict]:
    """The endpoint answers JSON-RPC as an SSE stream; pull the payloads back out."""
    return [
        json.loads(line[len("data:") :].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


async def test_disabled_by_default_looks_absent():
    """404 rather than 403: a surface that is switched off should not announce itself to a
    caller probing for it."""
    async with _client(_container(enabled=False)) as client:
        response = await _post(client, token="secret")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Not Found"


@pytest.mark.parametrize("method", ["POST", "GET", "DELETE", "PUT", "PATCH"])
async def test_a_disabled_surface_never_confirms_it_exists(method: str):
    """Anything the outer router answers itself — a 307 to `/mcp/`, a 405 with an `Allow` header
    for a method the route didn't claim — tells an unauthenticated prober the endpoint is there,
    which is the whole point of the 404."""
    async with _client(_container(enabled=False)) as client:
        for path in ("/mcp", "/mcp/"):
            response = await client.request(method, path, headers=_JSON_RPC_HEADERS)
            assert response.status_code == 404, (method, path)
            assert "allow" not in response.headers


async def test_both_path_spellings_reach_the_endpoint():
    async with _client(_container(auth="none", token=None)) as client:
        for path in ("/mcp", "/mcp/"):
            response = await client.post(path, json=_INITIALIZE, headers=_JSON_RPC_HEADERS)
            assert response.status_code == 200, path  # served, not redirected


async def test_the_app_can_be_started_more_than_once():
    """The MCP session manager may only run once per instance, so the lifespan has to build a
    fresh one each time — otherwise an embedder (or two consecutive TestClient blocks) restarting
    one app object fails to come up, whether or not MCP is even enabled."""
    app = create_app(container=_container(enabled=False))
    for _ in range(2):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.parametrize("token", [None, "wrong"])
async def test_requires_the_api_token(token: str | None):
    async with _client(_container()) as client:
        response = await _post(client, token=token)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_accepts_the_x_api_key_header():
    async with _client(_container()) as client:
        response = await _post(client, headers={"x-api-key": "secret"})
    assert response.status_code == 200


async def test_auth_none_opts_out():
    async with _client(_container(auth="none", token=None)) as client:
        response = await _post(client)
    assert response.status_code == 200


async def test_initialize_and_list_tools_over_http():
    async with _client(_container()) as client:
        initialized = await _post(client, token="secret")
        assert initialized.status_code == 200
        assert _frames(initialized.text)[0]["result"]["serverInfo"]["name"] == "mom"

        listed = await _post(
            client,
            token="secret",
            body={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    tools = _frames(listed.text)[0]["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "list_llms",
        "list_ensembles",
        "consult",
        "runs",
        "usage",
        "cache_stats",
    ]


async def test_calls_a_tool_over_http():
    async with _client(_container(auth="none", token=None)) as client:
        await _post(client)  # initialize
        called = await _post(
            client,
            body={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "consult", "arguments": {"prompt": "hi", "ensemble": "e"}},
            },
        )
    payload = _frames(called.text)[0]["result"]["structuredContent"]
    assert payload["status"] == "ok"
    assert payload["answer"] == "synthesized answer"
    # Derived from the request the client actually made, and pointing at the gateway root rather
    # than inside the MCP mount.
    assert payload["progress_url"] == "http://test/v1/progress/req-1"


async def test_serves_under_any_hostname():
    """The SDK's DNS-rebinding guard defaults to localhost-only. Bearer auth is this surface's
    gate and a gateway answers to whatever hostname it is deployed behind."""
    async with _client(_container()) as client:
        response = await _post(client, token="secret", headers={"Host": "mom.example.com"})
    assert response.status_code == 200


async def test_the_rest_of_the_gateway_is_unaffected():
    async with _client(_container(auth="none", token=None)) as client:
        assert (await client.get("/v1/models")).status_code == 200
        assert (await client.get("/health")).status_code == 200
