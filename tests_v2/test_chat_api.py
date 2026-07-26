"""End-to-end chat surface over ASGI with a fake provider (no network)."""

from __future__ import annotations

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


def _catalog(auth: str = "none", show_work: str = "off"):
    text = dedent(f"""
        version: 2
        server: {{ auth: {auth} }}
        llms:
          a: {{ model: openai/a }}
          b: {{ model: openai/b }}
        ensembles:
          e:
            members: [{{ llm: a }}, {{ llm: b }}]
            synthesizer: {{ llm: a, prompt: p }}
            show_work: {show_work}
        prompts:
          p: "synthesize"
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def _client(container: Container) -> httpx.AsyncClient:
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _container(*, catalog=None, settings=None, client=None) -> Container:
    return Container(
        settings=settings or Settings(_env_file=None),
        catalog=catalog or _catalog(),
        client=client or FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
    )


def _sse_payloads(body: str) -> list[dict]:
    payloads = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data: ") and not block.endswith("[DONE]"):
            payloads.append(json.loads(block[len("data: ") :]))
    return payloads


async def test_non_streaming_completion():
    async with _client(_container()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "synthesized answer"
    assert body["choices"][0]["finish_reason"] == "stop"
    # aggregate ensemble usage: synth (20) + two members (5 each)
    assert body["usage"]["completion_tokens"] == 30


async def test_streaming_completion():
    async with _client(_container()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert resp.status_code == 200
    assert resp.text.endswith("data: [DONE]\n\n")
    payloads = _sse_payloads(resp.text)
    # first delta carries the assistant role
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if p.get("choices") and p["choices"][0].get("delta")
    )
    assert "synthesized answer" in text
    # a terminal finish_reason and a usage chunk are present
    assert any(
        p.get("choices") and p["choices"][0].get("finish_reason") == "stop" for p in payloads
    )
    assert any(p.get("usage") for p in payloads)


async def test_show_work_inline_renders_think_block():
    async with _client(_container(catalog=_catalog(show_work="inline"))) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("<think>")
    assert "synthesized answer" in content


async def test_unknown_model_is_404():
    async with _client(_container()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_models_endpoint():
    async with _client(_container()) as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert "e" in ids


async def test_auth_enforced(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOM_API_TOKEN", "s3cret")
    container = _container(catalog=_catalog(auth="bearer"), settings=Settings(_env_file=None))
    async with _client(container) as client:
        missing = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert missing.status_code == 401
        ok = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer s3cret"},
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert ok.status_code == 200
