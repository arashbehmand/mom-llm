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
from mom.domain.ports import CompletionChunk
from mom.domain.results import Usage
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


async def test_streaming_response_carries_anti_buffering_headers():
    """Previously unset on every SSE surface — a buffering intermediary could swallow
    with_heartbeat's keepalive comments entirely, defeating the one thing meant to stop a slow
    fan-out from tripping a client's idle read-timeout."""
    async with _client(_container()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"


async def test_show_work_inline_renders_think_block():
    async with _client(_container(catalog=_catalog(show_work="inline"))) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("<think>")
    assert "synthesized answer" in content


async def test_show_work_inline_includes_progress_link_in_think_block():
    # A plain link/EventSource can't carry an Authorization header, so the progress link (and
    # the token it carries) needs to be somewhere the user actually sees it — the think block,
    # since lobe-chat has no UI for surfacing an arbitrary response header.
    settings = Settings(_env_file=None, MOM_API_TOKEN="secret-token")
    catalog = _catalog(auth="bearer", show_work="inline")
    async with _client(_container(catalog=catalog, settings=settings)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer secret-token"},
        )
    content = resp.json()["choices"][0]["message"]["content"]
    request_id = resp.headers["x-request-id"]
    assert f"Progress: http://test/v1/progress/{request_id}?token=secret-token" in content
    # the progress line precedes the member dump, inside the think block
    assert content.index("Progress:") < content.index("Model:")


async def test_show_work_inline_streaming_includes_progress_link():
    settings = Settings(_env_file=None, MOM_API_TOKEN="secret-token")
    catalog = _catalog(auth="bearer", show_work="inline")
    async with _client(_container(catalog=catalog, settings=settings)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"authorization": "Bearer secret-token"},
        )
    request_id = resp.headers["x-request-id"]
    text = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in _sse_payloads(resp.text)
        if p.get("choices") and p["choices"][0].get("delta")
    )
    assert f"Progress: http://test/v1/progress/{request_id}?token=secret-token" in text


async def test_show_work_off_has_no_progress_link():
    settings = Settings(_env_file=None, MOM_API_TOKEN="secret-token")
    catalog = _catalog(auth="bearer", show_work="off")
    async with _client(_container(catalog=catalog, settings=settings)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer secret-token"},
        )
    content = resp.json()["choices"][0]["message"]["content"] or ""
    assert "Progress:" not in content


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


async def test_client_params_reach_synthesizer():
    fake = FakeLLM()
    async with _client(_container(client=fake)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.13,
                "top_p": 0.9,
                "max_tokens": 256,
                "stop": ["END"],
                "seed": 42,
                "response_format": {"type": "json_object"},
                "tools": [{"type": "function", "function": {"name": "foo", "parameters": {}}}],
                "tool_choice": "required",
                "parallel_tool_calls": False,
            },
        )
    assert resp.status_code == 200
    # The synthesizer owns the client-visible generation, so client params must reach it.
    params = fake.streams[0].params
    assert params["temperature"] == 0.13
    assert params["top_p"] == 0.9
    assert params["max_tokens"] == 256
    assert params["stop"] == ["END"]
    assert params["seed"] == 42
    assert params["response_format"] == {"type": "json_object"}
    assert params["tool_choice"] == "required"
    assert params["parallel_tool_calls"] is False
    assert params["tools"][0]["function"]["name"] == "foo"


async def test_specific_tool_choice_maps_to_named_function():
    fake = FakeLLM()
    async with _client(_container(client=fake)) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "foo"}}],
                "tool_choice": {"type": "function", "function": {"name": "foo"}},
            },
        )
    assert fake.streams[0].params["tool_choice"] == {
        "type": "function",
        "function": {"name": "foo"},
    }


async def test_advisory_members_do_not_get_client_sampling():
    fake = FakeLLM()
    async with _client(_container(client=fake)) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.13,
            },
        )
    # Members are advisory experts: client sampling shapes the synthesizer, not their tuning.
    assert "temperature" not in fake.completions[0].params


class _ReasoningSynth(FakeLLM):
    async def stream(self, spec):
        self.streams.append(spec)
        yield CompletionChunk(reasoning="thinking hard")
        yield CompletionChunk(content="answer")
        yield CompletionChunk(
            finish_reason="stop", usage=Usage(prompt_tokens=1, completion_tokens=1)
        )


async def test_reasoning_content_non_streaming():
    async with _client(_container(client=_ReasoningSynth())) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "thinking hard"
    assert message["content"] == "answer"


async def test_reasoning_content_streaming():
    async with _client(_container(client=_ReasoningSynth())) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    payloads = _sse_payloads(resp.text)
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "")
        for p in payloads
        if p.get("choices") and p["choices"][0].get("delta")
    )
    assert reasoning == "thinking hard"
