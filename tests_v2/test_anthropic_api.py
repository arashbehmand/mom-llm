"""End-to-end Anthropic /v1/messages surface over ASGI with a fake provider."""

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


CONFIG = dedent("""
    version: 2
    server: {{ auth: {auth} }}
    llms:
      a: {{ model: openai/a }}
      b: {{ model: openai/b }}
    ensembles:
      e:
        members: [{{ llm: a }}, {{ llm: b }}]
        synthesizer: {{ llm: a, prompt: p }}
    prompts:
      p: "synthesize"
""")


def _client(fake: FakeLLM, *, auth: str = "none", settings: Settings | None = None):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG.format(auth=auth))))
    container = Container(
        settings=settings or Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.split("\n\n"):
        lines = block.strip().splitlines()
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event_type and data is not None:
            out.append((event_type, data))
    return out


async def test_non_streaming_message():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "e",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0] == {"type": "text", "text": "synthesized answer"}
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["cache_read_input_tokens"] == 0


async def test_streaming_message_events():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "e",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    events = _events(resp.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "message_start"
    assert "content_block_start" in kinds
    assert "content_block_delta" in kinds
    assert kinds[-1] == "message_stop"
    # message_delta carries the stop_reason
    delta = next(d for k, d in events if k == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"
    text = "".join(
        d["delta"]["text"]
        for k, d in events
        if k == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text == "synthesized answer"


async def test_message_tool_use():
    fake = FakeLLM(
        tool_calls=({"id": "toolu_1", "name": "get_weather", "arguments": '{"c":"SF"}'},)
    )
    async with _client(fake) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "e",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            },
        )
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_block = next(b for b in body["content"] if b["type"] == "tool_use")
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"c": "SF"}


async def test_count_tokens():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hello world " * 20}],
            },
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0


async def test_auth_via_x_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOM_API_TOKEN", "s3cret")
    async with _client(FakeLLM(), auth="bearer", settings=Settings(_env_file=None)) as client:
        missing = await client.post(
            "/v1/messages",
            json={"model": "e", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert missing.status_code == 401
        ok = await client.post(
            "/v1/messages",
            headers={"x-api-key": "s3cret"},
            json={"model": "e", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert ok.status_code == 200
