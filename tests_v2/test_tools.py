"""Tool calling: turn classification, synth-owned tool calls, and relay continuation."""

from __future__ import annotations

import json
from textwrap import dedent

import httpx
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.request import MessageIR, ToolCallIR
from mom.domain.tooling import classify_turn
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")

TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
SCRIPTED = ({"id": "call_1", "name": "get_weather", "arguments": '{"city":"SF"}'},)


def _client(client: FakeLLM) -> httpx.AsyncClient:
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=client,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_classify_turn():
    fresh = (MessageIR(role="user", content="hi"),)
    assert classify_turn(fresh) == "fresh"
    relay = (
        MessageIR(role="user", content="hi"),
        MessageIR(role="assistant", tool_calls=(ToolCallIR("call_1", "get_weather", "{}"),)),
        MessageIR(role="tool", content="sunny", tool_call_id="call_1"),
    )
    assert classify_turn(relay) == "relay"


async def test_non_streaming_tool_calls():
    fake = FakeLLM(tool_calls=SCRIPTED)
    async with _client(fake) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather in SF?"}],
                "tools": TOOLS,
            },
        )
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "SF"}
    # the synthesizer received the tools
    assert any("tools" in spec.params for spec in fake.streams)


async def test_streaming_tool_calls():
    fake = FakeLLM(tool_calls=SCRIPTED)
    async with _client(fake) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": TOOLS,
                "stream": True,
            },
        )
    payloads = [
        json.loads(block[len("data: ") :])
        for block in resp.text.split("\n\n")
        if block.strip().startswith("data: ") and not block.strip().endswith("[DONE]")
    ]
    # reassemble the streamed tool call
    name = ""
    args = ""
    for p in payloads:
        for choice in p.get("choices", []):
            for tc in choice.get("delta", {}).get("tool_calls", []):
                name = name or tc.get("function", {}).get("name", "")
                args += tc.get("function", {}).get("arguments", "")
    assert name == "get_weather"
    assert json.loads(args) == {"city": "SF"}
    assert any(
        c.get("finish_reason") == "tool_calls" for p in payloads for c in p.get("choices", [])
    )


async def test_relay_continuation_skips_fanout():
    fake = FakeLLM(replies={"a": "x", "b": "y"})
    async with _client(fake) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "tools": TOOLS,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
            },
        )
    assert resp.status_code == 200
    # relay: the continuation went straight to the synthesizer, no fan-out member calls
    assert fake.completions == []
    assert len(fake.streams) == 1
