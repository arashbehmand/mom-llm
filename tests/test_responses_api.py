"""End-to-end OpenAI Responses surface over ASGI with a fake provider."""

from __future__ import annotations

import json
from textwrap import dedent

import httpx
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
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


def _client(fake: FakeLLM):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
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
        etype = data = None
        for line in block.strip().splitlines():
            if line.startswith("event: "):
                etype = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if etype and data is not None:
            out.append((etype, data))
    return out


async def test_non_streaming_response_object():
    async with _client(FakeLLM()) as client:
        resp = await client.post("/v1/responses", json={"model": "e", "input": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    message = next(o for o in body["output"] if o["type"] == "message")
    assert message["content"][0] == {
        "type": "output_text",
        "text": "synthesized answer",
        "annotations": [],
    }
    assert body["usage"]["output_tokens"] == 30


async def test_streaming_events():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "e", "input": "hi", "stream": True}
        )
    events = _events(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "response.created"
    assert "response.output_text.delta" in kinds
    assert kinds[-1] == "response.completed"
    # sequence_number is monotonic and unique
    seqs = [d["sequence_number"] for _, d in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    text = "".join(d["delta"] for k, d in events if k == "response.output_text.delta")
    assert text == "synthesized answer"


async def test_input_item_list_and_tool_call():
    fake = FakeLLM(tool_calls=({"id": "call_1", "name": "get_weather", "arguments": '{"c":"SF"}'},))
    async with _client(fake) as client:
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "e",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "weather?"}],
                    }
                ],
                "tools": [
                    {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
                ],
            },
        )
    body = resp.json()
    call = next(o for o in body["output"] if o["type"] == "function_call")
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"c": "SF"}


async def test_previous_response_id_is_400():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "e", "input": "hi", "previous_response_id": "resp_x"}
        )
    assert resp.status_code == 400
    assert "stateless" in resp.json()["error"]["message"]


async def test_mcp_tool_is_rejected():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses",
            json={"model": "e", "input": "hi", "tools": [{"type": "mcp", "server_label": "x"}]},
        )
    # ensemble "e" synthesizes on the chat API, which cannot forward remote MCP -> clean 400
    assert resp.status_code == 400
    assert "MCP" in resp.json()["error"]["message"]


_MCP_CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
      resp: { model: openai/gpt-5, api: responses }
    ensembles:
      forward:
        members: [{ llm: a }]
        synthesizer: { llm: resp }
    prompts: {}
""")


async def test_mcp_tool_forwarded_when_synth_supports_it():
    fake = FakeLLM()
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(_MCP_CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    mcp_tool = {"type": "mcp", "server_label": "x", "server_url": "https://mcp.example/sse"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "forward", "input": "hi", "tools": [mcp_tool]}
        )
    # a Responses-API synthesizer that supports remote MCP: forwarded opaquely, no 400
    assert resp.status_code == 200
    assert fake.streams[0].params["tools"] == [mcp_tool]
