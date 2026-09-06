"""End-to-end OpenAI Responses surface over ASGI with a fake provider."""

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


_HEARTBEAT_CONFIG = dedent("""
    version: 2
    server: { auth: none, stream_heartbeat: 20ms }
    llms:
      a: { model: openai/a }
      slow: { model: openai/slow }
    ensembles:
      e:
        members: [{ llm: a }, { llm: slow }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _heartbeat_client(fake: FakeLLM):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(_HEARTBEAT_CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_streaming_response_carries_anti_buffering_headers():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "e", "input": "hi", "stream": True}
        )
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"


async def test_streaming_now_carries_a_heartbeat_previously_missing_on_this_surface():
    """Regression: this surface had NO ``with_heartbeat`` wiring at all before this fix — only
    ``chat.py`` did — so a slow member here could idle-timeout a client with zero keepalive."""
    fake = FakeLLM(delays={"slow": 0.05})
    async with _heartbeat_client(fake) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "e", "input": "hi", "stream": True}
        )
    assert ": keepalive" in resp.text


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


def test_namespace_tools_are_flattened():
    # Codex groups function tools under a `namespace` container. The nested function tools must be
    # flattened (not 400-ed like an unknown type); non-function entries inside are ignored.
    from mom.api.translate_responses import _tools

    specs, web_search, mcp = _tools(
        [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "grouped tools",
                "tools": [
                    {"type": "function", "name": "close_agent", "parameters": {"type": "object"}},
                    {"type": "function", "name": "spawn_agent"},
                    {"type": "something_else"},  # ignored
                ],
            },
            {"type": "function", "name": "exec_command"},
        ]
    )
    assert [s.name for s in specs] == ["close_agent", "spawn_agent", "exec_command"]
    assert web_search is False
    assert mcp == ()


def test_genuinely_unknown_tool_type_still_400s():
    from mom.api.translate_responses import _tools
    from mom.domain.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        _tools([{"type": "computer_use_preview"}])


class ReasoningFakeLLM(FakeLLM):
    """A synthesizer that streams a reasoning summary before its answer text."""

    async def stream(self, spec):
        self.streams.append(spec)
        yield CompletionChunk(reasoning="let me think")
        yield CompletionChunk(content="final answer")
        yield CompletionChunk(
            finish_reason="stop", usage=Usage(prompt_tokens=50, completion_tokens=20)
        )


async def test_non_streaming_reasoning_item_precedes_message():
    async with _client(ReasoningFakeLLM()) as client:
        resp = await client.post("/v1/responses", json={"model": "e", "input": "hi"})
    body = resp.json()
    assert [o["type"] for o in body["output"]] == ["reasoning", "message"]
    assert body["output"][0]["summary"][0]["text"] == "let me think"


async def test_streaming_reasoning_summary_events():
    async with _client(ReasoningFakeLLM()) as client:
        resp = await client.post(
            "/v1/responses", json={"model": "e", "input": "hi", "stream": True}
        )
    events = _events(resp.text)
    kinds = [k for k, _ in events]
    assert "response.reasoning_summary_text.delta" in kinds
    reasoning_text = "".join(
        d["delta"] for k, d in events if k == "response.reasoning_summary_text.delta"
    )
    assert reasoning_text == "let me think"
    # The reasoning item is listed before the message item in the completed response.
    output = next(d for k, d in events if k == "response.completed")["response"]["output"]
    assert [o["type"] for o in output] == ["reasoning", "message"]
    # sequence_number remains monotonic and unique across the added reasoning events.
    seqs = [d["sequence_number"] for _, d in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


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


_SHOW_WORK_CONFIG = dedent("""
    version: 2
    server: { auth: bearer }
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
        show_work: inline
    prompts:
      p: "synthesize"
""")


def _authed_client(fake: FakeLLM):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(_SHOW_WORK_CONFIG)))
    container = Container(
        settings=Settings(_env_file=None, MOM_API_TOKEN="secret-token"),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_show_work_inline_includes_progress_link_non_streaming():
    async with _authed_client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses",
            json={"model": "e", "input": "hi"},
            headers={"authorization": "Bearer secret-token"},
        )
    body = resp.json()
    request_id = resp.headers["x-request-id"]
    member_dump = next(o for o in body["output"] if o["type"] == "reasoning")
    text = member_dump["summary"][0]["text"]
    assert f"Progress: http://test/v1/progress/{request_id}?token=" in text
    assert "secret-token" not in text  # a scoped link token, not the gateway credential
    assert text.index("Progress:") < text.index("Model:")


async def test_show_work_inline_includes_progress_link_streaming():
    async with _authed_client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/responses",
            json={"model": "e", "input": "hi", "stream": True},
            headers={"authorization": "Bearer secret-token"},
        )
    request_id = resp.headers["x-request-id"]
    events = _events(resp.text)
    reasoning_text = "".join(
        d["delta"] for k, d in events if k == "response.reasoning_summary_text.delta"
    )
    assert f"Progress: http://test/v1/progress/{request_id}?token=" in reasoning_text
    assert "secret-token" not in reasoning_text  # a scoped link token, not the gateway credential


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
