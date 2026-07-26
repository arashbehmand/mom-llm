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
from mom.domain.ports import CompletionChunk
from mom.domain.results import Usage
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


def _client(
    fake: FakeLLM,
    *,
    auth: str = "none",
    settings: Settings | None = None,
    estimator: object | None = None,
):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG.format(auth=auth))))
    container = Container(
        settings=settings or Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
        token_estimator=estimator,  # type: ignore[arg-type]
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class _StubEstimator:
    """A model-aware token estimator double; records the (model, messages) it was asked to count."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def count(self, *, model: str, messages: list[dict[str, object]]) -> int:
        self.calls.append((model, messages))
        return self.value


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
    async with _client(FakeLLM()) as client:  # no estimator -> chars/4 fallback
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "hello world " * 20}],
            },
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0


async def test_count_tokens_uses_model_aware_estimator():
    estimator = _StubEstimator(4242)
    async with _client(FakeLLM(), estimator=estimator) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "e", "messages": [{"role": "user", "content": "hello world"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 4242
    # counted against the ensemble's *synthesizer* model, not the ensemble name.
    assert len(estimator.calls) == 1
    assert estimator.calls[0][0] == "openai/a"


async def test_count_tokens_falls_back_for_unknown_ensemble():
    estimator = _StubEstimator(4242)
    async with _client(FakeLLM(), estimator=estimator) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "ghost", "messages": [{"role": "user", "content": "hello world " * 20}]},
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] != 4242  # unresolved model -> chars/4, estimator untouched
    assert estimator.calls == []


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


async def test_streaming_message_start_reports_input_tokens():
    async with _client(FakeLLM()) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "e",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hello world " * 20}],
            },
        )
    events = _events(resp.text)
    start = next(d for k, d in events if k == "message_start")
    # Best-effort transcript estimate (~chars/4), not the pipeline's 0 default.
    assert start["message"]["usage"]["input_tokens"] > 0


class ReasoningFakeLLM(FakeLLM):
    """A synthesizer that streams thinking (reasoning) deltas before its text answer."""

    async def stream(self, spec):
        self.streams.append(spec)
        yield CompletionChunk(reasoning="let me think")
        yield CompletionChunk(reasoning=" carefully")
        yield CompletionChunk(content="final answer")
        yield CompletionChunk(
            finish_reason="stop", usage=Usage(prompt_tokens=50, completion_tokens=20)
        )


async def test_streaming_reasoning_emits_thinking_block():
    async with _client(ReasoningFakeLLM()) as client:
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
    kinds = [k for k, _ in events]
    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"

    # A thinking content block opens and carries thinking_delta text.
    thinking_start = next(
        d
        for k, d in events
        if k == "content_block_start" and d["content_block"]["type"] == "thinking"
    )
    assert thinking_start["content_block"] == {"type": "thinking", "thinking": ""}
    thinking_text = "".join(
        d["delta"]["thinking"]
        for k, d in events
        if k == "content_block_delta" and d["delta"]["type"] == "thinking_delta"
    )
    assert thinking_text == "let me think carefully"

    # The thinking block is signed (signature_delta) then closed before the text block opens, and
    # indexes stay monotonic.
    signature = next(
        d
        for k, d in events
        if k == "content_block_delta" and d["delta"]["type"] == "signature_delta"
    )
    assert signature["index"] == thinking_start["index"]
    assert signature["delta"]["signature"]  # opaque, deterministic, non-empty
    text_start = next(
        d for k, d in events if k == "content_block_start" and d["content_block"]["type"] == "text"
    )
    assert thinking_start["index"] < text_start["index"]
    order = [(k, d.get("index")) for k, d in events if k.startswith("content_block")]
    assert order == [
        ("content_block_start", thinking_start["index"]),
        ("content_block_delta", thinking_start["index"]),  # thinking "let me think"
        ("content_block_delta", thinking_start["index"]),  # thinking " carefully"
        ("content_block_delta", thinking_start["index"]),  # signature_delta
        ("content_block_stop", thinking_start["index"]),
        ("content_block_start", text_start["index"]),
        ("content_block_delta", text_start["index"]),
        ("content_block_stop", text_start["index"]),
    ]
    text = "".join(
        d["delta"]["text"]
        for k, d in events
        if k == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text == "final answer"


def test_thinking_signature_is_deterministic():
    from mom.api.encoders.anthropic import _thinking_signature

    # Same thinking text -> same opaque signature; different text -> different signature.
    assert _thinking_signature("let me think") == _thinking_signature("let me think")
    assert _thinking_signature("a") != _thinking_signature("b")
    assert _thinking_signature("hello")  # non-empty
