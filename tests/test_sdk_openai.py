"""SDK-in-the-loop contract tests: the official OpenAI SDK parses our real ASGI byte streams.

These drive the actual FastAPI app over an in-process ASGI transport (no network) with a scripted
``FakeLLM``, so a wire-format regression surfaces as an SDK parse error, not a silent shape drift.
"""

from __future__ import annotations

from textwrap import dedent

import httpx
import openai
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


def _catalog():
    text = dedent("""
        version: 2
        server: { auth: none }
        llms:
          a: { model: openai/a }
          b: { model: openai/b }
        ensembles:
          e:
            members: [{ llm: a }, { llm: b }]
            synthesizer: { llm: a }
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def _sdk(client: object | None = None) -> openai.AsyncOpenAI:
    container = Container(
        settings=Settings(_env_file=None),
        catalog=_catalog(),
        client=client or FakeLLM(),  # type: ignore[arg-type]
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return openai.AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)


async def test_openai_sdk_parses_non_streaming():
    client = _sdk()
    resp = await client.chat.completions.create(
        model="e", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.content == "synthesized answer"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens > 0
    await client.close()


async def test_openai_sdk_parses_streaming_with_usage():
    client = _sdk()
    stream = await client.chat.completions.create(
        model="e",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text, usage = "", None
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
        if chunk.usage:
            usage = chunk.usage
    assert "synthesized answer" in text
    assert usage is not None
    assert usage.total_tokens > 0
    await client.close()


async def test_openai_sdk_parses_tool_calls():
    fake = FakeLLM(
        finish_reason="tool_calls",
        tool_calls=({"id": "call_1", "name": "get_weather", "arguments": '{"city":"Paris"}'},),
    )
    client = _sdk(fake)
    resp = await client.chat.completions.create(
        model="e",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="required",
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert call.function.arguments == '{"city":"Paris"}'
    await client.close()


async def test_openai_sdk_parses_responses_api():
    client = _sdk()
    resp = await client.responses.create(model="e", input="hi")
    assert resp.output_text == "synthesized answer"
    await client.close()
