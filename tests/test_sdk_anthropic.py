"""SDK-in-the-loop contract tests: the official Anthropic SDK parses our real /v1/messages streams.

Drives the FastAPI app over an in-process ASGI transport with a scripted ``FakeLLM`` — a Messages
wire-format regression surfaces as an SDK parse error rather than silent shape drift.
"""

from __future__ import annotations

from textwrap import dedent

import anthropic
import httpx
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
          a: { model: anthropic/a }
          b: { model: anthropic/b }
        ensembles:
          e:
            members: [{ llm: a }, { llm: b }]
            synthesizer: { llm: a }
    """)
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def _sdk(client: object | None = None) -> anthropic.AsyncAnthropic:
    container = Container(
        settings=Settings(_env_file=None),
        catalog=_catalog(),
        client=client or FakeLLM(),  # type: ignore[arg-type]
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return anthropic.AsyncAnthropic(api_key="test", base_url="http://test", http_client=http)


async def test_anthropic_sdk_parses_message():
    client = _sdk()
    msg = await client.messages.create(
        model="e", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.content[0].text == "synthesized answer"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.input_tokens >= 0
    await client.close()


async def test_anthropic_sdk_parses_streaming():
    client = _sdk()
    async with client.messages.stream(
        model="e", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        async for _event in stream:
            pass
        final = await stream.get_final_message()
    assert final.content[0].text == "synthesized answer"
    assert final.stop_reason == "end_turn"
    await client.close()
