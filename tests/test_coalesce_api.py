"""End-to-end coalescing through ``/v1/chat/completions``: two concurrent, identical requests
against a running app share one fan-out + synthesis instead of paying for it twice.
"""

from __future__ import annotations

import asyncio
from textwrap import dedent

import httpx
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.runtime.wiring import build_coalesce
from mom.testing import FakeLLM, ManualClock, SequentialIds


# `a` sleeps briefly on every fan-out call — long enough to give a second, concurrently-fired
# request time to reach the coalescing registry while the first is still in flight, short enough
# to keep the test fast.
_CONFIG = dedent("""
    version: 2
    server: { auth: none, dedupe: { enabled: true, orphan_grace: 5s } }
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

_DISABLED_CONFIG = _CONFIG.replace("dedupe: { enabled: true, orphan_grace: 5s }", "dedupe: {}")


def _client(fake: FakeLLM, *, config: str = _CONFIG) -> httpx.AsyncClient:
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(config)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
        # The same wiring `mom serve` uses (runtime/wiring.py::build_container) — these router
        # tests build a Container by hand, so this has to be called explicitly here too, or
        # `server.dedupe.enabled` would silently be a no-op.
        coalesce=build_coalesce(catalog),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _post(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
    )


async def test_two_concurrent_identical_requests_coalesce_onto_one_run():
    fake = FakeLLM(delays={"a": 0.05})
    async with _client(fake) as client:
        resp_1, resp_2 = await asyncio.wait_for(
            asyncio.gather(_post(client), _post(client)), timeout=5.0
        )
    assert resp_1.status_code == resp_2.status_code == 200
    # One real fan-out (2 members) and one real synthesis — not two of each.
    assert len(fake.completions) == 2
    assert len(fake.streams) == 1
    body_1, body_2 = resp_1.json(), resp_2.json()
    assert body_1["choices"][0]["message"]["content"] == body_2["choices"][0]["message"]["content"]
    # Both responses point at the SAME request — the leader's — via X-Request-Id.
    assert resp_1.headers["x-request-id"] == resp_2.headers["x-request-id"]
    # Exactly one of the two carries the coalesced marker; the other is the leader.
    coalesced_flags = {
        resp_1.headers.get("x-mom-coalesced"),
        resp_2.headers.get("x-mom-coalesced"),
    }
    assert coalesced_flags == {"1", None}


async def test_sequential_identical_requests_do_not_coalesce():
    # No overlap in time -> the first request's run completes (and is dropped from the registry)
    # long before the second even starts, so this must produce two independent runs.
    fake = FakeLLM()
    async with _client(fake) as client:
        resp_1 = await _post(client)
        resp_2 = await _post(client)
    assert resp_1.status_code == resp_2.status_code == 200
    assert len(fake.completions) == 4  # 2 members x 2 independent requests
    assert len(fake.streams) == 2
    assert "x-mom-coalesced" not in resp_1.headers
    assert "x-mom-coalesced" not in resp_2.headers
    assert resp_1.headers["x-request-id"] != resp_2.headers["x-request-id"]


async def test_dedupe_disabled_by_default_never_coalesces_even_when_concurrent():
    fake = FakeLLM(delays={"a": 0.05})
    async with _client(fake, config=_DISABLED_CONFIG) as client:
        resp_1, resp_2 = await asyncio.wait_for(
            asyncio.gather(_post(client), _post(client)), timeout=5.0
        )
    assert resp_1.status_code == resp_2.status_code == 200
    assert len(fake.completions) == 4  # each request ran its own independent fan-out
    assert len(fake.streams) == 2
    assert resp_1.headers["x-request-id"] != resp_2.headers["x-request-id"]
