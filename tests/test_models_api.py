"""Rich /v1/models metadata, the Anthropic list shape, and /v1/model/info."""

from __future__ import annotations

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
      a: { model: openai/a, search: {} }
      b: { model: openai/b, capabilities: { vision: false } }
      syn: { model: openai/syn, capabilities: { context_length: 200000, max_output_tokens: 16000 } }
    ensembles:
      tiered:
        description: "A tiered ensemble."
        effort_tiers: [low, high]
        default_tier: low
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: syn }
""")


def _client():
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_openai_models_rich_metadata():
    async with _client() as client:
        resp = await client.get("/v1/models")
    model = resp.json()["data"][0]
    assert model["id"] == "tiered"
    assert model["context_length"] == 200000
    assert model["reasoning_effort_levels"] == ["low", "high"]
    assert "web_search" in model["supported_parameters"]
    assert "reasoning_effort" in model["supported_parameters"]
    mom = model["mom"]
    assert mom["supports_web_search"] is True  # member `a` declares search
    assert mom["supports_vision"] is True  # `a` is vision-capable (only `b` is not)
    assert mom["members"] == ["a", "b"]
    assert mom["remote_mcp"] is False
    # Flat booleans some OpenAI-compatible clients (e.g. lobe-chat) read directly off the list
    # entry instead of the nested `mom` block.
    assert model["search"] is True
    assert model["vision"] is True
    assert model["functionCall"] == mom["supports_tools"]
    assert model["reasoning"] == mom["supports_reasoning"]


async def test_get_single_model():
    async with _client() as client:
        resp = await client.get("/v1/models/tiered")
    assert resp.status_code == 200
    assert resp.json()["id"] == "tiered"


async def test_unknown_model_404():
    async with _client() as client:
        resp = await client.get("/v1/models/ghost")
    assert resp.status_code == 404


async def test_anthropic_list_shape_on_header():
    async with _client() as client:
        resp = await client.get("/v1/models", headers={"x-api-key": "whatever"})
    body = resp.json()
    assert body["has_more"] is False
    assert body["data"][0] == {
        "type": "model",
        "id": "tiered",
        "display_name": "tiered",
        "created_at": body["data"][0]["created_at"],
    }


async def test_model_info_litellm_shape():
    async with _client() as client:
        resp = await client.get("/v1/model/info")
    entry = resp.json()["data"][0]
    assert entry["model_name"] == "tiered"
    assert entry["litellm_params"]["model"] == "mom/tiered"
    assert entry["model_info"]["max_input_tokens"] == 200000
    assert entry["model_info"]["supports_web_search"] is True
