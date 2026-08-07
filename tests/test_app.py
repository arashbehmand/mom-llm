"""App-factory smoke tests over the in-process ASGI transport (no network)."""

from __future__ import annotations

from textwrap import dedent

import httpx
import yaml

from mom import __version__
from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


async def test_health_endpoint():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    # No container was supplied (lifespan never ran over the ASGI transport) -> no metrics sink
    # to read `dropped` from, so the field is simply absent, not a 500.
    assert "metrics_dropped" not in body


_CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
    ensembles:
      e:
        members: [{ llm: a }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


class _RecorderWithDrops:
    """A minimal MetricsSink whose `dropped` health should surface (`getattr`, not a widened
    Protocol member — most real MetricsSink implementations may not have one)."""

    def __init__(self, dropped: int) -> None:
        self.dropped = dropped

    def record(self, metric: object) -> None:
        return None


async def test_health_reports_metrics_dropped_when_the_container_has_one():
    container = Container(
        settings=Settings(_env_file=None),
        catalog=resolve_catalog(Config.model_validate(yaml.safe_load(_CONFIG))),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        metrics=_RecorderWithDrops(dropped=7),
    )
    app = create_app(container=container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.json()["metrics_dropped"] == 7


def test_create_app_is_pure_construction():
    # Building the app must be pure construction (the container is built in the lifespan).
    app = create_app()
    assert app.title == "MoM — Mixture of Models"
    # Recent Starlette nests included routers (`_IncludedRouter`) instead of flattening them into
    # app.routes, so assert route existence via the generated OpenAPI schema (version-stable).
    assert "/v1/chat/completions" in app.openapi()["paths"]
