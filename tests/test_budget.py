"""Daily-budget alarm middleware: a soft ``X-MoM-Budget: exceeded`` header, never a block."""

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
    server: {{ auth: none }}
    {budgets}
    llms:
      a: {{ model: openai/a }}
    ensembles:
      e:
        members: [{{ llm: a }}]
        synthesizer: {{ llm: a }}
""")


class _StubReader:
    """A metrics reader returning a fixed daily cost; records the aggregate windows it was asked."""

    def __init__(self, cost_usd: float) -> None:
        self.cost_usd = cost_usd
        self.starts: list[float | None] = []

    async def aggregate(
        self, *, start: float | None = None, end: float | None = None, ensemble: str | None = None
    ) -> dict[str, object]:
        self.starts.append(start)
        return {"cost_usd": self.cost_usd, "calls": 1}


def _client(reader: _StubReader | None, *, budgets: str):
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG.format(budgets=budgets))))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        metrics_reader=reader,
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _chat(content: str = "hi", **extra: object) -> dict[str, object]:
    return {"model": "e", "messages": [{"role": "user", "content": content}], **extra}


async def test_over_budget_sets_alarm_header():
    reader = _StubReader(cost_usd=5.0)  # today's spend well over the 1.0 budget
    async with _client(reader, budgets="budgets: { daily_usd: 1.0 }") as client:
        resp = await client.post("/v1/chat/completions", json=_chat())
    assert resp.status_code == 200  # soft alarm: the request is still served
    assert resp.headers.get("X-MoM-Budget") == "exceeded"
    assert len(reader.starts) == 1  # queried the metrics reader once
    assert reader.starts[0] is not None  # for a since-midnight window


async def test_under_budget_has_no_alarm_header():
    reader = _StubReader(cost_usd=0.25)
    async with _client(reader, budgets="budgets: { daily_usd: 1.0 }") as client:
        resp = await client.post("/v1/chat/completions", json=_chat())
    assert resp.status_code == 200
    assert "X-MoM-Budget" not in resp.headers


async def test_no_budget_configured_skips_the_reader_entirely():
    reader = _StubReader(cost_usd=9999.0)  # would trip any budget — but none is set
    async with _client(reader, budgets="") as client:
        resp = await client.post("/v1/chat/completions", json=_chat())
    assert resp.status_code == 200
    assert "X-MoM-Budget" not in resp.headers
    assert reader.starts == []  # short-circuited before any query


async def test_alarm_header_also_set_on_streaming_responses():
    reader = _StubReader(cost_usd=5.0)
    async with _client(reader, budgets="budgets: { daily_usd: 1.0 }") as client:
        resp = await client.post("/v1/chat/completions", json=_chat(stream=True))
    assert resp.status_code == 200
    assert resp.headers.get("X-MoM-Budget") == "exceeded"  # pure-ASGI header survives streaming
