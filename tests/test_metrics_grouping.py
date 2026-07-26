"""Metrics grouping: store.aggregate_by(...) and GET /v1/metrics/usage?by=..."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx
import pytest
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.store.metrics import CallMetric, MetricsStore
from mom.testing import FakeLLM, ManualClock, SequentialIds


def _metric(**overrides: object) -> CallMetric:
    base: dict[str, object] = {
        "request_id": "req-1",
        "ts": 1000.0,
        "ensemble": "bmom",
        "llm": "gpt",
        "model": "openai/gpt-x",
        "role": "fanout",
        "status": "ok",
    }
    base.update(overrides)
    return CallMetric(**base)  # type: ignore[arg-type]


@pytest.fixture
async def store(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "metrics.db")
    yield store
    await store.close()


# ---------------------------------------------------------------------------------------------
# store.aggregate_by
# ---------------------------------------------------------------------------------------------
async def test_group_by_member(store: MetricsStore):
    await store.insert_many(
        [
            _metric(llm="a", prompt_tokens=10, cost_usd=0.01),
            _metric(llm="a", prompt_tokens=20, cost_usd=0.02),
            _metric(llm="b", prompt_tokens=5, cost_usd=0.03),
        ]
    )
    rows = await store.aggregate_by("member")
    by_member = {row["member"]: row for row in rows}
    assert by_member["a"]["calls"] == 2
    assert by_member["a"]["prompt_tokens"] == 30
    assert abs(by_member["a"]["cost_usd"] - 0.03) < 1e-9
    assert by_member["b"]["calls"] == 1


async def test_group_by_turn_type(store: MetricsStore):
    await store.insert_many(
        [
            _metric(turn_type="ensemble"),
            _metric(turn_type="ensemble"),
            _metric(turn_type="relay", role="synthesis", status="ok"),
        ]
    )
    rows = await store.aggregate_by("turn_type")
    assert {row["turn_type"]: row["calls"] for row in rows} == {"ensemble": 2, "relay": 1}


async def test_group_by_day(store: MetricsStore):
    day1 = 1_700_000_000.0  # 2023-11-14T22:13:20Z
    day2 = day1 + 86_400.0  # next UTC day
    await store.insert_many([_metric(ts=day1), _metric(ts=day1), _metric(ts=day2)])
    rows = await store.aggregate_by("day")
    assert len(rows) == 2
    assert sum(row["calls"] for row in rows) == 3
    assert all(isinstance(row["day"], str) for row in rows)  # date() -> 'YYYY-MM-DD'


async def test_group_by_respects_time_and_ensemble_window(store: MetricsStore):
    await store.insert_many(
        [
            _metric(ts=10.0, llm="a", ensemble="x"),
            _metric(ts=100.0, llm="a", ensemble="x"),
            _metric(ts=100.0, llm="a", ensemble="y"),
        ]
    )
    rows = await store.aggregate_by("member", start=50.0, ensemble="x")
    # Only the ts=100, ensemble=x call survives the window.
    assert len(rows) == 1
    assert rows[0]["member"] == "a"
    assert rows[0]["calls"] == 1


async def test_group_by_unknown_dimension_raises(store: MetricsStore):
    with pytest.raises(ValueError, match="unknown grouping dimension"):
        await store.aggregate_by("provider")


# ---------------------------------------------------------------------------------------------
# GET /v1/metrics/usage?by=...
# ---------------------------------------------------------------------------------------------
CONFIG = dedent("""
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


def _container(store: MetricsStore) -> Container:
    return Container(
        settings=Settings(_env_file=None),
        catalog=resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG))),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        metrics_reader=store,
    )


async def test_usage_endpoint_grouped_by_member(store: MetricsStore):
    await store.insert_many([_metric(llm="a"), _metric(llm="a"), _metric(llm="b")])
    app = create_app(container=_container(store))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/metrics/usage", params={"by": "member"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["by"] == "member"
    assert {g["member"]: g["calls"] for g in body["groups"]} == {"a": 2, "b": 1}


async def test_usage_endpoint_ungrouped_is_unchanged(store: MetricsStore):
    await store.insert_many([_metric(), _metric()])
    app = create_app(container=_container(store))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/metrics/usage")
    assert resp.status_code == 200
    assert resp.json()["calls"] == 2  # no "groups" key on the plain aggregate


async def test_usage_endpoint_rejects_unknown_dimension(store: MetricsStore):
    app = create_app(container=_container(store))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/metrics/usage", params={"by": "bogus"})
    assert resp.status_code == 422  # FastAPI validates the Literal
