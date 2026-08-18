"""Metrics: the pipeline records per-call rows and /v1/metrics/usage aggregates them."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.runtime.settings import Settings
from mom.store.metrics import MetricsRecorder, MetricsStore
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


async def test_metrics_recorded_and_aggregated(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "m.db")
    recorder = MetricsRecorder(store)
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        metrics=recorder,
        metrics_reader=store,
    )
    app = create_app(container=container)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        await recorder.flush()  # drain queued metrics to the store

        # 2 fan-out members + 1 synthesis = 3 recorded calls
        agg = await store.aggregate()
        assert agg["calls"] == 3

        usage = await client.get("/v1/metrics/usage")
        assert usage.status_code == 200
        body = usage.json()
        assert body["calls"] == 3
        assert body["completion_tokens"] == 30  # 2 members (5 each) + synth (20)
    await store.close()
