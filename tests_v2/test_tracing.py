"""Tracing: the pipeline emits one observation per member and per synthesis."""

from __future__ import annotations

from textwrap import dedent

import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.testing import FakeLLM, ManualClock, RecordingTracer


CONFIG = dedent("""
    version: 2
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


async def test_pipeline_emits_observations():
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))
    plan = resolve_plan(catalog, ChatRequestIR(model="e", messages=(MessageIR("user", "hi"),)))
    tracer = RecordingTracer()
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), tracer=tracer, request_id="req-1")
    await collect(run_ensemble(plan, deps))

    roles = [o["role"] for o in tracer.observations]
    assert roles.count("fanout") == 2
    assert roles.count("synthesis") == 1
    synth = next(o for o in tracer.observations if o["role"] == "synthesis")
    assert synth["output"] == "synthesized answer"
    assert all(o["request_id"] == "req-1" for o in tracer.observations)
