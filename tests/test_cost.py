"""Cost engine: cache-aware per-call cost and pipeline accumulation."""

from __future__ import annotations

import pytest
import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.cost import Pricing, compute_cost
from mom.domain.ports import Completion, CompletionChunk
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.results import Usage
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.testing import FakeLLM, ManualClock


def test_compute_cost_is_cache_aware():
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        reasoning_tokens=50,
        cached_prompt_tokens=200,
        cache_write_tokens=100,
    )
    pricing = Pricing(
        input_per_1m=3.0,
        output_per_1m=6.0,
        reasoning_per_1m=6.0,
        cache_read_per_1m=0.3,
        cache_write_per_1m=3.75,
    )
    # billable input = 1000 - 200 cached; cached at read rate, writes billed additively.
    expected = (800 * 3.0 + 200 * 0.3 + 100 * 3.75 + 500 * 6.0 + 50 * 6.0) / 1_000_000
    assert compute_cost(usage, pricing) == pytest.approx(expected)


def test_compute_cost_no_pricing_is_zero():
    assert compute_cost(Usage(prompt_tokens=100, completion_tokens=100), None) == 0.0


async def test_pipeline_accumulates_member_and_synth_cost():
    text = """
        version: 2
        llms:
          a: { model: openai/a, pricing: { input_per_1m: 1.0, output_per_1m: 2.0 } }
          b: { model: openai/b, pricing: { input_per_1m: 1.0, output_per_1m: 2.0 } }
        ensembles:
          e:
            members: [{ llm: a }, { llm: b }]
            synthesizer: { llm: a }
    """
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(text)))
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), request_id="req-1")
    result = await collect(run_ensemble(plan, deps))
    # members a,b: (10*1 + 5*2)/1e6 = 2e-5 each; synth a: (50*1 + 20*2)/1e6 = 9e-5
    assert result.total_cost_usd == pytest.approx((2e-5 * 2) + 9e-5)
    assert result.total_cost_usd > 0


class _CostingLLM(FakeLLM):
    """A fake whose calls report an adapter-computed cost (as the litellm cost map would)."""

    async def complete(self, spec):
        self.completions.append(spec)
        return Completion(
            content=f"reply from {spec.llm_name}",
            reasoning=None,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            cost_usd=0.001,
        )

    async def stream(self, spec):
        self.streams.append(spec)
        yield CompletionChunk(content="synth")
        yield CompletionChunk(
            finish_reason="stop",
            usage=Usage(prompt_tokens=50, completion_tokens=20),
            cost_usd=0.005,
        )


async def test_pipeline_falls_back_to_adapter_cost_without_config_pricing():
    text = """
        version: 2
        llms:
          a: { model: openai/a }
          b: { model: openai/b }
        ensembles:
          e:
            members: [{ llm: a }, { llm: b }]
            synthesizer: { llm: a }
    """
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(text)))
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    deps = PipelineDeps(client=_CostingLLM(), clock=ManualClock())
    result = await collect(run_ensemble(plan, deps))
    # no config pricing -> adapter cost used: 2 members * 0.001 + synth 0.005
    assert result.total_cost_usd == pytest.approx(0.007)
