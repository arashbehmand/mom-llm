"""Provider prompt caching: pure breakpoint/key planning + pipeline wiring."""

from __future__ import annotations

import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.prompt_caching import (
    inject_anthropic_cache,
    is_anthropic_family,
    openai_prompt_cache_key,
)
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.testing import FakeLLM, ManualClock


def _catalog(text: str):
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def test_is_anthropic_family():
    assert is_anthropic_family("anthropic/claude-sonnet-5")
    assert is_anthropic_family("vertex_ai/claude-3")
    assert not is_anthropic_family("openai/gpt-5")


def test_openai_cache_key_is_prefix_affine():
    a = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}]
    b = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}]
    assert openai_prompt_cache_key(a) == openai_prompt_cache_key(b)  # volatile tail excluded
    c = [{"role": "system", "content": "other"}, {"role": "user", "content": "q1"}]
    assert openai_prompt_cache_key(a) != openai_prompt_cache_key(c)  # different stable prefix


def test_inject_anthropic_cache_marks_stable_prefix_without_mutating():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "now do X"},
    ]
    out = inject_anthropic_cache(messages, ttl="5m")
    sys_blocks = out[0]["content"]
    assert isinstance(sys_blocks, list)
    assert sys_blocks[-1]["cache_control"] == {"type": "ephemeral"}
    # the message before the last (end of stable history) is also a breakpoint
    assert out[-2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert messages[0]["content"] == "you are helpful"  # original untouched


def test_inject_anthropic_cache_extended_ttl():
    out = inject_anthropic_cache(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], ttl="1h"
    )
    assert out[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_synth_messages_get_cache_control():
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
          c: { model: anthropic/claude-sonnet-5 }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: c }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    fake = FakeLLM()
    await collect(run_ensemble(plan, PipelineDeps(client=fake, clock=ManualClock())))
    synth_messages = fake.streams[0].messages
    assert any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in m["content"])
        for m in synth_messages
    )


async def test_openai_synth_gets_prompt_cache_key():
    catalog = _catalog(
        """
        version: 2
        llms:
          a: { model: openai/a }
        ensembles:
          e:
            members: [{ llm: a }]
            synthesizer: { llm: a }
        """
    )
    ir = ChatRequestIR(model="e", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    fake = FakeLLM()
    await collect(run_ensemble(plan, PipelineDeps(client=fake, clock=ManualClock())))
    assert "prompt_cache_key" in fake.streams[0].params
