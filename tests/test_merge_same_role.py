"""`defaults.call.merge_same_role`: joining consecutive same-role turns before a call goes out."""

from __future__ import annotations

from textwrap import dedent

import yaml

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.synthesis import merge_same_role
from mom.engine.plan import resolve_plan


def test_consecutive_user_turns_become_one():
    merged = merge_same_role(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "the question"},
            {"role": "user", "content": "the candidates"},
            {"role": "user", "content": "the synthesis prompt"},
        ]
    )
    assert [m["role"] for m in merged] == ["system", "user"]
    assert merged[1]["content"] == "the question\n\nthe candidates\n\nthe synthesis prompt"


def test_turns_that_alternate_are_left_alone():
    conversation = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    assert merge_same_role(conversation) == conversation


def test_multipart_content_keeps_its_parts():
    """An image, or a cache_control breakpoint already placed on a block, must survive the join."""
    merged = merge_same_role(
        [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]},
            {"role": "user", "content": [{"type": "text", "text": "what is this?"}]},
            {"role": "user", "content": "and be brief"},
        ]
    )
    assert len(merged) == 1
    assert [part["type"] for part in merged[0]["content"]] == ["image_url", "text", "text"]
    assert merged[0]["content"][-1] == {"type": "text", "text": "and be brief"}


def test_tool_plumbing_is_never_merged():
    """A tool result and an assistant turn carrying tool_calls each answer for one call id."""
    calls = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function"}]},
        {"role": "tool", "content": "result a", "tool_call_id": "a"},
        {"role": "tool", "content": "result b", "tool_call_id": "b"},
    ]
    assert merge_same_role(calls) == calls


def test_messages_from_different_named_speakers_stay_apart():
    named = [
        {"role": "user", "content": "from ana", "name": "ana"},
        {"role": "user", "content": "from bo", "name": "bo"},
    ]
    assert merge_same_role(named) == named


CONFIG = dedent("""
    version: 2
    llms:
      a: { model: openai/a }
    ensembles:
      e:
        members: [{ llm: a }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _plan(*, on: bool):
    text = (
        CONFIG
        if not on
        else CONFIG.replace(
            "version: 2", "version: 2\ndefaults: { call: { merge_same_role: true } }"
        )
    )
    catalog = resolve_catalog(Config.model_validate(yaml.safe_load(text)))
    ir = ChatRequestIR(
        model="e",
        messages=(
            MessageIR(role="user", content="first"),
            MessageIR(role="user", content="second"),
        ),
    )
    return resolve_plan(catalog, ir)


def test_the_knob_is_off_by_default():
    plan = _plan(on=False)
    assert plan.merge_same_role is False
    assert [m["content"] for m in plan.client_messages] == ["first", "second"]


def test_turning_it_on_merges_what_members_are_sent():
    plan = _plan(on=True)
    assert plan.merge_same_role is True
    assert [m["content"] for m in plan.client_messages] == ["first\n\nsecond"]


async def _synthesis_call(*, on: bool):
    """Run a real ensemble and return the CallSpec the synthesizer was given."""
    from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
    from mom.testing import FakeLLM, ManualClock

    client = FakeLLM()
    plan = _plan(on=on)
    await collect(run_ensemble(plan, PipelineDeps(client=client, clock=ManualClock())))
    return client.streams[-1]


async def test_the_synthesizer_receives_one_turn_when_merging_is_on():
    """The bug this exists for: history, candidate block and synthesis prompt are three user
    turns in a row, and an upstream that keeps only the last of them drops the question and every
    candidate answer — the synthesizer then answers from the system prompt alone."""
    spec = await _synthesis_call(on=True)
    assert [m["role"] for m in spec.messages] == ["user"]
    body = spec.messages[0]["content"]
    assert "first" in body  # the question survived
    assert "second" in body
    assert "RESPONSE 1 of 1" in body  # so did the candidate block
    assert "synthesize" in body  # and the synthesis prompt


async def test_the_synthesizer_sees_the_turns_as_written_by_default():
    spec = await _synthesis_call(on=False)
    assert [m["role"] for m in spec.messages] == ["user", "user", "user", "user"]
