"""Responses SSE encoder: no cross-item text bleed, unique item ids, clean failure finalization."""

from __future__ import annotations

import json

from mom.api.encoders.responses import build_response, encode_sse
from mom.domain.events import (
    AnswerDelta,
    Completed,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.results import EnsembleResult, ModelOutcome, Usage


async def _collect(events: list[StreamEvent], *, show_work: str = "off") -> list[dict]:
    async def gen():
        for event in events:
            yield event

    payloads: list[dict] = []
    async for block in encode_sse(
        gen(), response_id="resp-1", model="e", created=1, show_work=show_work
    ):
        payloads.extend(
            json.loads(line[len("data: ") :])
            for line in block.decode().splitlines()
            if line.startswith("data: ")
        )
    return payloads


def _of_type(payloads: list[dict], event_type: str) -> list[dict]:
    return [p for p in payloads if p.get("type") == event_type]


async def test_text_tool_text_does_not_concatenate_and_ids_are_unique():
    payloads = await _collect(
        [
            AnswerDelta(content="A"),
            ToolCallStarted(index=0, call_id="c0", name="fn"),
            ToolCallDelta(index=0, arguments_fragment="{}"),
            AnswerDelta(content="B"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ]
    )
    done = _of_type(payloads, "response.output_text.done")
    assert [d["text"] for d in done] == ["A", "B"]  # second item is "B", not "AB"
    assert len({d["item_id"] for d in done}) == 2  # distinct item ids per message item


async def test_reasoning_summary_precedes_answer_and_ids_stay_ordered():
    payloads = await _collect(
        [
            AnswerDelta(reasoning="think "),
            AnswerDelta(reasoning="more"),
            AnswerDelta(content="answer"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ]
    )
    # A reasoning output item opens before the message item, at a lower output_index.
    added = _of_type(payloads, "response.output_item.added")
    assert [a["item"]["type"] for a in added] == ["reasoning", "message"]
    assert added[0]["output_index"] < added[1]["output_index"]
    # The summary deltas carry the reasoning text, and a summary part is opened + closed.
    deltas = _of_type(payloads, "response.reasoning_summary_text.delta")
    assert "".join(d["delta"] for d in deltas) == "think more"
    assert _of_type(payloads, "response.reasoning_summary_part.added")
    summary_done = _of_type(payloads, "response.reasoning_summary_text.done")
    assert summary_done[0]["text"] == "think more"
    # The completed response lists the reasoning item first, with its summary populated.
    output = _of_type(payloads, "response.completed")[0]["response"]["output"]
    assert [o["type"] for o in output] == ["reasoning", "message"]
    assert output[0]["summary"] == [{"type": "summary_text", "text": "think more"}]
    # sequence_number stays monotonic and unique across the added reasoning events.
    seqs = [p["sequence_number"] for p in payloads]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


async def test_reasoning_only_response_still_emits_a_reasoning_item():
    payloads = await _collect(
        [
            AnswerDelta(reasoning="hmm"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ]
    )
    output = _of_type(payloads, "response.completed")[0]["response"]["output"]
    assert [o["type"] for o in output] == ["reasoning"]
    assert output[0]["summary"][0]["text"] == "hmm"


def _outcome(model: str, content: str) -> ModelOutcome:
    return ModelOutcome(identity=model, llm=model, model=model, status="ok", content=content)


async def test_show_work_off_ignores_member_completed():
    payloads = await _collect(
        [
            MemberCompleted(outcome=_outcome("a", "perspective A")),
            AnswerDelta(content="answer"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ],
        show_work="off",
    )
    assert not _of_type(payloads, "response.reasoning_summary_text.delta")
    output = _of_type(payloads, "response.completed")[0]["response"]["output"]
    assert [o["type"] for o in output] == ["message"]


async def test_show_work_inline_surfaces_member_dump_as_its_own_reasoning_item():
    payloads = await _collect(
        [
            MemberCompleted(outcome=_outcome("a", "perspective A")),
            MemberCompleted(outcome=_outcome("b", "perspective B")),
            SynthesisStarted(llm="syn", model="syn/model"),
            AnswerDelta(reasoning="genuine synth thinking"),
            AnswerDelta(content="answer"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ],
        show_work="inline",
    )
    output = _of_type(payloads, "response.completed")[0]["response"]["output"]
    # member-dump reasoning, genuine synth reasoning, then the message — three distinct items.
    assert [o["type"] for o in output] == ["reasoning", "reasoning", "message"]
    member_text = output[0]["summary"][0]["text"]
    assert "Model: a" in member_text
    assert "perspective A" in member_text
    assert "Model: b" in member_text
    assert "perspective B" in member_text
    assert output[1]["summary"][0]["text"] == "genuine synth thinking"
    assert output[0]["id"] != output[1]["id"]


async def test_show_work_inline_with_no_genuine_reasoning_still_shows_member_dump():
    payloads = await _collect(
        [
            MemberCompleted(outcome=_outcome("a", "perspective A")),
            SynthesisStarted(llm="syn", model="syn/model"),
            AnswerDelta(content="answer"),
            Completed(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=1),
                total_cost_usd=0.0,
            ),
        ],
        show_work="inline",
    )
    output = _of_type(payloads, "response.completed")[0]["response"]["output"]
    assert [o["type"] for o in output] == ["reasoning", "message"]
    assert "perspective A" in output[0]["summary"][0]["text"]


# ---------------------------------------------------------------------------------------------
# Non-streaming build_response
# ---------------------------------------------------------------------------------------------
def test_build_response_show_work_inline_adds_member_dump_reasoning_item():
    result = EnsembleResult(
        text="final answer",
        outcomes=(_outcome("a", "perspective A"), _outcome("b", "perspective B")),
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        total_cost_usd=0.0,
        finish_reason="stop",
        reasoning="genuine synth thinking",
    )
    obj = build_response(result, response_id="resp-1", model="e", created=1, show_work="inline")
    assert [o["type"] for o in obj["output"]] == ["reasoning", "reasoning", "message"]
    member_text = obj["output"][0]["summary"][0]["text"]
    assert "perspective A" in member_text
    assert "perspective B" in member_text
    assert obj["output"][1]["summary"][0]["text"] == "genuine synth thinking"


def test_build_response_show_work_off_has_no_member_dump():
    result = EnsembleResult(
        text="final answer",
        outcomes=(_outcome("a", "perspective A"),),
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        total_cost_usd=0.0,
        finish_reason="stop",
    )
    obj = build_response(result, response_id="resp-1", model="e", created=1, show_work="off")
    assert [o["type"] for o in obj["output"]] == ["message"]


async def test_pipeline_failed_finalizes_open_tool_items_and_carries_no_dangler():
    payloads = await _collect(
        [
            ToolCallStarted(index=0, call_id="c0", name="fn"),
            ToolCallDelta(index=0, arguments_fragment="{}"),
            PipelineFailed(code="upstream_error", message="boom", http_status=502),
        ]
    )
    # every opened item is finalized (added count == done count), then response.failed.
    assert len(_of_type(payloads, "response.output_item.added")) == 1
    assert len(_of_type(payloads, "response.output_item.done")) == 1
    failed = _of_type(payloads, "response.failed")
    assert failed
    assert failed[0]["response"]["status"] == "failed"
