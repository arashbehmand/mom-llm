"""Responses SSE encoder: no cross-item text bleed, unique item ids, clean failure finalization."""

from __future__ import annotations

import json

from mom.api.encoders.responses import encode_sse
from mom.domain.events import (
    AnswerDelta,
    Completed,
    PipelineFailed,
    StreamEvent,
    ToolCallDelta,
    ToolCallStarted,
)
from mom.domain.results import Usage


async def _collect(events: list[StreamEvent]) -> list[dict]:
    async def gen():
        for event in events:
            yield event

    payloads: list[dict] = []
    async for block in encode_sse(gen(), response_id="resp-1", model="e", created=1):
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
