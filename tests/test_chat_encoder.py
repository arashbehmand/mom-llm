"""Chat Completions SSE encoder: the inline `<think>` preamble opens at fan-out start, not on the
first member completion (Issue 2), and closes cleanly on every exit path."""

from __future__ import annotations

import json

from mom.api.encoders.chat import ChatFrame, encode_sse
from mom.domain.events import (
    AnswerDelta,
    Completed,
    FanoutStarted,
    MemberCompleted,
    PipelineFailed,
    StreamEvent,
    SynthesisStarted,
)
from mom.domain.results import ModelOutcome, Usage


_FRAME = ChatFrame(id="chatcmpl-1", created=1, model="e")


async def _collect(
    events: list[StreamEvent],
    *,
    show_work: str = "inline",
    progress_url: str | None = None,
    notices: tuple[str, ...] = (),
) -> list[dict]:
    async def gen():
        for event in events:
            yield event

    payloads: list[dict] = []
    async for block in encode_sse(
        gen(),
        _FRAME,
        show_work=show_work,
        include_usage=False,
        progress_url=progress_url,
        notices=notices,
    ):
        payloads.extend(
            json.loads(line[len("data: ") :])
            for line in block.decode().splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    return payloads


def _content_deltas(payloads: list[dict]) -> list[str]:
    """The ordered sequence of non-empty `delta.content` strings across all chunks."""
    out = []
    for p in payloads:
        choices = p.get("choices") or []
        if not choices:
            continue
        content = choices[0].get("delta", {}).get("content")
        if content:
            out.append(content)
    return out


def _outcome(identity: str = "m", *, model: str | None = None) -> ModelOutcome:
    return ModelOutcome(
        identity=identity,
        llm=identity,
        model=model or f"openai/{identity}",
        status="ok",
        content="hi",
    )


async def test_inline_preamble_emitted_on_first_fanout_started_before_any_member_content():
    payloads = await _collect(
        [
            FanoutStarted("a", "openai/a"),
            FanoutStarted("b", "openai/b"),
            MemberCompleted(_outcome("a")),
            MemberCompleted(_outcome("b")),
            SynthesisStarted("synth", "openai/synth"),
            AnswerDelta(content="answer"),
            Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0),
        ],
        show_work="inline",
        progress_url="https://mom.local/v1/progress/req-1",
    )
    deltas = _content_deltas(payloads)
    # <think> and the Progress link land before the FIRST member's content line — immediately at
    # fan-out start, not after the first member finishes (which can be minutes on a slow panel).
    assert deltas[0] == "<think>\n"
    assert deltas[1] == "Progress: https://mom.local/v1/progress/req-1\n\n"
    assert deltas[2].startswith("Model: openai/a")
    assert deltas[3].startswith("Model: openai/b")
    # Exactly one <think> opened across the whole stream (second FanoutStarted is a no-op).
    assert deltas.count("<think>\n") == 1


async def test_no_preamble_when_show_work_is_off():
    payloads = await _collect(
        [
            FanoutStarted("a", "openai/a"),
            MemberCompleted(_outcome("a")),
            SynthesisStarted("synth", "openai/synth"),
            AnswerDelta(content="answer"),
            Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0),
        ],
        show_work="off",
        progress_url="https://mom.local/v1/progress/req-1",
    )
    deltas = _content_deltas(payloads)
    assert not any("<think>" in d for d in deltas)
    assert not any("Progress:" in d for d in deltas)
    assert deltas == ["answer"]


async def test_member_completed_still_opens_think_when_fanout_started_is_absent():
    """Fallback path: an event stream that (for whatever reason) never yields FanoutStarted still
    gets its think block opened lazily on the first MemberCompleted, exactly as before this fix."""
    payloads = await _collect(
        [
            MemberCompleted(_outcome("a")),
            SynthesisStarted("synth", "openai/synth"),
            AnswerDelta(content="answer"),
            Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0),
        ],
        show_work="inline",
    )
    deltas = _content_deltas(payloads)
    assert deltas[0] == "<think>\n"
    assert deltas[1].startswith("Model: openai/a")


async def test_pipeline_failed_closes_an_open_think_block():
    payloads = await _collect(
        [FanoutStarted("a", "openai/a"), PipelineFailed(code="quorum_not_met", message="failed")],
        show_work="inline",
    )
    deltas = _content_deltas(payloads)
    assert deltas == ["<think>\n", "</think>\n\n"]
    # The error frame itself is still emitted after the think block closes.
    assert any("error" in p for p in payloads)


async def test_truncated_stream_closes_think_before_the_synthesized_stop_chunk():
    """No terminal event at all (e.g. the underlying generator ended early) — the fallback path at
    the bottom of encode_sse must not leave an unclosed <think> block."""
    payloads = await _collect([FanoutStarted("a", "openai/a")], show_work="inline")
    deltas = _content_deltas(payloads)
    assert deltas == ["<think>\n", "</think>\n\n"]


async def test_fanout_started_without_show_work_inline_does_nothing():
    payloads = await _collect(
        [
            FanoutStarted("a", "openai/a"),
            Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0),
        ],
        show_work="off",
    )
    assert _content_deltas(payloads) == []


async def test_a_notice_opens_its_own_think_block_even_with_show_work_off():
    """An ignored `<<SYSTEM>>` directive is the one thing a client must be told regardless of what
    the ensemble shows: with no member dump coming, the notice gets the block to itself."""
    payloads = await _collect(
        [
            FanoutStarted("a", "openai/a"),
            SynthesisStarted("s", "openai/s"),
            AnswerDelta(content="answer"),
            Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0),
        ],
        show_work="off",
        notices=("<<SYSTEM>> exclude: 'k33' is not a member — ignored.",),
    )
    deltas = _content_deltas(payloads)
    assert deltas[0] == "<think>\n"
    assert "k33" in deltas[1]
    assert deltas[2] == "</think>\n\n"
    assert deltas[3] == "answer"


async def test_a_notice_heads_the_member_dump_it_shares_a_block_with():
    payloads = await _collect(
        [FanoutStarted("a", "openai/a"), MemberCompleted(_outcome("a"))],
        show_work="inline",
        progress_url="http://p/1",
        notices=("<<SYSTEM>> synth: 'nope' is not an llm — ignored.",),
    )
    deltas = _content_deltas(payloads)
    assert deltas[0] == "<think>\n"
    assert deltas[1] == "Progress: http://p/1\n\n"
    assert "nope" in deltas[2]
    assert "Model: openai/a" in deltas[3]  # one block: notice first, then the panel
