"""``CoalesceRegistry``: in-flight request coalescing, exercised directly against small scripted
event generators rather than the full pipeline — the behavior under test is the registry's own
leader/follower bookkeeping, not fan-out/synthesis.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib

from mom.domain.events import AnswerDelta, Completed, PipelineFailed, StreamEvent
from mom.domain.results import Usage
from mom.engine.coalesce import CoalesceRegistry


async def _abandon(events: AsyncIterator[StreamEvent], *, wait_seconds: float = 0.05) -> None:
    """Simulate a subscriber disconnecting mid-stream: pull once (so the underlying async
    generator genuinely starts and suspends waiting for the next event — closing/cancelling a
    generator that never started running is a documented no-op, it does NOT run the generator's
    own ``finally``), then let the pull time out and cancel it. ``asyncio.wait_for`` waits for the
    cancellation to actually finish unwinding before returning, so this only returns once the
    subscriber's cleanup (``run_state.subscribers.discard``) has genuinely run."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(events.__anext__(), timeout=wait_seconds)


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Cooperatively yield until a background task reaches an observable state (bounded)."""
    for _ in range(100_000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


async def _drain(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


def _scripted_run(
    *, calls: list[int], gate: asyncio.Event | None = None, content: str = "hi"
) -> Callable[[], AsyncIterator[StreamEvent]]:
    """A ``run_ensemble``-shaped factory: increments ``calls`` each time it's actually invoked (so
    a test can assert coalescing by counting real starts), optionally waits on ``gate`` before
    yielding anything (to hold a run open across a controlled window), then yields one delta and a
    terminal ``Completed``.
    """

    async def run() -> AsyncIterator[StreamEvent]:
        calls.append(1)
        if gate is not None:
            await gate.wait()
        yield AnswerDelta(content=content)
        yield Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0)

    return run


def _hanging_run(
    calls: list[int], *, on_cancel: list[str] | None = None
) -> Callable[[], AsyncIterator[StreamEvent]]:
    """A run that never completes on its own — only cancellation ends it."""

    async def run() -> AsyncIterator[StreamEvent]:
        calls.append(1)
        try:
            await asyncio.Event().wait()
            yield Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0)
        finally:
            if on_cancel is not None:
                on_cancel.append("cancelled")

    return run


async def test_second_attach_becomes_a_follower_of_the_first():
    registry = CoalesceRegistry()
    calls: list[int] = []
    gate = asyncio.Event()
    leader_events, leader_id = registry.attach(
        "id-1", "req-a", _scripted_run(calls=calls, gate=gate)
    )
    follower_events, follower_leader_id = registry.attach(
        "id-1", "req-b", _scripted_run(calls=calls, gate=gate)
    )
    assert leader_id == "req-a"
    assert follower_leader_id == "req-a"  # the follower is told the LEADER's id, not its own
    gate.set()
    leader_out, follower_out = await asyncio.wait_for(
        asyncio.gather(_drain(leader_events), _drain(follower_events)), timeout=1.0
    )
    assert leader_out == follower_out
    assert [type(e).__name__ for e in leader_out] == ["AnswerDelta", "Completed"]
    # `run_factory` (the closure passed to the LOSING attach()) is never invoked — one real run.
    assert len(calls) == 1


async def test_unrelated_identities_never_coalesce():
    registry = CoalesceRegistry()
    calls: list[int] = []
    a_events, a_leader = registry.attach("id-a", "req-a", _scripted_run(calls=calls))
    b_events, b_leader = registry.attach("id-b", "req-b", _scripted_run(calls=calls))
    assert a_leader == "req-a"
    assert b_leader == "req-b"
    await asyncio.wait_for(asyncio.gather(_drain(a_events), _drain(b_events)), timeout=1.0)
    assert len(calls) == 2


async def test_follower_dropping_out_does_not_affect_the_leader():
    registry = CoalesceRegistry()
    calls: list[int] = []
    gate = asyncio.Event()
    leader_events, _ = registry.attach("id-1", "req-a", _scripted_run(calls=calls, gate=gate))
    follower_events, _ = registry.attach("id-1", "req-b", _scripted_run(calls=calls, gate=gate))
    # The follower never reads from its subscription at all (a stalled/dropped client) — the
    # leader must be completely unaffected by that.
    del follower_events
    gate.set()
    leader_out = await asyncio.wait_for(_drain(leader_events), timeout=1.0)
    assert [type(e).__name__ for e in leader_out] == ["AnswerDelta", "Completed"]
    assert len(calls) == 1


async def test_post_completion_attach_starts_a_fresh_run():
    registry = CoalesceRegistry()
    calls: list[int] = []
    first_events, _first_leader = registry.attach("id-1", "req-a", _scripted_run(calls=calls))
    await asyncio.wait_for(_drain(first_events), timeout=1.0)
    assert len(calls) == 1
    # The run is terminal now — dropped from the registry the instant it completed — so a second
    # attach for the SAME identity must become a brand new leader, not replay the old answer.
    second_events, second_leader = registry.attach("id-1", "req-b", _scripted_run(calls=calls))
    assert second_leader == "req-b"
    await asyncio.wait_for(_drain(second_events), timeout=1.0)
    assert len(calls) == 2


async def test_crash_outside_run_ensemble_yields_a_synthetic_pipeline_failed():
    registry = CoalesceRegistry()

    async def broken_run() -> AsyncIterator[StreamEvent]:
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator

    events, _ = registry.attach("id-1", "req-a", broken_run)
    out = await asyncio.wait_for(_drain(events), timeout=1.0)
    assert len(out) == 1
    failure = out[0]
    assert isinstance(failure, PipelineFailed)
    assert failure.code == "internal_error"


async def test_orphan_grace_cancels_an_unwatched_run():
    registry = CoalesceRegistry(orphan_grace_seconds=0.01)
    calls: list[int] = []
    cancelled: list[str] = []
    events, _ = registry.attach("id-1", "req-a", _hanging_run(calls, on_cancel=cancelled))
    await _spin_until(lambda: len(calls) == 1)
    await _abandon(events)  # the only subscriber leaves -> starts the grace timer
    await _spin_until(lambda: bool(cancelled))
    assert "id-1" not in registry._runs  # white-box check of the eviction itself


async def test_new_attach_before_grace_expires_rescues_the_run():
    registry = CoalesceRegistry(orphan_grace_seconds=1.0)
    calls: list[int] = []
    gate = asyncio.Event()
    first_events, leader_id = registry.attach(
        "id-1", "req-a", _scripted_run(calls=calls, gate=gate)
    )
    await _abandon(first_events)  # zero subscribers -> the 1s grace timer starts
    # A second attach lands well before that 1s grace would fire.
    second_events, second_leader = registry.attach(
        "id-1", "req-b", _scripted_run(calls=calls, gate=gate)
    )
    assert second_leader == leader_id  # rejoined the SAME run, not a fresh one
    gate.set()
    out = await asyncio.wait_for(_drain(second_events), timeout=1.0)
    assert [type(e).__name__ for e in out] == ["AnswerDelta", "Completed"]
    assert len(calls) == 1  # the original run_factory, never re-invoked


async def test_buffer_cap_stops_new_attachments_but_keeps_the_original_subscriber_fed():
    registry = CoalesceRegistry(max_buffer_chars=10)
    calls: list[int] = []
    release = asyncio.Event()

    async def big_run() -> AsyncIterator[StreamEvent]:
        calls.append(1)
        yield AnswerDelta(content="x" * 20)  # blows straight past the 10-char cap
        await release.wait()
        yield Completed(finish_reason="stop", usage=Usage(), total_cost_usd=0.0)

    events, _leader_id = registry.attach("id-1", "req-a", big_run)
    await _spin_until(lambda: "id-1" not in registry._runs)
    # A second attach for the same identity now starts its OWN fresh run instead of joining.
    calls2: list[int] = []
    other_events, other_leader = registry.attach("id-1", "req-b", _scripted_run(calls=calls2))
    assert other_leader == "req-b"
    await asyncio.wait_for(_drain(other_events), timeout=1.0)
    assert len(calls2) == 1
    # The original subscriber is unaffected — it still gets the rest of ITS run.
    release.set()
    out = await asyncio.wait_for(_drain(events), timeout=1.0)
    assert [type(e).__name__ for e in out] == ["AnswerDelta", "Completed"]
    assert len(calls) == 1
