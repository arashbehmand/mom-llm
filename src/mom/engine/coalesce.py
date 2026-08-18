"""In-flight request coalescing: a duplicate concurrent turn attaches to the one live run instead
of starting a second fan-out + synthesis.

Structurally mirrors :class:`~mom.adapters.eventbus.InMemoryEventBus`: each in-flight run is a
growing history of typed :data:`~mom.domain.events.StreamEvent`\\ s plus a live subscriber set. The
actual pipeline (``run_ensemble``) runs inside a **detached leader task** that feeds the broadcast,
and every consumer — including the very first caller, the one whose request actually triggered the
run — is "just a subscriber" of it. That symmetry is what makes the leader's own disconnect
harmless to any follower still attached: nothing distinguishes "the original caller went away"
from "a follower went away", so the run keeps going as long as *anyone* is attached (the orphan
grace below), and it is the same reasoning behind ``CachingClient``'s ``asyncio.shield`` — a
follower's cancellation must never be able to reach back and kill work others depend on.

Buffering *typed events*, not encoded bytes, is deliberate: each subscriber folds the same replay
through its own encoder, ``ChatFrame``/response id, stream profile and ``include_usage`` — a
byte-level buffer would freeze all of those to whichever caller happened to be the leader.

In-flight only, no post-completion linger: a run is dropped from the registry the instant its
terminal event is buffered, so a deliberate identical regenerate *after* completion starts a fresh
run rather than replaying a stale answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib
from dataclasses import dataclass, field

from mom.domain.events import Completed, PipelineFailed, StreamEvent
from mom.runtime.logging import get_logger


logger = get_logger("mom.coalesce")

_DEFAULT_ORPHAN_GRACE_SECONDS = 90.0
_DEFAULT_MAX_BUFFER_CHARS = 8 * 1024 * 1024

# Fields whose accumulated text is what actually grows a long run's buffered history — this is a
# cheap backstop (a character count, not exact wire bytes) against a pathological run buffering
# an unbounded amount for a slow-draining or absent subscriber, not a precise memory accounting.
_TEXT_FIELDS = ("content", "reasoning", "arguments_fragment", "message", "detail", "preview")


def _is_terminal(event: StreamEvent) -> bool:
    return isinstance(event, (Completed, PipelineFailed))


def _event_chars(event: StreamEvent) -> int:
    total = 0
    for attr in _TEXT_FIELDS:
        value = getattr(event, attr, None)
        if isinstance(value, str):
            total += len(value)
    return total


@dataclass(slots=True)
class _Run:
    request_id: str
    history: list[StreamEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue[StreamEvent | None]] = field(default_factory=set)
    terminal: bool = False
    buffered_chars: int = 0
    task: asyncio.Task[None] | None = None
    grace_handle: asyncio.TimerHandle | None = None


class CoalesceRegistry:
    """Registry of in-flight runs keyed by request identity (see ``domain/requestkey.py``)."""

    def __init__(
        self,
        *,
        orphan_grace_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS,
        max_buffer_chars: int = _DEFAULT_MAX_BUFFER_CHARS,
    ) -> None:
        self._runs: dict[str, _Run] = {}
        self._grace_seconds = orphan_grace_seconds
        self._max_buffer_chars = max_buffer_chars
        # Strong references so a leader task can never be GC'd mid-run purely because nothing
        # else in the process happens to hold onto it (asyncio only weakly references tasks) —
        # independent of `_runs`, which a run leaves the moment it goes terminal or orphaned.
        self._tasks: set[asyncio.Task[None]] = set()

    def attach(
        self,
        identity: str,
        request_id: str,
        run_factory: Callable[[], AsyncIterator[StreamEvent]],
    ) -> tuple[AsyncIterator[StreamEvent], str]:
        """Attach to the in-flight run for ``identity``, starting one if none exists.

        Returns ``(events, leader_request_id)``. ``leader_request_id`` equals ``request_id`` when
        this call became the leader (nothing was in flight, so it started a fresh run via
        ``run_factory``); it is the *original* caller's id when this call coalesced onto an
        already-running one instead — callers use that to decide whether to advertise
        ``X-MoM-Coalesced`` and to point the progress URL at the leader's channel rather than a
        follower's own (nothing is ever published there).

        No ``await`` happens between the lookup and registration below, so at most one leader is
        ever created per identity — the same register-and-snapshot idiom ``CachingClient`` and
        ``InMemoryEventBus.subscribe`` use for exactly this reason.
        """
        existing = self._runs.get(identity)
        if existing is not None and not existing.terminal:
            return self._subscribe(existing), existing.request_id
        run_state = _Run(request_id=request_id)
        self._runs[identity] = run_state
        task = asyncio.create_task(self._drive(identity, run_state, run_factory))
        run_state.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self._subscribe(run_state), request_id

    def _subscribe(self, run_state: _Run) -> AsyncIterator[StreamEvent]:
        self._cancel_grace(run_state)
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        # Register and snapshot history in one synchronous step (no `await` between): a concurrent
        # `_drive` publish cannot interleave here, so nothing is missed and nothing replayed twice.
        run_state.subscribers.add(queue)
        replay = list(run_state.history)
        return self._events(run_state, queue, replay)

    async def _events(
        self,
        run_state: _Run,
        queue: asyncio.Queue[StreamEvent | None],
        replay: list[StreamEvent],
    ) -> AsyncIterator[StreamEvent]:
        try:
            for event in replay:
                yield event
                if _is_terminal(event):
                    return
            while True:
                item = await queue.get()
                if item is None:  # orphan-cancelled / crashed-before-terminal sentinel
                    return
                yield item
                if _is_terminal(item):
                    return
        finally:
            run_state.subscribers.discard(queue)
            if not run_state.subscribers and not run_state.terminal:
                self._start_grace(run_state)

    def _start_grace(self, run_state: _Run) -> None:
        loop = asyncio.get_running_loop()

        def _expire() -> None:
            if run_state.subscribers or run_state.terminal:
                return  # a new subscriber (or completion) beat the timer — nothing to do
            logger.info(
                "coalesced run orphaned (no subscribers); cancelling",
                request_id=run_state.request_id,
            )
            if run_state.task is not None:
                run_state.task.cancel()

        run_state.grace_handle = loop.call_later(self._grace_seconds, _expire)

    def _cancel_grace(self, run_state: _Run) -> None:
        if run_state.grace_handle is not None:
            run_state.grace_handle.cancel()
            run_state.grace_handle = None

    def _broadcast(self, run_state: _Run, event: StreamEvent | None) -> None:
        for queue in run_state.subscribers:
            queue.put_nowait(event)

    async def _drive(
        self,
        identity: str,
        run_state: _Run,
        run_factory: Callable[[], AsyncIterator[StreamEvent]],
    ) -> None:
        gen = run_factory()
        try:
            async for event in gen:
                run_state.history.append(event)
                run_state.buffered_chars += _event_chars(event)
                self._broadcast(run_state, event)
                if _is_terminal(event):
                    run_state.terminal = True
                    return
                if (
                    run_state.buffered_chars > self._max_buffer_chars
                    and self._runs.get(identity) is run_state
                ):
                    # Stop offering this run to NEW attachers — existing subscribers, whose
                    # queues already hold everything buffered so far, keep draining normally
                    # as `_drive` keeps publishing; a request landing after this point simply
                    # becomes the leader of its own fresh run instead of joining a huge one.
                    logger.warning(
                        "coalesced run exceeded its buffer cap; no longer accepting "
                        "new attachments",
                        request_id=run_state.request_id,
                        buffered_chars=run_state.buffered_chars,
                    )
                    del self._runs[identity]
        except asyncio.CancelledError:
            raise
        except Exception:
            # `run_ensemble` documents itself as never raising (failures are events) — reaching
            # here means something OUTSIDE its own error handling broke (e.g. this drive loop
            # itself). Without a synthetic terminal event, every attached subscriber would simply
            # hang forever waiting on a queue nothing will ever fill again.
            logger.warning("coalesced run crashed outside its own error handling", exc_info=True)
            failure = PipelineFailed(code="internal_error", message="Internal server error")
            run_state.history.append(failure)
            self._broadcast(run_state, failure)
            run_state.terminal = True
        finally:
            # `gen` is whatever `run_ensemble` returned — declared as `AsyncIterator`, but in
            # practice always an async generator, so it needs an explicit `aclose()` on every
            # early exit (an abandoned/orphan-cancelled run, or the crash branch above) to run
            # its own `finally` (the "client disconnected" progress safety net) instead of being
            # silently abandoned mid-generator. A normal StopAsyncIteration exit already closed
            # it — `aclose()` on an already-exhausted generator is a documented no-op.
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            self._cancel_grace(run_state)
            self._broadcast(run_state, None)
            if self._runs.get(identity) is run_state:
                del self._runs[identity]
