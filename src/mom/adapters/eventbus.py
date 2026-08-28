"""Event-bus adapters: an in-memory default and an optional Redis-backed implementation.

An :class:`~mom.domain.ports.EventBus` fans a request's progress events out to any number of
subscribers, keyed by request id. :class:`InMemoryEventBus` is the default (single process, no
dependencies); :class:`RedisEventBus` is selected when a ``redis_url`` is configured so progress
survives across worker processes. Publishing is fire-and-forget on both — it never blocks the
request path and never raises.

``redis`` is an OPTIONAL dependency: it is imported lazily inside :meth:`RedisEventBus.from_url`,
so importing this module (and running the in-memory bus) never requires it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib
from dataclasses import dataclass, field
import time
from typing import Any, Literal, Protocol

from mom.domain.ports import EventBus, RunMemberSummary, RunSummary
from mom.domain.progress import ProgressEvent
from mom.runtime.logging import get_logger


logger = get_logger("mom.eventbus")


class ClosableBus(EventBus, Protocol):
    """An event bus that also owns resources to release at shutdown (both of the ones here do)."""

    async def aclose(self) -> None: ...


class _Channel:
    """Per-request state: retained history (for late subscribers) plus live subscriber queues."""

    __slots__ = ("closed", "expires_at", "history", "subscribers")

    def __init__(self) -> None:
        self.history: list[ProgressEvent] = []
        self.subscribers: set[asyncio.Queue[ProgressEvent | None]] = set()
        self.expires_at: float = 0.0
        self.closed: bool = False


class InMemoryEventBus:
    """Asyncio pub/sub keyed by request id, with a short TTL-evicted history buffer.

    Progress is subscribed to in a race with the request that produces it: the observer opens the
    stream at roughly the same moment the gateway starts publishing. So each channel retains its
    events for ``ttl_seconds`` after the last activity, and a subscriber first replays that history
    before going live — a subscriber that arrives slightly late still sees everything.

    Eviction is lazy (swept whenever any request publishes or subscribes), so the bus owns no
    background task and nothing has to be started or stopped in tests.
    """

    def __init__(
        self, *, ttl_seconds: float = 300.0, now: Callable[[], float] = time.monotonic
    ) -> None:
        self._channels: dict[str, _Channel] = {}
        self._ttl = ttl_seconds
        self._now = now

    def _evict_expired(self) -> None:
        now = self._now()
        expired = [rid for rid, ch in self._channels.items() if ch.expires_at <= now]
        for rid in expired:
            channel = self._channels.pop(rid, None)
            if channel is None:
                continue
            for queue in channel.subscribers:  # unblock any lingering subscribers so they close
                queue.put_nowait(None)

    def _touch(self, request_id: str) -> _Channel:
        channel = self._channels.get(request_id)
        if channel is None:
            channel = _Channel()
            self._channels[request_id] = channel
        channel.expires_at = self._now() + self._ttl
        return channel

    def publish(self, request_id: str, event: ProgressEvent) -> None:
        # Touch BEFORE sweeping: this channel's own `expires_at` might already be in the past
        # (a long gap since its last activity — e.g. between `synthesis_started` and `completed`
        # on a slow call). Sweeping first would evict it — destroying its history and
        # sentinel-closing its own live subscribers — one line before this same publish would have
        # kept it alive. Touching first means "my channel is alive; sweep the OTHERS."
        channel = self._touch(request_id)
        self._evict_expired()
        channel.history.append(event)
        if event.terminal:
            channel.closed = True
        for queue in channel.subscribers:
            queue.put_nowait(event)

    async def subscribe(self, request_id: str) -> AsyncIterator[ProgressEvent]:
        # Same reordering as `publish` — a subscriber must never evict the very history it came to
        # replay. `_evict_expired` is synchronous (no `await`), so this doesn't reopen the
        # register-and-snapshot race the comment below describes.
        channel = self._touch(request_id)
        self._evict_expired()
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        # Register and snapshot the history in one synchronous step (no await between): a publish
        # cannot interleave, so nothing is missed and nothing is replayed twice.
        channel.subscribers.add(queue)
        replay = list(channel.history)
        try:
            for event in replay:
                yield event
                if event.terminal:
                    return
            if channel.closed:  # a terminal event was already in the replayed history
                return
            while True:
                item = await queue.get()
                if item is None:  # eviction/shutdown sentinel
                    return
                yield item
                if item.terminal:
                    return
        finally:
            channel.subscribers.discard(queue)

    async def aclose(self) -> None:
        """Signal every subscriber to close and drop all channels (called on shutdown)."""
        for channel in list(self._channels.values()):
            for queue in channel.subscribers:
                queue.put_nowait(None)
        self._channels.clear()


class RunIndexBus:
    """An :class:`~mom.domain.ports.EventBus` decorator that also indexes the runs it carries.

    Progress events are published for a *subscriber*, so nothing in the bus contract can answer
    "what is running right now" — ``InMemoryEventBus`` only keeps history for ids you already
    know, and ``RedisEventBus`` keeps nothing at all. Folding each event into a compact per-run
    summary as it passes through gives that answer for whichever bus is configured, without
    widening ``EventBus`` for the one caller that wants it (see ``RunIndex``).

    Process-local by construction: it indexes what *this* process published, so a multi-worker
    deployment sees one worker's runs. metrics.db remains the cross-process, durable record —
    this covers the gap that ledger structurally cannot, the run that has not finished yet.
    """

    def __init__(
        self,
        inner: ClosableBus,
        *,
        ttl_seconds: float = 900.0,
        max_runs: int = 200,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner  # public: the wrapped bus is what deployment configured
        self._ttl = ttl_seconds
        self._max_runs = max_runs
        self._clock = clock  # wall clock: timestamps a reader can compare against metrics rows
        self._monotonic = monotonic  # eviction: immune to clock steps
        self._runs: dict[str, _RunState] = {}

    def publish(self, request_id: str, event: ProgressEvent) -> None:
        try:
            self._observe(request_id, event)
        except Exception:  # indexing must never break the thing it observes
            logger.debug("run index update failed", exc_info=True)
        self.inner.publish(request_id, event)

    def subscribe(self, request_id: str) -> AsyncIterator[ProgressEvent]:
        return self.inner.subscribe(request_id)

    async def aclose(self) -> None:
        self._runs.clear()
        await self.inner.aclose()

    def snapshot(self, request_id: str | None = None) -> list[RunSummary]:
        """Indexed runs, newest activity first (one entry when ``request_id`` is given)."""
        self._evict()
        states = (
            [s for rid, s in self._runs.items() if rid == request_id]
            if request_id is not None
            else sorted(self._runs.values(), key=lambda s: s.updated_at, reverse=True)
        )
        return [state.to_summary() for state in states]

    def _observe(self, request_id: str, event: ProgressEvent) -> None:
        state = self._runs.get(request_id)
        if state is None:
            state = _RunState(request_id=request_id, started_at=self._clock())
            self._runs[request_id] = state
        state.apply(event, now=self._clock())
        state.expires_at = self._monotonic() + self._ttl
        self._evict()

    def _evict(self) -> None:
        """Drop terminal runs past their TTL, then the oldest if still over ``max_runs``.

        In-flight runs are never evicted by either rule: a fan-out can legitimately outlive the
        TTL (a 20-minute member timeout is configurable), and dropping it would report a live run
        as gone — the one thing this index exists to prevent.
        """
        now = self._monotonic()
        for rid, state in list(self._runs.items()):
            if state.terminal and state.expires_at <= now:
                del self._runs[rid]
        overflow = len(self._runs) - self._max_runs
        if overflow <= 0:
            return
        evictable = sorted(
            (s for s in self._runs.values() if s.terminal), key=lambda s: s.updated_at
        )
        for state in evictable[:overflow]:
            self._runs.pop(state.request_id, None)


@dataclass
class _RunState:
    """Mutable fold of one run's progress events (see :class:`RunIndexBus`)."""

    request_id: str
    started_at: float
    ensemble: str = ""
    state: Literal["running", "synthesizing", "completed", "failed"] = "running"
    updated_at: float = 0.0
    expires_at: float = 0.0
    members_total: int | None = None
    members: dict[str, RunMemberSummary] = field(default_factory=dict)
    cost_usd: float = 0.0
    finish_reason: str | None = None
    detail: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in ("completed", "failed")

    def apply(self, event: ProgressEvent, *, now: float) -> None:
        self.updated_at = now
        if event.ensemble:
            self.ensemble = event.ensemble
        if event.members_total is not None:
            self.members_total = event.members_total
        if event.kind == "fanout_started":
            for identity, model in event.members or ():
                self.members.setdefault(identity, RunMemberSummary(identity=identity, model=model))
        elif event.kind == "member_completed" and event.member is not None:
            self.members[event.member] = RunMemberSummary(
                identity=event.member,
                model=event.model,
                status=event.status,
                duration_ms=event.duration_ms,
                cost_usd=event.cost_usd,
            )
            # Per-member costs accumulate; the terminal event's total replaces the running sum
            # (it also covers synthesis), so the two never double-count.
            self.cost_usd += event.cost_usd or 0.0
        elif event.kind == "synthesis_started":
            self.state = "synthesizing"
        elif event.kind == "completed":
            self.state = "completed"
            self.finish_reason = event.status
            if event.cost_usd is not None:
                self.cost_usd = event.cost_usd
        elif event.kind == "failed":
            self.state = "failed"
            self.detail = event.detail

    def to_summary(self) -> RunSummary:
        return RunSummary(
            request_id=self.request_id,
            ensemble=self.ensemble,
            state=self.state,
            started_at=self.started_at,
            updated_at=self.updated_at,
            members_total=self.members_total,
            members=tuple(self.members.values()),
            cost_usd=self.cost_usd,
            finish_reason=self.finish_reason,
            detail=self.detail,
        )


class RedisEventBus:
    """Redis pub/sub event bus (progress survives across worker processes).

    Publishing schedules a ``PUBLISH`` on the running loop and returns immediately — it never
    awaits in the request path. Subscribing opens a dedicated pubsub connection and yields decoded
    events, reconnecting with capped exponential backoff if the connection drops. A subscriber
    still completes cleanly on a terminal event.
    """

    _CHANNEL_PREFIX = "mom:progress:"

    def __init__(
        self, client: Any, *, backoff_base: float = 0.5, backoff_max: float = 10.0
    ) -> None:
        self._client = client
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisEventBus:
        from redis.asyncio import Redis  # lazy: redis is an optional dependency

        return cls(Redis.from_url(url), **kwargs)

    def _channel(self, request_id: str) -> str:
        return f"{self._CHANNEL_PREFIX}{request_id}"

    def publish(self, request_id: str, event: ProgressEvent) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop -> nothing to publish onto
            return
        task = loop.create_task(self._publish(request_id, event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _publish(self, request_id: str, event: ProgressEvent) -> None:
        try:
            await self._client.publish(self._channel(request_id), event.to_json())
        except Exception:  # fire-and-forget: a broker hiccup must never surface to the caller
            logger.debug("redis progress publish failed", exc_info=True)

    async def subscribe(self, request_id: str) -> AsyncIterator[ProgressEvent]:
        channel = self._channel(request_id)
        backoff = self._backoff_base
        while True:
            pubsub = self._client.pubsub()
            try:
                await pubsub.subscribe(channel)
                backoff = self._backoff_base  # reset after a healthy (re)connect
                async for message in pubsub.listen():
                    if not isinstance(message, dict) or message.get("type") != "message":
                        continue  # skip subscribe/unsubscribe confirmations
                    event = _decode(message.get("data"))
                    if event is None:
                        continue
                    yield event
                    if event.terminal:
                        return
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception:
                logger.debug("redis subscribe dropped; reconnecting", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._backoff_max)
            finally:
                await _aclose(pubsub)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await _aclose(self._client)


def _decode(data: Any) -> ProgressEvent | None:
    if data is None:
        return None
    try:
        return ProgressEvent.from_json(data)
    except Exception:
        logger.debug("dropping undecodable progress message", exc_info=True)
        return None


async def _aclose(closable: Any) -> None:
    """Close a redis pubsub/client across versions (``aclose`` on new, ``close`` on old)."""
    closer = getattr(closable, "aclose", None) or getattr(closable, "close", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):
        result = closer()
        if asyncio.iscoroutine(result):
            await result
