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
import time
from typing import Any

from mom.domain.progress import ProgressEvent
from mom.runtime.logging import get_logger


logger = get_logger("mom.eventbus")


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
        self._evict_expired()
        channel = self._touch(request_id)
        channel.history.append(event)
        if event.terminal:
            channel.closed = True
        for queue in channel.subscribers:
            queue.put_nowait(event)

    async def subscribe(self, request_id: str) -> AsyncIterator[ProgressEvent]:
        self._evict_expired()
        channel = self._touch(request_id)
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
