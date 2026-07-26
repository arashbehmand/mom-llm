"""The progress event bus: ProgressEvent serialization, the in-memory bus, the Redis bus.

The Redis bus is exercised against a small in-process fake redis client (no server, no sockets) so
its publish/subscribe, terminal-close, and backoff-reconnect logic are tested deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from mom.adapters.eventbus import InMemoryEventBus, RedisEventBus
from mom.api.reqid import resolve_request_id
from mom.domain.progress import ProgressEvent
from mom.testing import SequentialIds


def _ev(kind: str, **kw: Any) -> ProgressEvent:
    return ProgressEvent(kind=kind, ensemble="e", **kw)  # type: ignore[arg-type]


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Cooperatively yield until a background task reaches an observable state (bounded)."""
    for _ in range(100_000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


# ---------------------------------------------------------------------------------------------
# ProgressEvent (pure domain)
# ---------------------------------------------------------------------------------------------
def test_to_dict_keeps_required_and_drops_unset_optionals():
    event = ProgressEvent(kind="synthesis_started", ensemble="bmom", member="synth", model="m")
    assert event.to_dict() == {
        "kind": "synthesis_started",
        "ensemble": "bmom",
        "member": "synth",
        "model": "m",
    }


def test_json_round_trip_preserves_all_fields():
    event = ProgressEvent(
        kind="member_completed",
        ensemble="e",
        member="a",
        model="openai/a",
        status="ok",
        duration_ms=12.5,
        members_total=3,
        completed=1,
    )
    assert ProgressEvent.from_json(event.to_json()) == event


@pytest.mark.parametrize(
    ("kind", "terminal"),
    [
        ("fanout_started", False),
        ("member_completed", False),
        ("synthesis_started", False),
        ("completed", True),
        ("failed", True),
    ],
)
def test_terminal_flag(kind: str, terminal: bool):
    assert ProgressEvent(kind=kind).terminal is terminal  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# request-id resolution
# ---------------------------------------------------------------------------------------------
def test_resolve_request_id_honors_well_formed_client_id():
    assert resolve_request_id("trace-abc.123", SequentialIds()) == "trace-abc.123"


def test_resolve_request_id_mints_fresh_for_missing_or_unsafe():
    ids = SequentialIds()
    assert resolve_request_id(None, ids) == "req-1"
    assert resolve_request_id("has/slash", ids) == "req-2"  # unsafe path/channel char
    assert resolve_request_id("x" * 129, ids) == "req-3"  # too long


# ---------------------------------------------------------------------------------------------
# InMemoryEventBus
# ---------------------------------------------------------------------------------------------
async def _drain(bus: InMemoryEventBus, request_id: str) -> list[str]:
    return [event.kind async for event in bus.subscribe(request_id)]


async def test_late_subscriber_replays_history_then_stops_at_terminal():
    bus = InMemoryEventBus()
    bus.publish("r1", _ev("fanout_started", members_total=2))
    bus.publish("r1", _ev("member_completed", member="a"))
    bus.publish("r1", _ev("completed", status="stop"))
    assert await _drain(bus, "r1") == ["fanout_started", "member_completed", "completed"]


async def test_live_subscriber_receives_published_events():
    bus = InMemoryEventBus()
    task = asyncio.create_task(_drain(bus, "r2"))
    await _spin_until(lambda: "r2" in bus._channels)  # subscriber registered
    bus.publish("r2", _ev("fanout_started"))
    bus.publish("r2", _ev("completed"))
    assert await asyncio.wait_for(task, timeout=1.0) == ["fanout_started", "completed"]


async def test_multiple_subscribers_each_receive_every_event():
    bus = InMemoryEventBus()
    t1 = asyncio.create_task(_drain(bus, "r3"))
    t2 = asyncio.create_task(_drain(bus, "r3"))
    await _spin_until(lambda: "r3" in bus._channels and len(bus._channels["r3"].subscribers) == 2)
    bus.publish("r3", _ev("synthesis_started"))
    bus.publish("r3", _ev("completed"))
    got1, got2 = await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1.0)
    assert got1 == got2 == ["synthesis_started", "completed"]


async def test_ttl_eviction_drops_idle_channel():
    now = {"t": 1000.0}
    bus = InMemoryEventBus(ttl_seconds=10.0, now=lambda: now["t"])
    bus.publish("old", _ev("fanout_started"))
    assert "old" in bus._channels
    now["t"] = 1100.0  # advance well past the TTL
    bus.publish("new", _ev("fanout_started"))  # any activity sweeps expired channels
    assert "old" not in bus._channels
    assert "new" in bus._channels


async def test_ttl_eviction_closes_a_live_subscriber():
    now = {"t": 1000.0}
    bus = InMemoryEventBus(ttl_seconds=10.0, now=lambda: now["t"])
    task = asyncio.create_task(_drain(bus, "idle"))
    await _spin_until(lambda: "idle" in bus._channels)
    bus.publish("idle", _ev("fanout_started"))
    now["t"] = 1100.0
    bus.publish("other", _ev("fanout_started"))  # sweep sends the idle subscriber a close sentinel
    # FIFO: the one real event is delivered before the sentinel, then the stream closes.
    assert await asyncio.wait_for(task, timeout=1.0) == ["fanout_started"]


async def test_aclose_closes_all_subscribers():
    bus = InMemoryEventBus()
    task = asyncio.create_task(_drain(bus, "r"))
    await _spin_until(lambda: "r" in bus._channels)
    await bus.aclose()
    assert await asyncio.wait_for(task, timeout=1.0) == []
    assert bus._channels == {}


# ---------------------------------------------------------------------------------------------
# RedisEventBus — against an in-process fake client (no server / sockets)
# ---------------------------------------------------------------------------------------------
class _FakePubSub:
    def __init__(self, server: _FakeRedis) -> None:
        self._server = server
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._channels: list[str] = []

    async def subscribe(self, channel: str) -> None:
        self._channels.append(channel)
        self._server.subscribers.setdefault(channel, []).append(self._queue)
        await self._queue.put({"type": "subscribe", "channel": channel, "data": 1})

    async def listen(self) -> Any:
        while True:
            yield await self._queue.get()

    async def aclose(self) -> None:
        for channel in self._channels:
            subs = self._server.subscribers.get(channel, [])
            if self._queue in subs:
                subs.remove(self._queue)


class _FakeRedis:
    def __init__(self) -> None:
        self.subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self.closed = False

    def pubsub(self) -> _FakePubSub:
        return _FakePubSub(self)

    async def publish(self, channel: str, data: str) -> int:
        queues = list(self.subscribers.get(channel, []))
        for queue in queues:
            await queue.put({"type": "message", "channel": channel, "data": data})
        return len(queues)

    async def aclose(self) -> None:
        self.closed = True


async def test_redis_bus_publish_subscribe_round_trip():
    client = _FakeRedis()
    bus = RedisEventBus(client)
    task = asyncio.create_task(_drain_bus(bus, "r1"))
    await _spin_until(lambda: "mom:progress:r1" in client.subscribers)
    bus.publish("r1", _ev("fanout_started"))
    bus.publish("r1", _ev("completed"))
    assert await asyncio.wait_for(task, timeout=1.0) == ["fanout_started", "completed"]
    await bus.aclose()
    assert client.closed


async def test_redis_bus_uses_prefixed_channel():
    client = _FakeRedis()
    bus = RedisEventBus(client)
    task = asyncio.create_task(_drain_bus(bus, "abc"))
    await _spin_until(lambda: "mom:progress:abc" in client.subscribers)
    bus.publish("abc", _ev("completed"))
    await asyncio.wait_for(task, timeout=1.0)


async def test_redis_bus_reconnects_with_backoff_after_a_drop():
    class _BrokenPubSub:
        async def subscribe(self, channel: str) -> None:
            raise ConnectionError("broker down")

        async def listen(self) -> Any:  # pragma: no cover - never reached
            if False:
                yield {}

        async def aclose(self) -> None:
            return None

    class _FlakyRedis(_FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def pubsub(self) -> Any:
            self.attempts += 1
            return _BrokenPubSub() if self.attempts == 1 else super().pubsub()

    client = _FlakyRedis()
    bus = RedisEventBus(client, backoff_base=0.001, backoff_max=0.01)
    task = asyncio.create_task(_drain_bus(bus, "r1"))
    # subscribers is only populated after the first (broken) attempt fails and the retry connects
    await _spin_until(lambda: "mom:progress:r1" in client.subscribers)
    bus.publish("r1", _ev("completed"))
    assert await asyncio.wait_for(task, timeout=1.0) == ["completed"]
    assert client.attempts >= 2


async def _drain_bus(bus: RedisEventBus, request_id: str) -> list[str]:
    return [event.kind async for event in bus.subscribe(request_id)]
