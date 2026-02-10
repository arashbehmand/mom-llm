from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mom_service.events import MoMEvent, RedisPublisher


@pytest.mark.asyncio
async def test_publish_noop_when_disabled():
    publisher = RedisPublisher(redis_url=None)
    publisher.enabled = False
    publisher.redis_client = None

    event = MoMEvent(type="request_start", request_id="rid-1", timestamp=1.0, data={})
    await publisher.publish(event)


@pytest.mark.asyncio
async def test_publish_uses_mom_events_channel(monkeypatch):
    mock_client = SimpleNamespace(publish=AsyncMock())

    def fake_from_url(_url, decode_responses=True):
        assert decode_responses is True
        return mock_client

    monkeypatch.setattr("mom_service.events.redis.from_url", fake_from_url)

    publisher = RedisPublisher(redis_url="redis://localhost:6379")
    event = MoMEvent(type="request_complete", request_id="rid-2", timestamp=2.0, data={"ok": True})

    await publisher.publish(event)

    mock_client.publish.assert_awaited_once()
    call_args = mock_client.publish.await_args.args
    assert call_args[0] == "mom_events"
    assert '"type":"request_complete"' in call_args[1]
    assert '"request_id":"rid-2"' in call_args[1]


@pytest.mark.asyncio
async def test_publish_swallow_errors(monkeypatch):
    mock_client = SimpleNamespace(publish=AsyncMock(side_effect=RuntimeError("redis down")))

    def fake_from_url(_url, decode_responses=True):
        assert decode_responses is True
        return mock_client

    monkeypatch.setattr("mom_service.events.redis.from_url", fake_from_url)

    publisher = RedisPublisher(redis_url="redis://localhost:6379")
    event = MoMEvent(type="error", request_id="rid-3", timestamp=3.0, data={"error": "boom"})

    await publisher.publish(event)
    mock_client.publish.assert_awaited_once()
