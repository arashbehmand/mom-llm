from __future__ import annotations

# pylint: disable=redefined-outer-name,unused-argument
import pytest

from mom_service.reporting.main import active_requests, apply_event_to_state, get_progress_status


@pytest.fixture
def clean_active_requests():
    active_requests.clear()
    yield
    active_requests.clear()


def test_apply_event_to_state_happy_path(clean_active_requests):
    request_id = "req-123"

    apply_event_to_state(
        active_requests,
        {
            "type": "request_start",
            "request_id": request_id,
            "timestamp": 100.0,
            "data": {"model_requested": "mom", "num_messages": 1},
        },
    )
    apply_event_to_state(
        active_requests,
        {
            "type": "fanout_start",
            "request_id": request_id,
            "timestamp": 101.0,
            "data": {"model": "oai-g4.1"},
        },
    )
    apply_event_to_state(
        active_requests,
        {
            "type": "fanout_complete",
            "request_id": request_id,
            "timestamp": 102.0,
            "data": {"model": "oai-g4.1", "status": "success", "tokens": 321, "error": None},
        },
    )
    apply_event_to_state(
        active_requests,
        {
            "type": "concluding_start",
            "request_id": request_id,
            "timestamp": 103.0,
            "data": {"concluding_llm": "g25f"},
        },
    )
    apply_event_to_state(
        active_requests,
        {
            "type": "request_complete",
            "request_id": request_id,
            "timestamp": 104.0,
            "data": {},
        },
    )

    state = active_requests[request_id]
    assert state["status"] == "completed"
    assert state["fanouts"][0]["model"] == "oai-g4.1"
    assert state["fanouts"][0]["status"] == "success"
    assert state["fanouts"][0]["tokens"] == 321
    assert state["concluding"]["model"] == "g25f"
    assert state["concluding"]["status"] == "completed"


def test_apply_event_to_state_ignores_missing_request_id(clean_active_requests):
    apply_event_to_state(
        active_requests,
        {"type": "request_start", "timestamp": 100.0, "data": {"model_requested": "mom"}},
    )
    assert not active_requests


@pytest.mark.asyncio
async def test_get_progress_status_from_active_requests(clean_active_requests):
    request_id = "req-live"
    apply_event_to_state(
        active_requests,
        {
            "type": "request_start",
            "request_id": request_id,
            "timestamp": 100.0,
            "data": {"model_requested": "mom", "num_messages": 1},
        },
    )
    status = await get_progress_status(request_id)
    assert status["request_id"] == request_id
    assert status["status"] == "processing"


@pytest.mark.asyncio
async def test_get_progress_status_unknown_returns_default(clean_active_requests, monkeypatch):
    monkeypatch.setattr("mom_service.reporting.main.metrics_db.query_metrics", lambda limit=100: [])
    status = await get_progress_status("missing")
    assert status["request_id"] == "missing"
    assert status["status"] == "unknown"
    assert status["fanouts"] == []
    assert status["concluding"] is None
