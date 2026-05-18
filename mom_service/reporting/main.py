from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from mom_service.reporting import metrics_db
from mom_service.reporting.metrics_api import metrics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mom_reporting")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL")

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Global Redis client for subscription
redis_subscriber: redis.Redis | None = None

# In-memory store for active requests (simplified for now, could be Redis-backed)
# Structure: { request_id: { status: str, fanouts: list, concluding: dict, ... } }
active_requests: dict[str, Any] = {}


def _build_initial_request_state(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": "unknown",
        "fanouts": [],
        "concluding": None,
        "start_time": None,
    }


def apply_event_to_state(
    state_store: dict[str, Any],
    event_dict: dict[str, Any],
) -> None:
    event_type = event_dict.get("type")
    request_id = event_dict.get("request_id")
    data = event_dict.get("data", {})

    if not request_id:
        return

    if request_id not in state_store:
        state_store[request_id] = _build_initial_request_state(request_id)

    req_state = state_store[request_id]

    if event_type == "request_start":
        req_state["status"] = "processing"
        req_state["start_time"] = event_dict.get("timestamp")
        req_state["model_requested"] = data.get("model_requested")
        req_state["num_messages"] = data.get("num_messages")

    elif event_type == "fanout_start":
        existing = next((f for f in req_state["fanouts"] if f["model"] == data.get("model")), None)
        if not existing:
            req_state["fanouts"].append(
                {
                    "model": data.get("model"),
                    "status": "processing",
                    "tokens": 0,
                    "error": None,
                }
            )

    elif event_type == "fanout_complete":
        existing = next((f for f in req_state["fanouts"] if f["model"] == data.get("model")), None)
        if existing:
            existing["status"] = data.get("status")
            existing["tokens"] = data.get("tokens")
            existing["error"] = data.get("error")
        else:
            req_state["fanouts"].append(
                {
                    "model": data.get("model"),
                    "status": data.get("status"),
                    "tokens": data.get("tokens"),
                    "error": data.get("error"),
                }
            )

    elif event_type == "concluding_start":
        req_state["concluding"] = {
            "model": data.get("concluding_llm"),
            "status": "processing",
        }

    elif event_type == "request_complete":
        req_state["status"] = "completed"
        if req_state["concluding"]:
            req_state["concluding"]["status"] = "completed"

    elif event_type == "request_aborted":
        req_state["status"] = "client_dropped"
        req_state["error"] = data.get("reason", "Client disconnected")
        if req_state["concluding"] and req_state["concluding"].get("status") == "processing":
            req_state["concluding"]["status"] = "aborted"

    elif event_type == "error":
        req_state["status"] = "error"
        req_state["error"] = data.get("error")
        if req_state["concluding"] and req_state["concluding"].get("status") == "processing":
            req_state["concluding"]["status"] = "error"


async def event_listener():
    """
    Background task that listens for events on the Redis 'mom_events' channel
    and updates the local metrics database.
    """
    if not REDIS_URL:
        logger.warning("Redis URL not set. Event listener disabled.")
        return

    logger.info(f"Starting event listener on {REDIS_URL}...")

    # Use a separate connection for subscription
    sub_client = redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = sub_client.pubsub()

    try:
        await pubsub.subscribe("mom_events")
        logger.info("Subscribed to 'mom_events' channel.")

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                payload = message["data"]
                event_dict = json.loads(payload)
                apply_event_to_state(active_requests, event_dict)

            except json.JSONDecodeError:
                logger.error(f"Failed to decode event payload: {message['data']}")
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    except asyncio.CancelledError:
        logger.info("Event listener cancelled.")
    except Exception as e:
        logger.error(f"Event listener error: {e}")
    finally:
        await pubsub.unsubscribe("mom_events")
        await sub_client.close()
        logger.info("Event listener stopped.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    logger.info("Reporting Service starting up...")

    # Initialize DB (creates tables if needed)
    metrics_db.init_metrics_db()

    # Start background event listener
    task = asyncio.create_task(event_listener())

    yield

    # Shutdown
    logger.info("Reporting Service shutting down...")
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="MoM Reporting Service", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mom-reporting"}


@app.get("/progress/{request_id}", response_class=HTMLResponse)
async def progress_page(request: Request, request_id: str):
    """
    Serve the progress page for a specific request.
    """
    return templates.TemplateResponse(
        request, "progress.html", {"request_id": request_id}
    )


@app.get("/api/progress/{request_id}")
async def get_progress_status(request_id: str):
    """
    API endpoint to get current status of a request.
    Called by HTMX/polling from the frontend.
    """
    # Check in-memory active requests first
    if request_id in active_requests:
        return active_requests[request_id]

    # If not found in memory, check metrics DB (for historical/completed requests)
    # This handles cases where the reporting service restarted or the request is old
    try:
        metrics = metrics_db.query_metrics(limit=100)  # Get recent metrics to find this ID
        # Filter for this request_id
        req_metrics = [m for m in metrics if m["request_id"] == request_id]

        if req_metrics:
            # Reconstruct state from metrics
            fanouts = []
            concluding = None
            status = "completed"  # Assume completed if in DB, though could be failed

            for m in req_metrics:
                if m["call_type"] == "fanout":
                    fanouts.append(
                        {
                            "model": m["llm_name"],
                            "status": m["status"].lower(),
                            "tokens": m["total_tokens"],
                        }
                    )
                elif m["call_type"] == "concluding":
                    concluding = {"model": m["llm_name"], "status": m["status"].lower()}

            return {
                "request_id": request_id,
                "status": status,
                "fanouts": fanouts,
                "concluding": concluding,
            }
    except Exception as e:
        logger.error(f"Error querying metrics DB for {request_id}: {e}")

    # If not found anywhere
    return {
        "request_id": request_id,
        "status": "unknown",
        "fanouts": [],
        "concluding": None,
    }


app.include_router(metrics_router)
