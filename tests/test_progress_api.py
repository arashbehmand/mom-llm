"""Progress SSE: the endpoint streams bus events, and the pipeline publishes them end-to-end."""

from __future__ import annotations

import contextlib
import json
from textwrap import dedent

import httpx
import yaml

from mom.adapters.eventbus import InMemoryEventBus
from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec, CompletionChunk
from mom.domain.progress import ProgressEvent
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _catalog():
    return resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))


def _plan():
    return resolve_plan(_catalog(), ChatRequestIR(model="e", messages=(MessageIR("user", "hi"),)))


def _container(bus: InMemoryEventBus, *, client=None) -> Container:
    return Container(
        settings=Settings(_env_file=None),
        catalog=_catalog(),
        client=client or FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        bus=bus,
    )


AUTH_CONFIG = dedent("""
    version: 2
    server: { auth: bearer }
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _authed_container(bus: InMemoryEventBus) -> Container:
    return Container(
        settings=Settings(_env_file=None, MOM_API_TOKEN="secret-token"),
        catalog=resolve_catalog(Config.model_validate(yaml.safe_load(AUTH_CONFIG))),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        bus=bus,
    )


def _asgi(container: Container) -> httpx.AsyncClient:
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _parse_sse(body: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):  # skip blanks and heartbeat comments
            continue
        frame: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                frame["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                frame["data"] = line[len("data:") :].strip()
        if frame:
            frames.append(frame)
    return frames


# ---------------------------------------------------------------------------------------------
# The pipeline publishes the milestone sequence onto the bus
# ---------------------------------------------------------------------------------------------
async def test_pipeline_publishes_full_progress_sequence():
    bus = InMemoryEventBus()
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), bus=bus, request_id="req-1")
    await collect(run_ensemble(_plan(), deps))

    events = [event async for event in bus.subscribe("req-1")]
    assert [e.kind for e in events] == [
        "fanout_started",
        "member_completed",
        "member_completed",
        "synthesis_started",
        "completed",
    ]
    assert events[0].members_total == 2
    first_member = events[1]
    assert first_member.member in {"a", "b"}
    assert first_member.status == "ok"
    assert first_member.completed == 1
    assert events[2].completed == 2
    assert events[-1].status == "stop"  # finish_reason carried on completion


async def test_pipeline_publishes_failed_when_synthesis_errors():
    class _SynthDown(FakeLLM):
        async def stream(self, spec: CallSpec):
            raise UpstreamError("synth down")
            yield CompletionChunk()  # pragma: no cover - marks this an async generator

    bus = InMemoryEventBus()
    deps = PipelineDeps(client=_SynthDown(), clock=ManualClock(), bus=bus, request_id="req-2")
    with contextlib.suppress(Exception):
        await collect(run_ensemble(_plan(), deps))

    events = [event async for event in bus.subscribe("req-2")]
    kinds = [e.kind for e in events]
    assert kinds[0] == "fanout_started"
    assert kinds[-1] == "failed"
    assert events[-1].detail  # a human-readable reason is attached


async def test_no_bus_is_a_silent_no_op():
    # A pipeline wired without a bus must still run to completion.
    deps = PipelineDeps(client=FakeLLM(), clock=ManualClock(), request_id="req-3")
    result = await collect(run_ensemble(_plan(), deps))
    assert result.text == "synthesized answer"


# ---------------------------------------------------------------------------------------------
# GET /v1/progress/{id} — SSE
# ---------------------------------------------------------------------------------------------
async def test_progress_endpoint_replays_published_events():
    bus = InMemoryEventBus()
    for event in (
        ProgressEvent(kind="fanout_started", ensemble="e", members_total=2),
        ProgressEvent(kind="member_completed", ensemble="e", member="a", status="ok"),
        ProgressEvent(kind="synthesis_started", ensemble="e", member="a"),
        ProgressEvent(kind="completed", ensemble="e", status="stop"),
    ):
        bus.publish("req-x", event)

    async with _asgi(_container(bus)) as client:
        resp = await client.get("/v1/progress/req-x")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(resp.text)
    assert [f["event"] for f in frames] == [
        "fanout_started",
        "member_completed",
        "synthesis_started",
        "completed",
    ]
    assert json.loads(frames[0]["data"])["members_total"] == 2


async def test_chat_then_progress_end_to_end():
    bus = InMemoryEventBus()
    async with _asgi(_container(bus)) as client:
        chat = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Request-Id": "req-42"},
        )
        assert chat.status_code == 200
        assert chat.headers["x-request-id"] == "req-42"  # echoed for correlation

        progress = await client.get("/v1/progress/req-42")

    kinds = [f["event"] for f in _parse_sse(progress.text)]
    assert kinds == [
        "fanout_started",
        "member_completed",
        "member_completed",
        "synthesis_started",
        "completed",
    ]


async def test_progress_endpoint_without_bus_returns_empty_stream():
    container = Container(
        settings=Settings(_env_file=None),
        catalog=_catalog(),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
    )  # no bus wired
    async with _asgi(container) as client:
        resp = await client.get("/v1/progress/whatever")
    assert resp.status_code == 200
    assert resp.text == ""


# ---------------------------------------------------------------------------------------------
# Browser-friendly HTML page (content-negotiated on the same endpoint)
# ---------------------------------------------------------------------------------------------
async def test_progress_html_page_for_browser_accept():
    bus = InMemoryEventBus()
    async with _asgi(_container(bus)) as client:
        resp = await client.get("/v1/progress/req-y", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "req-y" in resp.text
    assert "EventSource" in resp.text


async def test_progress_default_accept_still_returns_sse():
    bus = InMemoryEventBus()
    bus.publish("req-y", ProgressEvent(kind="completed", ensemble="e", status="stop"))
    async with _asgi(_container(bus)) as client:
        resp = await client.get("/v1/progress/req-y")
    assert resp.headers["content-type"].startswith("text/event-stream")


async def test_progress_html_page_escapes_request_id():
    bus = InMemoryEventBus()
    async with _asgi(_container(bus)) as client:
        resp = await client.get(
            "/v1/progress/<script>alert(1)</script>", headers={"accept": "text/html"}
        )
    assert "<script>alert" not in resp.text


# ---------------------------------------------------------------------------------------------
# Progress URL header + query-token auth (a plain link can't carry an Authorization header)
# ---------------------------------------------------------------------------------------------
async def test_chat_completions_carries_progress_url_header():
    bus = InMemoryEventBus()
    async with _asgi(_container(bus)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
        )
    url = resp.headers["x-mom-progress-url"]
    assert url.endswith(f"/v1/progress/{resp.headers['x-request-id']}")


async def test_progress_url_header_carries_token_when_auth_enabled():
    bus = InMemoryEventBus()
    async with _asgi(_authed_container(bus)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "e", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer secret-token"},
        )
    url = resp.headers["x-mom-progress-url"]
    assert "token=secret-token" in url


async def test_progress_query_token_auth():
    bus = InMemoryEventBus()
    for event in (
        ProgressEvent(kind="fanout_started", ensemble="e", members_total=2),
        ProgressEvent(kind="completed", ensemble="e", status="stop"),
    ):
        bus.publish("req-secure", event)

    async with _asgi(_authed_container(bus)) as client:
        denied = await client.get("/v1/progress/req-secure")
        assert denied.status_code == 401

        wrong = await client.get("/v1/progress/req-secure", params={"token": "nope"})
        assert wrong.status_code == 401

        via_header = await client.get(
            "/v1/progress/req-secure", headers={"authorization": "Bearer secret-token"}
        )
        assert via_header.status_code == 200

        via_query = await client.get("/v1/progress/req-secure", params={"token": "secret-token"})
        assert via_query.status_code == 200
