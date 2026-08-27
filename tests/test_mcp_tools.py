"""The MCP tool surface: what each tool returns, and how consult reports every kind of outcome."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from textwrap import dedent

from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError
import pytest
import yaml

from mom.adapters.eventbus import InMemoryEventBus, RunIndexBus
from mom.api.deps import Container
from mom.api.mcp.consult import INLINE_ENSEMBLE, RunObserver, with_progress
from mom.api.mcp.server import build_mcp_server
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.events import MemberAbandoned, StreamEvent, SynthesisStarted
from mom.engine.coalesce import CoalesceRegistry
from mom.runtime.settings import Settings
from mom.store.cache import SqliteCacheStore
from mom.store.metrics import CallMetric, MetricsStore
from mom.testing import FakeLLM, ManualClock, SequentialIds


CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a, capabilities: { context_length: 128000, vision: false } }
      b: { model: openai/b, search: { enabled: true } }
    ensembles:
      e:
        description: the panel
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


def _catalog(text: str = CONFIG):
    return resolve_catalog(Config.model_validate(yaml.safe_load(text)))


def _container(*, client=None, config: str = CONFIG, bus=None, **kwargs) -> Container:
    return Container(
        settings=Settings(_env_file=None),
        catalog=_catalog(config),
        client=client or FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        bus=bus if bus is not None else RunIndexBus(InMemoryEventBus()),
        **kwargs,
    )


def _server(container: Container):
    return build_mcp_server(lambda: container)


async def test_lists_the_six_tools_with_output_schemas():
    async with Client(_server(_container()), raise_exceptions=True) as client:
        listed = await client.list_tools()
    names = [tool.name for tool in listed.tools]
    assert names == ["list_llms", "list_ensembles", "consult", "runs", "usage", "cache_stats"]
    assert all(tool.output_schema is not None for tool in listed.tools)
    # Everything but consult is a read-only view; a leaked token must not be able to spend or
    # destroy through this surface.
    read_only = {t.name: t.annotations.read_only_hint for t in listed.tools if t.annotations}
    assert read_only == {
        "list_llms": True,
        "list_ensembles": True,
        "consult": False,
        "runs": True,
        "usage": True,
        "cache_stats": True,
    }


async def test_list_llms_projects_the_catalog():
    result = await _server(_container()).call_tool("list_llms", {})
    llms = {entry["name"]: entry for entry in result.structured_content["result"]}
    assert llms["a"]["model"] == "openai/a"
    assert llms["a"]["context_length"] == 128000
    assert llms["a"]["vision"] is False
    assert llms["b"]["web_search"] is True  # a `search:` block, not a capability flag


async def test_list_llms_falls_back_to_the_litellm_price_list(monkeypatch: pytest.MonkeyPatch):
    from mom.api.mcp import server as server_module

    monkeypatch.setattr(
        server_module, "_catalogue_pricing", lambda model: {"input_per_1m": 1.5} if model else None
    )
    result = await _server(_container()).call_tool("list_llms", {})
    entry = result.structured_content["result"][0]
    assert entry["pricing"] == {
        "input_per_1m": 1.5,
        "output_per_1m": None,
        "reasoning_per_1m": None,
        "cache_read_per_1m": None,
        "cache_write_per_1m": None,
    }
    assert entry["pricing_source"] == "litellm"


async def test_list_llms_prefers_configured_pricing(monkeypatch: pytest.MonkeyPatch):
    from mom.api.mcp import server as server_module

    monkeypatch.setattr(server_module, "_catalogue_pricing", lambda model: {"input_per_1m": 99.0})
    priced = CONFIG.replace(
        "a: { model: openai/a,", "a: { model: openai/a, pricing: { input_per_1m: 2.0 },"
    )
    result = await _server(_container(config=priced)).call_tool("list_llms", {})
    entry = next(e for e in result.structured_content["result"] if e["name"] == "a")
    # Config pricing is what mom actually bills against, so it wins the display too.
    assert entry["pricing"]["input_per_1m"] == 2.0
    assert entry["pricing_source"] == "config"


async def test_list_ensembles_reports_the_panel():
    result = await _server(_container()).call_tool("list_ensembles", {})
    (ensemble,) = result.structured_content["result"]
    assert ensemble["name"] == "e"
    assert ensemble["description"] == "the panel"
    assert [m["llm"] for m in ensemble["members"]] == ["a", "b"]
    assert ensemble["synthesizer"]["llm"] == "a"
    assert ensemble["context_length"] == 128000  # the card's min-across-panel aggregate


async def test_consult_returns_the_synthesized_answer_and_member_costs():
    container = _container()
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    payload = result.structured_content

    assert result.is_error is False
    assert payload["status"] == "ok"
    assert payload["answer"] == "synthesized answer"
    assert payload["ensemble"] == "e"
    assert {m["identity"] for m in payload["members"]} == {"a", "b"}
    assert all(m["status"] == "ok" for m in payload["members"])
    # Member text is heavy and rarely wanted; it is opt-in.
    assert all(m["answer"] is None for m in payload["members"])


async def test_consult_includes_member_answers_on_request():
    result = await _server(_container()).call_tool(
        "consult", {"prompt": "hi", "ensemble": "e", "include_member_answers": True}
    )
    members = result.structured_content["members"]
    assert {m["answer"] for m in members} == {"reply from a", "reply from b"}


async def test_consult_runs_an_inline_panel_without_touching_the_catalog():
    container = _container()
    result = await _server(container).call_tool(
        "consult", {"prompt": "hi", "panel": ["a", "b"], "synthesizer": "a"}
    )
    assert result.structured_content["status"] == "ok"
    assert result.structured_content["ensemble"] == INLINE_ENSEMBLE
    assert {m["identity"] for m in result.structured_content["members"]} == {"a", "b"}
    # The panel lived for the call only: nothing was added to the process catalog.
    assert INLINE_ENSEMBLE not in container.catalog.ensembles


async def test_inline_panels_never_coalesce_onto_each_other():
    """Two different panels asking the same question must not share one run.

    The coalescing key is the ensemble NAME plus the messages, and every inline panel runs under
    the same synthetic name — so without opting out, the second panel would silently receive the
    first panel's answer from a roster it never asked for.
    """
    dedupe_on = CONFIG.replace(
        "server: { auth: none }", "server: { auth: none, dedupe: {enabled: true} }"
    )
    container = _container(config=dedupe_on, coalesce=CoalesceRegistry())
    server = _server(container)

    first, second = await asyncio.gather(
        server.call_tool("consult", {"prompt": "same", "panel": ["a"], "synthesizer": "a"}),
        server.call_tool("consult", {"prompt": "same", "panel": ["b"], "synthesizer": "a"}),
    )
    assert first.structured_content["coalesced"] is False
    assert second.structured_content["coalesced"] is False
    assert first.structured_content["members"][0]["identity"] == "a"
    assert second.structured_content["members"][0]["identity"] == "b"


async def test_named_ensembles_still_coalesce():
    dedupe_on = CONFIG.replace(
        "server: { auth: none }", "server: { auth: none, dedupe: {enabled: true} }"
    )
    container = _container(config=dedupe_on, coalesce=CoalesceRegistry())
    server = _server(container)

    first, second = await asyncio.gather(
        server.call_tool("consult", {"prompt": "same", "ensemble": "e"}),
        server.call_tool("consult", {"prompt": "same", "ensemble": "e"}),
    )
    coalesced = [first.structured_content["coalesced"], second.structured_content["coalesced"]]
    assert coalesced.count(True) == 1  # the follower attached to the leader's run
    # ...and reports the leader's id, so its progress link points at the run doing the work.
    assert first.structured_content["request_id"] == second.structured_content["request_id"]


async def test_consult_reports_a_tool_call_as_its_own_outcome():
    client = FakeLLM(
        tool_calls=({"id": "c1", "name": "lookup", "arguments": '{"q":1}'},),
        finish_reason="tool_calls",
    )
    result = await _server(_container(client=client)).call_tool(
        "consult",
        {
            "prompt": "hi",
            "ensemble": "e",
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        },
    )
    payload = result.structured_content

    assert result.is_error is False  # a tool call is a successful run, not a failure
    assert payload["status"] == "tool_calls"
    assert payload["answer"] == ""
    assert payload["finish_reason"] == "tool_calls"
    assert payload["tool_calls"][0]["function"]["name"] == "lookup"
    # A text-only client still learns what happened rather than seeing an empty answer.
    assert "lookup" in result.content[0].text


async def test_consult_reports_an_upstream_failure_with_the_spend_it_incurred():
    container = _container(client=FakeLLM(fail=frozenset({"a", "b"})))
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    payload = result.structured_content

    assert result.is_error is True
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "quorum_not_met"
    assert payload["error"]["http_status"] == 502
    # `collect` raised, so these came from the events observed before it did — a failed panel
    # still has to account for the calls it made.
    assert {m["identity"] for m in payload["members"]} == {"a", "b"}
    assert all(m["status"] == "error" for m in payload["members"])


async def test_consult_never_leaks_operator_only_error_detail():
    container = _container(client=FakeLLM(fail=frozenset({"a"})))
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    failed = next(m for m in result.structured_content["members"] if m["identity"] == "a")
    assert failed["error"] == "a failed"  # the client-safe half
    assert "error_kind" not in failed
    assert "error_detail" not in failed


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"prompt": "x"}, "exactly one"),
        ({"prompt": "x", "ensemble": "e", "panel": ["a"]}, "exactly one"),
        ({"prompt": "x", "panel": ["a"]}, "synthesizer"),
        ({"prompt": "x", "panel": [], "synthesizer": "a"}, "at least one"),
        ({"prompt": "x", "panel": ["a", "a"], "synthesizer": "a"}, "more than once"),
        ({"prompt": "x", "panel": ["nope"], "synthesizer": "a"}, "unknown llm"),
        ({"prompt": "x", "ensemble": "nope"}, "nope"),
    ],
)
async def test_consult_rejects_a_malformed_call(arguments: dict, expected: str):
    """A caller mistake is a protocol error: the agent has to change the call, not read an
    outcome. Nothing is spent."""
    container = _container()
    with pytest.raises(ToolError, match=expected):
        await _server(container).call_tool("consult", arguments)
    assert container.client.completions == []  # type: ignore[attr-defined]


async def test_consult_reports_progress_per_member():
    container = _container()
    seen: list[tuple[float, float | None, str | None]] = []

    async with Client(_server(container), raise_exceptions=True) as client:
        await client.call_tool(
            "consult",
            {"prompt": "hi", "ensemble": "e"},
            progress_callback=lambda progress, total, message: seen.append(
                (progress, total, message)
            ),
        )

    assert [p for p, _, _ in seen] == sorted({p for p, _, _ in seen})  # strictly increasing
    assert any("a: ok" in (m or "") for _, _, m in seen)
    assert any("synthesizing" in (m or "") for _, _, m in seen)
    # Synthesis is usually the longest wait; the bar must not sit at 100% through it.
    assert all(progress < (total or 0) for progress, total, _ in seen)


async def test_progress_reporting_never_breaks_the_run():
    """`report_progress` raises with no request context (a direct call) and no-ops without a
    client progress token. Neither is a reason to abandon a fan-out."""

    class Boom:
        async def report_progress(self, *_args, **_kwargs):
            raise RuntimeError("no progress channel")

    async def events():
        yield SynthesisStarted("a", "openai/a")
        yield MemberAbandoned("b", "openai/b")

    observer = RunObserver()
    collected: list[StreamEvent] = [
        event async for event in with_progress(events(), Boom(), observer, total=3)
    ]
    assert len(collected) == 2
    assert observer.abandoned == [("b", "openai/b")]


async def test_consult_lists_a_member_the_deadline_abandoned():
    """An abandoned member produces no outcome at all, so without this it would simply vanish —
    the panel would report fewer members than the caller asked for, silently."""
    slow = CONFIG.replace("version: 2", "version: 2\ndefaults: { fanout: { deadline: 0.01s } }")
    container = _container(config=slow, client=FakeLLM(delays={"b": 5.0}))
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    members = {m["identity"]: m for m in result.structured_content["members"]}
    assert members["a"]["status"] == "ok"
    assert members["b"]["status"] == "abandoned"


async def test_an_aliased_member_is_labelled_the_same_way_however_it_ends():
    """An `as:` alias is what the engine calls a member (`ModelOutcome.llm` is the identity, and
    so is the metrics `llm` column). An abandoned member has no outcome to read that off, so it
    is the one place the two could disagree."""
    aliased = CONFIG.replace(
        "members: [{ llm: a }, { llm: b }]",
        "members: [{ llm: a, as: fast }, { llm: b, as: slow-reviewer }]",
    ).replace("version: 2", "version: 2\ndefaults: { fanout: { deadline: 0.01s } }")
    container = _container(config=aliased, client=FakeLLM(delays={"slow-reviewer": 5.0}))
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    members = {m["identity"]: m for m in result.structured_content["members"]}

    assert members["slow-reviewer"]["status"] == "abandoned"
    assert members["slow-reviewer"]["model"] == "openai/b"  # the model that actually ran
    # Identity in both fields, for the member that finished and the one that didn't alike.
    assert members["slow-reviewer"]["llm"] == "slow-reviewer"
    assert members["fast"]["llm"] == "fast"


async def test_consult_forwards_strict_tool_schemas():
    """The wire surfaces forward `strict`; dropping it here would weaken schema enforcement for
    the same tool definition depending on which surface the caller used."""
    container = _container()
    await _server(container).call_tool(
        "consult",
        {
            "prompt": "hi",
            "ensemble": "e",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )
    synth_call = container.client.streams[-1]  # type: ignore[attr-defined]
    assert synth_call.params["tools"][0]["function"]["strict"] is True


async def test_consult_never_hands_back_the_api_token():
    """The progress link is tool-result *data* — it lands in a model's context and travels with
    that agent's transcript. Over stdio the caller never even presented a token."""
    authed = CONFIG.replace(
        "server: { auth: none }",
        "server: { auth: bearer, public_url: 'https://mom.example.com' }",
    )
    container = Container(
        settings=Settings(_env_file=None, MOM_API_TOKEN="super-secret"),
        catalog=_catalog(authed),
        client=FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        bus=RunIndexBus(InMemoryEventBus()),
    )
    result = await _server(container).call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    url = result.structured_content["progress_url"]
    assert url == "https://mom.example.com/v1/progress/req-1"
    assert "super-secret" not in json.dumps(result.structured_content)


async def test_an_inline_panel_cannot_shadow_a_configured_ensemble():
    """The inline name lives in config's reserved-character namespace, so no configured ensemble
    can ever collide with it — nor lose its own coalescing by sharing the name."""
    assert ":" in INLINE_ENSEMBLE
    with pytest.raises(ValidationError, match="reserved characters"):
        Config.model_validate(yaml.safe_load(CONFIG.replace("  e:\n", f"  {INLINE_ENSEMBLE}:\n")))


async def test_runs_reports_live_and_finished_runs(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "metrics.db")
    await store.insert_many(
        [
            CallMetric(
                request_id="req-old",
                ts=1000.0,
                ensemble="e",
                llm="a",
                model="openai/a",
                role="fanout",
                status="ok",
                cost_usd=0.25,
            )
        ]
    )
    try:
        container = _container(metrics_reader=store)
        # A run the bus has seen start but not finish.
        container.bus.publish(  # type: ignore[union-attr]
            "req-live",
            _fanout_started(),
        )
        report = (await _server(container).call_tool("runs", {})).structured_content
        assert [r["request_id"] for r in report["in_flight"]] == ["req-live"]
        assert [r["request_id"] for r in report["recent"]] == ["req-old"]
        assert report["recent"][0]["cost_usd"] == 0.25
        assert report["in_flight_visibility"] == "process"

        detail = (
            await _server(container).call_tool("runs", {"request_id": "req-old"})
        ).structured_content
        assert [c["llm"] for c in detail["calls"]] == ["a"]
        assert detail["recent"] == []
    finally:
        await store.close()


def _fanout_started():
    from mom.domain.progress import ProgressEvent

    return ProgressEvent(
        kind="fanout_started",
        ensemble="e",
        members=(("a", "openai/a"), ("b", "openai/b")),
        members_total=2,
    )


async def test_runs_admits_when_it_cannot_see_live_runs():
    """A Redis-backed deployment has no in-flight view (pub/sub retains nothing). Saying so is
    the point — an empty list would read as "nothing is running"."""
    container = _container(bus=InMemoryEventBus())  # not wrapped in a run index
    report = (await _server(container).call_tool("runs", {})).structured_content
    assert report["in_flight"] == []
    assert report["in_flight_visibility"] == "none"


async def test_usage_aggregates_the_window(tmp_path: Path):
    store = await MetricsStore.open(tmp_path / "metrics.db")
    await store.insert_many(
        [
            CallMetric(
                request_id="r1",
                ts=1000.0,
                ensemble="e",
                llm="a",
                model="openai/a",
                role="fanout",
                status="ok",
                cost_usd=0.5,
            ),
            CallMetric(
                request_id="r1",
                ts=1001.0,
                ensemble="e",
                llm="b",
                model="openai/b",
                role="fanout",
                status="error",
                cost_usd=0.25,
            ),
        ]
    )
    try:
        server = _server(_container(metrics_reader=store))
        report = (await server.call_tool("usage", {"days": 0})).structured_content
        assert report["window_days"] is None  # 0 means all time, as in the CLI
        assert report["calls"] == 2
        assert report["cost_usd"] == 0.75
        assert report["errors"] == 1
        assert [g["key"] for g in report["by_ensemble"]] == ["e"]
        assert {g["key"] for g in report["by_llm"]} == {"a", "b"}
    finally:
        await store.close()


@pytest.mark.parametrize("limit", [-1, 0, 10_000])
async def test_runs_clamps_the_limit_it_is_given(tmp_path: Path, limit: int):
    """SQLite reads a negative LIMIT as unlimited, so an unclamped value would group and
    serialize every run the ledger holds — a stall any token holder could trigger at will."""
    store = await MetricsStore.open(tmp_path / "metrics.db")
    await store.insert_many(
        [
            CallMetric(
                request_id=f"r{i}",
                ts=float(i),
                ensemble="e",
                llm="a",
                model="openai/a",
                role="fanout",
                status="ok",
            )
            for i in range(30)
        ]
    )
    try:
        server = _server(_container(metrics_reader=store))
        report = (await server.call_tool("runs", {"limit": limit})).structured_content
        assert 1 <= len(report["recent"]) <= 200
    finally:
        await store.close()


async def test_usage_windows_against_the_containers_clock(tmp_path: Path):
    """Metric rows are stamped from the injected clock, so windowing against wall time would
    filter them against a different timeline — reporting $0 on a gateway that had spent."""
    store = await MetricsStore.open(tmp_path / "metrics.db")
    clock = ManualClock(start=100_000.0)  # far from wall time, so the two cannot coincide
    await store.insert_many(
        [
            CallMetric(
                request_id="r1",
                ts=clock.now() - 3600,  # an hour ago on the container's clock
                ensemble="e",
                llm="a",
                model="openai/a",
                role="fanout",
                status="ok",
                cost_usd=0.5,
            )
        ]
    )
    try:
        container = _container(metrics_reader=store)
        report = (
            await build_mcp_server(lambda: replace(container, clock=clock)).call_tool(
                "usage", {"days": 7}
            )
        ).structured_content
        assert report["calls"] == 1
        assert report["cost_usd"] == 0.5
    finally:
        await store.close()


async def test_usage_says_so_when_metrics_are_off():
    report = (await _server(_container()).call_tool("usage", {})).structured_content
    assert report["calls"] == 0
    assert "not enabled" in report["note"]


async def test_cache_stats_reads_the_live_store(tmp_path: Path):
    store = await SqliteCacheStore.open(tmp_path / "cache.db", ttl_seconds=60, max_bytes=1 << 20)
    try:
        await store.put("k", "a", "body", now=0.0)
        report = (
            await _server(_container(cache_store=store)).call_tool("cache_stats", {})
        ).structured_content
        assert report == {"enabled": True, "entries": 1, "bytes": len("body"), "hits": 0}
    finally:
        await store.close()


async def test_cache_stats_when_caching_is_disabled():
    report = (await _server(_container()).call_tool("cache_stats", {})).structured_content
    assert report == {"enabled": False, "entries": 0, "bytes": 0, "hits": 0}
