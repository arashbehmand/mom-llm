"""Tool-calling depth (issue #14): candidate envelope, vote/first, id custody, compat profiles.

Pure helpers, the engine pipeline (with a fake provider), and the ASGI chat surface.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

import httpx
import yaml

from mom.api.app import create_app
from mom.api.encoders.chat import resolve_stream_profile
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.request import ChatRequestIR, MessageIR
from mom.domain.results import ModelOutcome, Usage
from mom.domain.tooling import (
    provider_supports_remote_mcp,
    restore_provider_tool_ids,
    select_member_tool_call,
    summarize_member_tool_calls,
    tool_call_signature,
)
from mom.engine.pipeline import PipelineDeps, collect, run_ensemble
from mom.engine.plan import resolve_plan
from mom.runtime.container import Container
from mom.runtime.custody import InMemoryToolCallCustody
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
]


def _catalog(text: str) -> Any:
    return resolve_catalog(Config.model_validate(yaml.safe_load(dedent(text))))


def _outcome(identity: str, calls: tuple[dict[str, Any], ...]) -> ModelOutcome:
    return ModelOutcome(identity=identity, llm=identity, model="m", status="ok", tool_calls=calls)


def _call(name: str, arguments: str, call_id: str = "x") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ------------------------------------------------------------------------------------------------
# pure helpers
# ------------------------------------------------------------------------------------------------
def test_tool_call_signature_normalizes_arguments():
    a = _call("get_weather", '{"city": "SF", "unit": "c"}')
    b = _call("get_weather", '{"unit":"c","city":"SF"}')  # different key order + spacing
    assert tool_call_signature(a) == tool_call_signature(b)
    assert tool_call_signature(a)[0] == "get_weather"
    # invalid JSON falls back to trimmed raw text, never raises
    assert tool_call_signature(_call("f", "  not json  ")) == ("f", "not json")


def test_summarize_member_tool_calls():
    outcomes = [
        _outcome("a", (_call("get_weather", '{"city":"SF"}'),)),
        _outcome("b", ()),
    ]
    summary = summarize_member_tool_calls(outcomes)
    assert summary is not None
    assert "a proposed get_weather" in summary
    assert '{"city":"SF"}' in summary
    # None when nobody proposed anything (no empty block appended to synthesis)
    assert summarize_member_tool_calls([_outcome("a", ()), _outcome("b", ())]) is None


def test_select_first_returns_first_members_calls():
    outcomes = [
        _outcome("a", (_call("get_weather", "{}"), _call("get_time", "{}"))),
        _outcome("b", (_call("get_stock", "{}"),)),
    ]
    selected = select_member_tool_call(outcomes, strategy="first", threshold=2)
    assert selected is not None
    assert [c["function"]["name"] for c in selected] == ["get_weather", "get_time"]


def test_select_first_none_when_no_proposals():
    assert select_member_tool_call([_outcome("a", ())], strategy="first", threshold=2) is None


def test_select_vote_meets_threshold():
    outcomes = [
        _outcome("a", (_call("get_weather", '{"city":"SF"}', "a1"),)),
        _outcome("b", (_call("get_weather", '{"city": "SF"}', "b1"),)),  # equal after normalize
        _outcome("c", (_call("get_time", "{}", "c1"),)),
    ]
    selected = select_member_tool_call(outcomes, strategy="vote", threshold=2)
    assert selected is not None
    assert len(selected) == 1
    assert selected[0]["function"]["name"] == "get_weather"


def test_select_vote_below_threshold_returns_none():
    outcomes = [
        _outcome("a", (_call("get_weather", "{}"),)),
        _outcome("b", (_call("get_time", "{}"),)),
    ]
    assert select_member_tool_call(outcomes, strategy="vote", threshold=2) is None


def test_select_vote_ignores_duplicate_self_votes():
    # one member proposing the same call twice must not clear a threshold of 2 on its own
    outcomes = [_outcome("a", (_call("get_weather", "{}"), _call("get_weather", "{}")))]
    assert select_member_tool_call(outcomes, strategy="vote", threshold=2) is None


def test_restore_provider_tool_ids():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "mint_1", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "mint_1", "content": "ok"},
    ]
    restored = restore_provider_tool_ids(messages, {"mint_1": "prov__thought__x"}.get)
    assert restored[1]["tool_calls"][0]["id"] == "prov__thought__x"
    assert restored[2]["tool_call_id"] == "prov__thought__x"
    # pure: inputs untouched
    assert messages[1]["tool_calls"][0]["id"] == "mint_1"
    # unknown ids are left as-is
    kept = restore_provider_tool_ids(messages, {}.get)
    assert kept[2]["tool_call_id"] == "mint_1"


def test_provider_supports_remote_mcp():
    assert provider_supports_remote_mcp("openai/gpt-5", "responses") is True
    assert provider_supports_remote_mcp("azure/gpt-5", "responses") is True
    assert provider_supports_remote_mcp("openai/gpt-5", "chat") is False  # chat api can't forward
    assert provider_supports_remote_mcp("gemini/g", "responses") is False  # provider not capable


# ------------------------------------------------------------------------------------------------
# engine pipeline: candidate envelope + vote/first + minting/custody
# ------------------------------------------------------------------------------------------------
ARBITRATE = """
    version: 2
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
      s: { model: openai/s }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: s }
"""


def _ir(model: str = "e", messages: tuple[MessageIR, ...] | None = None) -> ChatRequestIR:
    from mom.domain.request import ToolSpec

    return ChatRequestIR(
        model=model,
        messages=messages or (MessageIR(role="user", content="weather?"),),
        tools=(ToolSpec(name="get_weather", parameters={"type": "object"}),),
    )


async def test_member_candidate_envelope_captured_and_surfaced():
    proposal = {"id": "a1", "name": "get_weather", "arguments": '{"city":"SF"}'}
    fake = FakeLLM(member_tool_calls={"a": (proposal,)})
    plan = resolve_plan(_catalog(ARBITRATE), _ir())
    deps = PipelineDeps(client=fake, clock=ManualClock(), ids=SequentialIds())
    result = await collect(run_ensemble(plan, deps))
    # the member's proposal is captured on its outcome
    by_id = {o.identity: o for o in result.outcomes}
    assert by_id["a"].tool_calls[0]["function"]["name"] == "get_weather"
    assert by_id["a"].ok  # tool-only proposal is a real answer, not "empty"
    # and surfaced to the synthesizer as advisory context
    synth_messages = fake.streams[0].messages
    assert any(
        isinstance(m["content"], str) and "a proposed get_weather" in m["content"]
        for m in synth_messages
    )


VOTE = """
    version: 2
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
      c: { model: openai/c }
      s: { model: openai/s }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }, { llm: c }]
        synthesizer: { llm: s }
        tools: { strategy: vote, vote_threshold: 2 }
"""


async def test_vote_short_circuits_when_members_agree():
    fake = FakeLLM(
        member_tool_calls={
            "a": ({"id": "a1", "name": "get_weather", "arguments": '{"city":"SF"}'},),
            "b": ({"id": "b1", "name": "get_weather", "arguments": '{"city": "SF"}'},),
            "c": ({"id": "c1", "name": "get_time", "arguments": "{}"},),
        }
    )
    plan = resolve_plan(_catalog(VOTE), _ir())
    deps = PipelineDeps(client=fake, clock=ManualClock(), ids=SequentialIds())
    result = await collect(run_ensemble(plan, deps))
    assert result.finish_reason == "tool_calls"
    assert result.text == ""  # no synthesis text
    assert fake.streams == []  # synthesizer was never called
    assert result.tool_calls[0]["function"]["name"] == "get_weather"
    # the returned id is minted, not any member's raw id
    assert result.tool_calls[0]["id"] not in {"a1", "b1", "c1"}
    assert result.tool_calls[0]["id"].startswith("call")


async def test_vote_below_threshold_falls_back_to_synthesis():
    fake = FakeLLM(
        member_tool_calls={
            "a": ({"id": "a1", "name": "get_weather", "arguments": "{}"},),
            "b": ({"id": "b1", "name": "get_time", "arguments": "{}"},),
            "c": ({"id": "c1", "name": "get_stock", "arguments": "{}"},),
        }
    )
    plan = resolve_plan(_catalog(VOTE), _ir())
    deps = PipelineDeps(client=fake, clock=ManualClock(), ids=SequentialIds())
    result = await collect(run_ensemble(plan, deps))
    assert result.text == "synthesized answer"  # synthesizer ran
    assert len(fake.streams) == 1


FIRST = """
    version: 2
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
      s: { model: openai/s }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: s }
        tools: { strategy: first }
"""


def test_vote_first_pass_tools_to_members_arbitrate_does_not():
    # vote/first make members deciders -> they receive the real tool schemas
    vote_plan = resolve_plan(_catalog(VOTE), _ir())
    assert vote_plan.members[0].spec.params["tools"][0]["function"]["name"] == "get_weather"
    first_plan = resolve_plan(_catalog(FIRST), _ir())
    assert "tools" in first_plan.members[0].spec.params
    # arbitrate keeps members advisory -> no tool schemas, just the summary context
    arb_plan = resolve_plan(_catalog(ARBITRATE), _ir())
    assert "tools" not in arb_plan.members[0].spec.params
    assert any(
        "get_weather" in m.get("content", "")
        for m in arb_plan.members[0].spec.messages
        if isinstance(m.get("content"), str)
    )


async def test_first_returns_first_member_proposal():
    fake = FakeLLM(
        member_tool_calls={
            "a": ({"id": "a1", "name": "get_weather", "arguments": '{"city":"SF"}'},),
            "b": ({"id": "b1", "name": "get_time", "arguments": "{}"},),
        }
    )
    plan = resolve_plan(_catalog(FIRST), _ir())
    deps = PipelineDeps(client=fake, clock=ManualClock(), ids=SequentialIds())
    result = await collect(run_ensemble(plan, deps))
    assert result.finish_reason == "tool_calls"
    assert fake.streams == []
    assert result.tool_calls[0]["function"]["name"] == "get_weather"  # member a (config-first)


async def test_synth_tool_call_id_is_minted_and_custody_stores_provider_id():
    provider_id = "call_abc__thought__SIGNATURE"
    fake = FakeLLM(tool_calls=({"id": provider_id, "name": "get_weather", "arguments": "{}"},))
    custody = InMemoryToolCallCustody()
    plan = resolve_plan(_catalog(ARBITRATE), _ir())
    deps = PipelineDeps(client=fake, clock=ManualClock(), ids=SequentialIds(), custody=custody)
    result = await collect(run_ensemble(plan, deps))
    minted = result.tool_calls[0]["id"]
    assert "__thought__" not in minted
    assert minted != provider_id
    # the provider id is retrievable only for the synthesizer that minted it
    assert custody.provider_id(minted, "s") == provider_id
    assert custody.provider_id(minted, "other") is None


# ------------------------------------------------------------------------------------------------
# ASGI chat surface: streaming compat/strict profiles, User-Agent, custody round-trip
# ------------------------------------------------------------------------------------------------
def _client(catalog: Any, fake: Any, custody: Any = None) -> httpx.AsyncClient:
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
        custody=custody,
    )
    app = create_app(container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class _MultiFragmentToolClient:
    """A synthesizer that streams one tool call across several argument fragments."""

    def __init__(self) -> None:
        self.streams: list[CallSpec] = []
        self.completions: list[CallSpec] = []

    async def complete(self, spec: CallSpec) -> Completion:
        self.completions.append(spec)
        return Completion(content="m", reasoning=None, finish_reason="stop", usage=Usage())

    async def stream(self, spec: CallSpec) -> Any:
        self.streams.append(spec)
        yield CompletionChunk(tool_call={"index": 0, "id": "prov_1", "name": "get_weather"})
        yield CompletionChunk(tool_call={"index": 0, "arguments": '{"city":'})
        yield CompletionChunk(tool_call={"index": 0, "arguments": '"SF"}'})
        yield CompletionChunk(
            finish_reason="tool_calls", usage=Usage(prompt_tokens=1, completion_tokens=1)
        )


PROFILE_CFG = """
    version: 2
    server: {{ auth: none }}
    llms:
      a: {{ model: openai/a }}
      s: {{ model: openai/s }}
    ensembles:
      e:
        members: [{{ llm: a }}]
        synthesizer: {{ llm: s }}
        tools: {{ stream_profile: {profile} }}
"""


def _tool_deltas(body: str) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: ") or block.endswith("[DONE]"):
            continue
        payload = json.loads(block[len("data: ") :])
        for choice in payload.get("choices", []):
            deltas.extend(choice.get("delta", {}).get("tool_calls", []))
    return deltas


async def _stream_tool_deltas(profile: str, headers: dict[str, str] | None = None) -> list[dict]:
    catalog = _catalog(PROFILE_CFG.format(profile=profile))
    async with _client(catalog, _MultiFragmentToolClient()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": TOOLS,
                "stream": True,
            },
            headers=headers,
        )
    return _tool_deltas(resp.text)


async def test_stream_profile_compat_reemits_header_on_every_delta():
    deltas = await _stream_tool_deltas("compat")
    arg_deltas = [d for d in deltas if d.get("function", {}).get("arguments")]
    assert len(arg_deltas) == 2  # two argument fragments
    for delta in deltas:
        assert isinstance(delta["index"], int)  # index always present + int
        assert delta["function"]["arguments"] is not None  # never null
    # compat: id/type/name re-emitted on EVERY delta (header + both arg fragments)
    assert all("id" in d and d["function"].get("name") == "get_weather" for d in deltas)


async def test_stream_profile_strict_sends_header_once():
    deltas = await _stream_tool_deltas("strict")
    with_id = [d for d in deltas if "id" in d]
    assert len(with_id) == 1  # only the first (header) delta carries id/type/name
    arg_deltas = [d for d in deltas if d.get("function", {}).get("arguments")]
    assert len(arg_deltas) == 2
    for delta in arg_deltas:
        assert "id" not in delta
        assert "name" not in delta["function"]
        assert delta["function"]["arguments"] is not None  # never null


async def test_user_agent_forces_compat_over_strict_config():
    # a strict-configured ensemble is upgraded to compat for a recognized AI-SDK client
    deltas = await _stream_tool_deltas("strict", headers={"user-agent": "ai-sdk/1.2 node"})
    assert all("id" in d for d in deltas)


def test_resolve_stream_profile():
    assert resolve_stream_profile("strict", "ai-sdk/1.0") == "compat"  # UA upgrades strict
    assert resolve_stream_profile("strict", "python-httpx/0.27") == "strict"
    assert resolve_stream_profile("compat", "ai-sdk/1.0") == "compat"
    assert resolve_stream_profile("compat", None) == "compat"


RELAY_CFG = """
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
      s: { model: openai/s }
    ensembles:
      e:
        members: [{ llm: a }]
        synthesizer: { llm: s }
"""


async def test_streaming_thought_signature_never_reaches_client():
    fake = FakeLLM(
        tool_calls=({"id": "call_x__thought__SIG", "name": "get_weather", "arguments": "{}"},)
    )
    async with _client(_catalog(RELAY_CFG), fake, InMemoryToolCallCustody()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": TOOLS,
                "stream": True,
            },
        )
    assert "__thought__" not in resp.text  # the raw provider signature never leaks
    deltas = _tool_deltas(resp.text)
    minted = next(d["id"] for d in deltas if "id" in d)
    assert minted.startswith("call")
    assert "__thought__" not in minted


async def test_custody_round_trip_restores_provider_id_on_relay():
    provider_id = "call_abc__thought__SIG"
    fake = FakeLLM(tool_calls=({"id": provider_id, "name": "get_weather", "arguments": "{}"},))
    custody = InMemoryToolCallCustody()
    async with _client(_catalog(RELAY_CFG), fake, custody) as client:
        first = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": TOOLS,
            },
        )
        minted = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
        assert "__thought__" not in minted
        second = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "tools": TOOLS,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": minted,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": minted, "content": "sunny"},
                ],
            },
        )
    assert second.status_code == 200
    # turn 1 fanned out (1 member call) + streamed; turn 2 relayed straight to the synthesizer
    assert len(fake.completions) == 1
    assert len(fake.streams) == 2
    relay_spec = fake.streams[-1]  # the relay synthesizer call
    call_ids = [tc["id"] for m in relay_spec.messages for tc in (m.get("tool_calls") or [])]
    tool_result_ids = [m.get("tool_call_id") for m in relay_spec.messages]
    # the synthesizer's own provider-native id (with its __thought__ signature) was restored
    assert provider_id in call_ids
    assert provider_id in tool_result_ids


async def test_parallel_tool_calls_stay_serialized_by_index():
    fake = FakeLLM(
        tool_calls=(
            {"id": "p0", "name": "get_weather", "arguments": '{"city":"SF"}'},
            {"id": "p1", "name": "get_time", "arguments": "{}"},
        )
    )
    async with _client(_catalog(RELAY_CFG), fake) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "e",
                "messages": [{"role": "user", "content": "weather and time?"}],
                "tools": TOOLS,
                "stream": True,
            },
        )
    deltas = _tool_deltas(resp.text)
    indexes = [d["index"] for d in deltas]
    assert set(indexes) == {0, 1}
    assert indexes == sorted(indexes)  # call 0 fully precedes call 1 (serialized, monotonic)
    assert all(d["function"]["arguments"] is not None for d in deltas)
