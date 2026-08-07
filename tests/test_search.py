"""Web-search propagation and vision-based member filtering."""

from __future__ import annotations

from textwrap import dedent

import httpx
import structlog
import yaml

from mom.api.app import create_app
from mom.api.deps import Container
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.request import ChatRequestIR, ImagePart, MessageIR, ToolSpec
from mom.engine.plan import resolve_plan
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      online: { model: openai/o, search: { web_search_options: { search_context_size: high } } }
      offline: { model: openai/x }
      novis: { model: openai/nv, capabilities: { vision: false } }
    ensembles:
      s:
        members: [{ llm: online }, { llm: offline }]
        synthesizer: { llm: online }
      v:
        members: [{ llm: online }, { llm: novis }]
        synthesizer: { llm: online }
""")


def _catalog():
    return resolve_catalog(Config.model_validate(yaml.safe_load(CONFIG)))


def test_search_propagates_to_capable_members_only():
    catalog = _catalog()
    ir = ChatRequestIR(
        model="s", messages=(MessageIR(role="user", content="latest news?"),), web_search=True
    )
    plan = resolve_plan(catalog, ir)
    by_id = {m.identity: m for m in plan.members}
    assert by_id["online"].spec.params["web_search_options"] == {"search_context_size": "high"}
    assert "web_search_options" not in by_id["offline"].spec.params
    # the synthesizer (search-capable) also gets search params
    assert plan.synth.params["web_search_options"] == {"search_context_size": "high"}


def test_no_search_when_not_requested():
    catalog = _catalog()
    ir = ChatRequestIR(model="s", messages=(MessageIR(role="user", content="hi"),))
    plan = resolve_plan(catalog, ir)
    by_id = {m.identity: m for m in plan.members}
    assert "web_search_options" not in by_id["online"].spec.params


def test_image_request_filters_non_vision_members():
    catalog = _catalog()
    ir = ChatRequestIR(
        model="v",
        messages=(MessageIR(role="user", content=(ImagePart(url="https://x/i.jpg"),)),),
    )
    plan = resolve_plan(catalog, ir)
    ids = {m.identity for m in plan.members}
    assert ids == {"online"}  # novis (vision: false) dropped


def test_warns_when_web_search_and_tools_combine_on_a_search_capable_llm():
    """Gemini/Vertex AI can't combine its search-grounding tool with function tools in one call —
    litellm silently drops the search tool with only its own internal warning, invisible to mom.
    This is the paper trail: the synthesizer (`online`, search-capable) also gets `ir.tools` wired
    in ensemble `s`'s default arbitrate mode, so the conflict is on the synth params."""
    catalog = _catalog()
    ir = ChatRequestIR(
        model="s",
        messages=(MessageIR(role="user", content="latest news?"),),
        web_search=True,
        tools=(ToolSpec(name="lookup"),),
    )
    with structlog.testing.capture_logs() as logs:
        resolve_plan(catalog, ir)
    warnings = [
        log
        for log in logs
        if log["log_level"] == "warning" and "search tool" in log.get("event", "")
    ]
    assert len(warnings) == 1
    assert warnings[0]["llm"] == "online"


def test_no_conflict_warning_for_web_search_alone():
    catalog = _catalog()
    ir = ChatRequestIR(
        model="s", messages=(MessageIR(role="user", content="latest news?"),), web_search=True
    )
    with structlog.testing.capture_logs() as logs:
        resolve_plan(catalog, ir)
    assert not any("search tool" in log.get("event", "") for log in logs)


def test_no_conflict_warning_for_tools_alone():
    catalog = _catalog()
    ir = ChatRequestIR(
        model="s",
        messages=(MessageIR(role="user", content="hi"),),
        tools=(ToolSpec(name="lookup"),),
    )
    with structlog.testing.capture_logs() as logs:
        resolve_plan(catalog, ir)
    assert not any("search tool" in log.get("event", "") for log in logs)


def test_no_conflict_warning_on_a_non_search_llm():
    catalog = _catalog()
    # ensemble "s" pairs `online` (search) with `offline` (no search) — the non-search member
    # never has `web_search_options` in its params at all, so no conflict is possible for it.
    ir = ChatRequestIR(
        model="s",
        messages=(MessageIR(role="user", content="hi"),),
        web_search=True,
        tools=(ToolSpec(name="lookup"),),
    )
    with structlog.testing.capture_logs() as logs:
        resolve_plan(catalog, ir)
    offline_warnings = [
        log for log in logs if log.get("llm") == "offline" and "search tool" in log.get("event", "")
    ]
    assert offline_warnings == []


async def test_web_search_via_chat_api():
    fake = FakeLLM()
    catalog = _catalog()
    container = Container(
        settings=Settings(_env_file=None),
        catalog=catalog,
        client=fake,
        clock=ManualClock(),
        ids=SequentialIds(),
    )
    app = create_app(container=container)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "s",
                "messages": [{"role": "user", "content": "news?"}],
                "web_search": True,
            },
        )
    assert resp.status_code == 200
    online = next(s for s in fake.completions if s.llm_name == "online")
    assert online.params["web_search_options"] == {"search_context_size": "high"}
