"""Hermetic unit tests for the LiteLLM adapter.

``litellm`` is a lazy, method-local import (and is *not* an installed dependency on this branch),
so we drive the adapter by injecting a fake ``litellm`` module into ``sys.modules``. Python's
import machinery consults ``sys.modules`` before any finder, so the adapter's ``import litellm``
binds our fake — no network, no real SDK, no httpx traffic. This exercises the adapter's own
normalization logic (usage/cost-less token extraction, tool-call shaping, chunk assembly, error
wrapping), which is exactly the low-coverage surface the hermetic suite was missing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mom.adapters.litellm_client import (
    LiteLLMClient,
    _call_params,
    _tool_call_dict,
    _tool_call_fragment,
    _usage,
)
from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec


# Concise alias for building provider-shaped fakes the adapter reads via getattr.
NS = SimpleNamespace


# --------------------------------------------------------------------------------------------
# Fakes: minimal provider-shaped objects the adapter reads via getattr / attribute access.
# --------------------------------------------------------------------------------------------


def _usage_obj(
    *,
    prompt: int = 0,
    completion: int = 0,
    cached: int | None = None,
    reasoning: int | None = None,
    cache_write: int = 0,
) -> NS:
    prompt_details = NS(cached_tokens=cached) if cached is not None else None
    completion_details = NS(reasoning_tokens=reasoning) if reasoning is not None else None
    return NS(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        cache_creation_input_tokens=cache_write,
    )


def _tool_call(id_: str, name: str, arguments: str) -> NS:
    return NS(id=id_, function=NS(name=name, arguments=arguments))


def _response(
    *,
    content: str | None = "hello",
    reasoning: str | None = None,
    tool_calls: list[NS] | None = None,
    finish_reason: str | None = "stop",
    usage: NS | None = None,
) -> NS:
    message = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return NS(choices=[NS(message=message, finish_reason=finish_reason)], usage=usage)


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[NS] | None = None,
    finish_reason: str | None = None,
    usage: NS | None = None,
    no_choices: bool = False,
) -> NS:
    if no_choices:
        return NS(choices=[], usage=usage)
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)], usage=usage)


class _Recorder:
    """Captures the kwargs the adapter passes to ``litellm.acompletion``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


async def _aiter(parts: list[NS], *, raise_at_end: Exception | None = None) -> AsyncIterator[NS]:
    for part in parts:
        yield part
    if raise_at_end is not None:
        raise raise_at_end


@pytest.fixture
def litellm_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Inject a bare ``litellm`` module; tests attach an ``acompletion`` fake to it."""
    module = ModuleType("litellm")
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def _install_complete(module: ModuleType, response: NS) -> _Recorder:
    recorder = _Recorder()

    async def acompletion(**kwargs: Any) -> NS:
        recorder.calls.append(kwargs)
        return response

    module.acompletion = acompletion  # type: ignore[attr-defined]
    return recorder


def _install_stream(
    module: ModuleType, parts: list[NS], *, raise_at_end: Exception | None = None
) -> _Recorder:
    recorder = _Recorder()

    async def acompletion(**kwargs: Any) -> AsyncIterator[NS]:
        recorder.calls.append(kwargs)
        return _aiter(parts, raise_at_end=raise_at_end)

    module.acompletion = acompletion  # type: ignore[attr-defined]
    return recorder


def _spec(**overrides: Any) -> CallSpec:
    base: dict[str, Any] = {
        "llm_name": "member-a",
        "model": "openai/gpt-x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return CallSpec(**base)


# --------------------------------------------------------------------------------------------
# _usage: token normalization, including provider prompt-cache and reasoning details.
# --------------------------------------------------------------------------------------------


def test_usage_none_is_all_zeros() -> None:
    usage = _usage(None)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.reasoning_tokens == 0
    assert usage.cached_prompt_tokens == 0
    assert usage.cache_write_tokens == 0


def test_usage_full_details() -> None:
    raw = _usage_obj(prompt=100, completion=40, cached=25, reasoning=12, cache_write=8)
    usage = _usage(raw)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 40
    assert usage.cached_prompt_tokens == 25
    assert usage.reasoning_tokens == 12
    assert usage.cache_write_tokens == 8


def test_usage_without_detail_blocks() -> None:
    # prompt_tokens_details / completion_tokens_details absent -> cached/reasoning collapse to 0.
    raw = _usage_obj(prompt=7, completion=3)
    usage = _usage(raw)
    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 3
    assert usage.cached_prompt_tokens == 0
    assert usage.reasoning_tokens == 0


def test_usage_coerces_none_counts_to_zero() -> None:
    raw = NS(
        prompt_tokens=None,
        completion_tokens=None,
        prompt_tokens_details=NS(cached_tokens=None),
        completion_tokens_details=NS(reasoning_tokens=None),
        cache_creation_input_tokens=None,
    )
    usage = _usage(raw)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.cached_prompt_tokens == 0
    assert usage.reasoning_tokens == 0


# --------------------------------------------------------------------------------------------
# Tool-call shaping helpers.
# --------------------------------------------------------------------------------------------


def test_tool_call_dict_full_and_missing_function() -> None:
    full = _tool_call_dict(_tool_call("call_1", "search", '{"q":"x"}'))
    assert full == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q":"x"}'},
    }
    bare = _tool_call_dict(NS(id=None, function=None))
    assert bare == {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}


def test_tool_call_fragment_full_and_missing_function() -> None:
    frag = _tool_call_fragment(NS(index=2, id="call_2", function=NS(name="do", arguments="{")))
    assert frag == {"index": 2, "id": "call_2", "name": "do", "arguments": "{"}
    bare = _tool_call_fragment(NS(index=0, id=None, function=None))
    assert bare == {"index": 0, "id": None, "name": None, "arguments": None}


# --------------------------------------------------------------------------------------------
# _call_params: param merge + optional timeout.
# --------------------------------------------------------------------------------------------


def test_call_params_merges_and_omits_timeout() -> None:
    params = _call_params(_spec(params={"temperature": 0.3}))
    assert params["model"] == "openai/gpt-x"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["temperature"] == 0.3
    assert "timeout" not in params


def test_call_params_includes_timeout() -> None:
    params = _call_params(_spec(timeout_seconds=12.5))
    assert params["timeout"] == 12.5


# --------------------------------------------------------------------------------------------
# complete(): non-streaming path.
# --------------------------------------------------------------------------------------------


async def test_complete_maps_content_reasoning_usage_and_tool_calls(
    litellm_module: ModuleType,
) -> None:
    response = _response(
        content="the answer",
        reasoning="because",
        tool_calls=[_tool_call("call_9", "lookup", "{}")],
        finish_reason="tool_calls",
        usage=_usage_obj(prompt=50, completion=10, cached=5, reasoning=4),
    )
    recorder = _install_complete(litellm_module, response)

    result = await LiteLLMClient().complete(
        _spec(params={"temperature": 0.1}, timeout_seconds=30.0)
    )

    assert result.content == "the answer"
    assert result.reasoning == "because"
    assert result.finish_reason == "tool_calls"
    assert result.usage.prompt_tokens == 50
    assert result.usage.cached_prompt_tokens == 5
    assert result.usage.reasoning_tokens == 4
    assert result.tool_calls == (
        {"id": "call_9", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
    )
    # The transport forwarded a non-streaming call with the merged params.
    (call,) = recorder.calls
    assert call["stream"] is False
    assert call["model"] == "openai/gpt-x"
    assert call["temperature"] == 0.1
    assert call["timeout"] == 30.0


async def test_complete_defaults_content_finish_and_tool_calls(
    litellm_module: ModuleType,
) -> None:
    # content None -> "", finish_reason None -> "stop", tool_calls None -> (), usage None -> zeros.
    response = _response(content=None, finish_reason=None, tool_calls=None, usage=None)
    _install_complete(litellm_module, response)

    result = await LiteLLMClient().complete(_spec())

    assert result.content == ""
    assert result.reasoning is None
    assert result.finish_reason == "stop"
    assert result.tool_calls == ()
    assert result.usage.total_tokens == 0


async def test_complete_wraps_provider_errors(litellm_module: ModuleType) -> None:
    boom = RuntimeError("upstream 500")

    async def acompletion(**_: Any) -> NS:
        raise boom

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    with pytest.raises(UpstreamError, match="member-a call failed") as excinfo:
        await LiteLLMClient().complete(_spec())
    assert excinfo.value.__cause__ is boom


# --------------------------------------------------------------------------------------------
# stream(): streaming path.
# --------------------------------------------------------------------------------------------


async def _drain(client: LiteLLMClient, spec: CallSpec) -> list[Any]:
    return [chunk async for chunk in client.stream(spec)]


async def test_stream_yields_content_reasoning_tool_calls_and_usage(
    litellm_module: ModuleType,
) -> None:
    parts = [
        _chunk(content=None),  # opening role-only delta -> nothing emitted
        _chunk(tool_calls=[NS(index=0, id="call_1", function=NS(name="f", arguments=None))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='{"a":1}'))]),
        _chunk(content="Hel"),
        _chunk(reasoning="think"),
        _chunk(content="lo"),
        _chunk(finish_reason="stop", usage=_usage_obj(prompt=12, completion=6)),
    ]
    recorder = _install_stream(litellm_module, parts)

    chunks = await _drain(LiteLLMClient(), _spec())

    # First real emission is the tool-call name fragment (opening delta produced nothing).
    assert chunks[0].tool_call == {"index": 0, "id": "call_1", "name": "f", "arguments": None}
    assert chunks[1].tool_call == {"index": 0, "id": None, "name": None, "arguments": '{"a":1}'}
    contents = [c.content for c in chunks if c.content is not None]
    assert contents == ["Hel", "lo"]
    assert any(c.reasoning == "think" for c in chunks)
    terminal = chunks[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.prompt_tokens == 12
    # Streaming requested usage so cost/token accounting is possible downstream.
    (call,) = recorder.calls
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


async def test_stream_emits_usage_only_chunk_when_choices_empty(
    litellm_module: ModuleType,
) -> None:
    # Some providers send a trailing usage frame with an empty ``choices`` array.
    parts = [
        _chunk(content="hi"),
        _chunk(no_choices=True, usage=_usage_obj(prompt=4, completion=2)),
    ]
    _install_stream(litellm_module, parts)

    chunks = await _drain(LiteLLMClient(), _spec())

    assert chunks[0].content == "hi"
    usage_chunk = chunks[-1]
    assert usage_chunk.content is None
    assert usage_chunk.finish_reason is None
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.completion_tokens == 2


async def test_stream_wraps_errors_raised_before_iteration(
    litellm_module: ModuleType,
) -> None:
    async def acompletion(**_: Any) -> AsyncIterator[NS]:
        raise RuntimeError("connect failed")

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    with pytest.raises(UpstreamError, match="member-a stream failed"):
        await _drain(LiteLLMClient(), _spec())


async def test_stream_wraps_errors_raised_mid_iteration(
    litellm_module: ModuleType,
) -> None:
    parts = [_chunk(content="partial")]
    _install_stream(litellm_module, parts, raise_at_end=RuntimeError("dropped"))

    with pytest.raises(UpstreamError, match="member-a stream failed"):
        await _drain(LiteLLMClient(), _spec())
