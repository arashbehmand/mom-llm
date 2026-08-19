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
    _call_with_retries,
    _chat_messages_to_responses_input,
    _chat_tool_choice_to_responses,
    _chat_tools_to_responses,
    _responses_params,
    _retry_after_seconds,
    _tool_call_dict,
    _tool_call_fragment,
    _usage,
    unknown_anthropic_models,
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
# Responses path: Chat-shaped tools/tool_choice must be flattened for litellm.aresponses.
# --------------------------------------------------------------------------------------------


def test_chat_tools_to_responses_flattens_function_and_passes_others_through() -> None:
    chat = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
        {"type": "mcp", "server_label": "x"},  # already Responses-shaped -> untouched
    ]
    out = _chat_tools_to_responses(chat)
    assert out[0] == {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    assert "function" not in out[0]  # the nested wrapper is gone (the API 400s on it)
    assert out[1] == {"type": "mcp", "server_label": "x"}


def test_chat_tool_choice_to_responses_flattens_specific_function() -> None:
    assert _chat_tool_choice_to_responses({"type": "function", "function": {"name": "f"}}) == {
        "type": "function",
        "name": "f",
    }
    # string choices and already-flat shapes pass through
    assert _chat_tool_choice_to_responses("auto") == "auto"


def test_chat_messages_to_responses_input_retypes_content_by_role() -> None:
    items = _chat_messages_to_responses_input(
        [
            {"role": "system", "content": "be brief"},  # string stays a string
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},  # -> input_text
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},  # -> output_text
        ]
    )
    assert items[0] == {"role": "system", "content": "be brief"}
    assert items[1] == {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    assert items[2] == {"role": "assistant", "content": [{"type": "output_text", "text": "yo"}]}


def test_chat_messages_to_responses_input_maps_tool_calls_and_outputs() -> None:
    items = _chat_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"x":1}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
    )
    assert items[0] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "f",
        "arguments": '{"x":1}',
    }
    assert items[1] == {"type": "function_call_output", "call_id": "c1", "output": "42"}


def test_responses_params_flattens_tools_at_the_boundary() -> None:
    spec = _spec(
        params={
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
        }
    )
    params = _responses_params(spec)
    assert params["input"] == [{"role": "user", "content": "hi"}]  # messages -> input
    assert params["tools"][0] == {"type": "function", "name": "f", "parameters": {}}
    assert params["tool_choice"] == {"type": "function", "name": "f"}


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


# --------------------------------------------------------------------------------------------
# _call_with_retries / retry wiring: mom owns retries now — litellm's own `num_retries` is never
# used (see the adapter module's docstring for why: zero backoff for most error classes, and it
# silently discards a fully-exhausted retry's own failure in favor of re-raising the FIRST
# attempt's exception).
# --------------------------------------------------------------------------------------------


class _StatusError(Exception):
    """A bare exception carrying just a ``status_code`` — enough for ``_classify``'s status-code
    fallback path. The real ``litellm.exceptions`` submodule isn't reachable through the faked
    bare ``litellm`` module these tests install (no ``__path__`` for submodule resolution), so
    classification exercises exactly the fallback a non-litellm-wrapped failure (e.g. a raw httpx
    error from the proxy path) would hit in production too.
    """

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _retryable_error(message: str = "rate limited") -> _StatusError:
    return _StatusError(429, message)  # -> "rate_limit", IS in _RETRYABLE_KINDS


def _non_retryable_error(message: str = "bad request") -> _StatusError:
    return _StatusError(400, message)  # -> "bad_request", NOT in _RETRYABLE_KINDS


async def test_call_with_retries_retries_then_succeeds() -> None:
    calls: list[int] = []

    async def make_call() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _retryable_error()
        return "ok"

    result, attempts = await _call_with_retries(
        make_call, spec=_spec(retries=3, retry_backoff_seconds=0.001), verb="call"
    )
    assert result == "ok"
    assert attempts == 3
    assert len(calls) == 3


async def test_call_with_retries_exhausts_and_raises_the_last_error() -> None:
    calls: list[int] = []

    async def make_call() -> str:
        calls.append(1)
        raise _retryable_error(f"failure #{len(calls)}")

    with pytest.raises(UpstreamError) as excinfo:
        await _call_with_retries(
            make_call, spec=_spec(retries=2, retry_backoff_seconds=0.001), verb="call"
        )
    # 1 initial + 2 retries = 3 attempts; the raised error is the LAST one, not the first —
    # litellm's own wrapper would have silently discarded this and re-raised failure #1 instead.
    assert len(calls) == 3
    assert "failure #3" in str(excinfo.value.__cause__)
    assert excinfo.value.attempts == 3


async def test_call_with_retries_never_retries_a_non_retryable_kind() -> None:
    calls: list[int] = []

    async def make_call() -> str:
        calls.append(1)
        raise _non_retryable_error()

    with pytest.raises(UpstreamError):
        await _call_with_retries(
            make_call, spec=_spec(retries=5, retry_backoff_seconds=0.001), verb="call"
        )
    assert len(calls) == 1  # a bad_request fails identically every time — retrying is pure waste


async def test_call_with_retries_default_zero_retries_means_one_attempt() -> None:
    calls: list[int] = []

    async def make_call() -> str:
        calls.append(1)
        raise _retryable_error()

    with pytest.raises(UpstreamError):
        await _call_with_retries(make_call, spec=_spec(), verb="call")  # retries=0 by default
    assert len(calls) == 1


async def test_complete_retries_a_retryable_error_then_succeeds(
    litellm_module: ModuleType,
) -> None:
    calls: list[int] = []

    async def acompletion(**_: Any) -> NS:
        calls.append(1)
        if len(calls) < 3:
            raise _retryable_error(f"attempt {len(calls)}")
        return _response(content="ok")

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    result = await LiteLLMClient().complete(_spec(retries=3, retry_backoff_seconds=0.001))
    assert result.content == "ok"
    assert result.attempts == 3
    assert len(calls) == 3


async def test_complete_never_retries_a_non_retryable_kind(litellm_module: ModuleType) -> None:
    calls: list[int] = []

    async def acompletion(**_: Any) -> NS:
        calls.append(1)
        raise _non_retryable_error()

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    with pytest.raises(UpstreamError):
        await LiteLLMClient().complete(_spec(retries=5, retry_backoff_seconds=0.001))
    assert len(calls) == 1


async def test_stream_retries_connection_establishment_then_succeeds(
    litellm_module: ModuleType,
) -> None:
    calls: list[int] = []
    parts = [_chunk(content="hi"), _chunk(finish_reason="stop", usage=_usage_obj())]

    async def acompletion(**_: Any) -> AsyncIterator[NS]:
        calls.append(1)
        if len(calls) < 2:
            raise _retryable_error()
        return _aiter(parts)

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    chunks = await _drain(LiteLLMClient(), _spec(retries=2, retry_backoff_seconds=0.001))
    assert len(calls) == 2
    assert chunks[0].attempts == 2  # stamped on the FIRST yielded chunk only
    assert all(c.attempts is None for c in chunks[1:])


async def test_stream_never_retries_a_mid_stream_failure(litellm_module: ModuleType) -> None:
    """Once content has been yielded, a failure can't be un-sent — retrying would duplicate or
    corrupt the client-visible answer. mom's retry loop only ever covers connection
    establishment (the initial ``await litellm.acompletion(stream=True, ...)``), never the
    ``async for`` iteration that follows."""
    calls: list[int] = []

    async def acompletion(**_: Any) -> AsyncIterator[NS]:
        calls.append(1)
        return _aiter([_chunk(content="partial")], raise_at_end=_retryable_error("dropped"))

    litellm_module.acompletion = acompletion  # type: ignore[attr-defined]

    with pytest.raises(UpstreamError, match="member-a stream failed"):
        await _drain(LiteLLMClient(), _spec(retries=3, retry_backoff_seconds=0.001))
    assert len(calls) == 1  # the mid-stream failure was NOT retried, despite being a retryable kind


# --------------------------------------------------------------------------------------------
# _retry_after_seconds: a provider's Retry-After header, preferred over blind backoff.
# --------------------------------------------------------------------------------------------


def test_retry_after_seconds_reads_a_sane_header() -> None:
    exc = NS(response=NS(headers={"retry-after": "2.5"}))
    assert _retry_after_seconds(exc) == 2.5


def test_retry_after_seconds_caps_a_huge_header() -> None:
    exc = NS(response=NS(headers={"retry-after": "9999"}))
    assert _retry_after_seconds(exc, cap=30.0) == 30.0


def test_retry_after_seconds_none_when_absent() -> None:
    assert _retry_after_seconds(NS(response=None)) is None
    assert _retry_after_seconds(RuntimeError("x")) is None


def test_retry_after_seconds_none_when_malformed() -> None:
    exc = NS(response=NS(headers={"retry-after": "not-a-number"}))
    assert _retry_after_seconds(exc) is None


def test_retry_after_seconds_ignores_negative_values() -> None:
    exc = NS(response=NS(headers={"retry-after": "-5"}))
    assert _retry_after_seconds(exc) is None


# --------------------------------------------------------------------------------------------
# Stale-catalog guard. mom reads the model catalog litellm bundles, so adopting a model newer
# than the pinned litellm knows leaves Anthropic calls silently mis-sized and mis-parameterized
# (2026-08-19: claude-opus-5 on litellm 1.93 -> 4096-token cap, forwarded top_p).
# --------------------------------------------------------------------------------------------


def _install_catalog(module: ModuleType, known: set[str]) -> None:
    """Fake litellm's routing plus a catalog holding exactly ``known`` (bare, unprefixed keys)."""

    def get_llm_provider(model: str) -> tuple[str, str, None, None]:
        if "/" not in model:
            raise ValueError(f"unroutable: {model}")
        provider, _, rest = model.partition("/")
        return rest, provider, None, None

    module.get_llm_provider = get_llm_provider  # type: ignore[attr-defined]
    module.model_cost = {key: {"max_output_tokens": 128000} for key in known}  # type: ignore[attr-defined]


def test_unknown_anthropic_models_flags_a_model_the_catalog_predates(
    litellm_module: ModuleType,
) -> None:
    _install_catalog(litellm_module, known={"claude-sonnet-5"})
    flagged = unknown_anthropic_models(["anthropic/claude-sonnet-5", "anthropic/claude-opus-5"])
    assert flagged == ["anthropic/claude-opus-5"]


def test_unknown_anthropic_models_ignores_other_providers(litellm_module: ModuleType) -> None:
    """Most of the configured panel is absent from litellm's catalog and works fine — only
    Anthropic substitutes a token cap and gates sampling params on the entry being there."""
    _install_catalog(litellm_module, known=set())
    assert unknown_anthropic_models(["openrouter/moonshotai/kimi-k3:nitro", "xai/grok-4.6"]) == []


def test_unknown_anthropic_models_skips_unroutable_ids(litellm_module: ModuleType) -> None:
    _install_catalog(litellm_module, known=set())
    assert unknown_anthropic_models(["not-a-routable-id"]) == []


def test_unknown_anthropic_models_is_quiet_when_the_catalog_is_current(
    litellm_module: ModuleType,
) -> None:
    _install_catalog(litellm_module, known={"claude-opus-5"})
    assert unknown_anthropic_models(["anthropic/claude-opus-5"]) == []


def test_unknown_anthropic_models_wants_an_exact_entry_not_a_near_miss(
    litellm_module: ModuleType,
) -> None:
    """The realistic drift is a plausible new id, which litellm resolves to a *generalized*
    entry — someone else's limits, quietly. Only an exact catalog key counts as known."""
    _install_catalog(litellm_module, known={"claude-opus-5"})
    assert unknown_anthropic_models(["anthropic/claude-opus-6"]) == ["anthropic/claude-opus-6"]


def test_model_cost_map_is_pinned_before_litellm_can_be_imported() -> None:
    """The pin lives in ``mom/__init__.py`` precisely so no import order can defeat it: importing
    any mom module at all must already have set it. Asserting it here (rather than in the adapter)
    is the regression test for moving it back somewhere that only wins the race by luck."""
    import os

    import mom

    assert mom.__name__ == "mom"  # the import above is the point, not the attribute
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
