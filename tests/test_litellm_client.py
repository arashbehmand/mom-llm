"""Adapter unit tests: param assembly, api-key resolution, and the no-fallback proxy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mom.adapters.litellm_client import (
    LiteLLMTokenEstimator,
    _call_params,
    _classify,
    _fallback_token_estimate,
    _proxy_client,
    _resolve_api_key,
    _responses_event_to_chunk,
    _responses_to_completion,
    _scrub,
    _upstream_error,
    _usage,
)
from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec
from mom.domain.results import Usage


def _spec(*, model: str = "openai/x", **kw: object) -> CallSpec:
    return CallSpec(
        llm_name="x",
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        **kw,  # type: ignore[arg-type]
    )


def test_call_params_forwards_api_key_and_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_KEY", "sk-abc")
    params = _call_params(_spec(key_env_candidates=("MY_KEY",), retries=3, timeout_seconds=12.0))
    assert params["api_key"] == "sk-abc"
    assert params["timeout"] == 12.0
    assert params["model"] == "openai/x"


def test_api_key_first_candidate_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert _resolve_api_key(("GEMINI_API_KEY", "GOOGLE_API_KEY")) == "g-key"


def test_no_api_key_when_none_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    assert "api_key" not in _call_params(_spec(key_env_candidates=("ABSENT_KEY",)))


def test_num_retries_is_always_zero_mom_owns_retries_now():
    """litellm's own `num_retries` retry wrapper is never used (see the module docstring on why:
    zero backoff for most error classes, and it silently discards the last failure in favor of
    re-raising the first). `spec.retries` still drives mom's own `_call_with_retries` loop
    instead."""
    assert _call_params(_spec())["num_retries"] == 0
    assert _call_params(_spec(retries=3))["num_retries"] == 0


def test_proxy_none_when_unconfigured():
    assert _proxy_client(None, 60.0, "x") is None


def test_proxy_missing_env_raises_no_direct_fallback(monkeypatch: pytest.MonkeyPatch):
    # The env var is declared for the model but unset: fail rather than connect directly.
    monkeypatch.delenv("MY_PROXY", raising=False)
    with pytest.raises(UpstreamError):
        _proxy_client("MY_PROXY", 60.0, "x")


def test_proxy_malformed_url_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_PROXY", "not-a-url")
    with pytest.raises(UpstreamError):
        _proxy_client("MY_PROXY", 60.0, "x")


def test_proxy_builds_handler(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_PROXY", "http://proxy.local:8080")
    assert _proxy_client("MY_PROXY", 60.0, "x") is not None


def test_reasoning_effort_xhigh_clamped_for_gemini():
    # Gemini rejects xhigh; it must be clamped to a value the provider accepts.
    spec = _spec(model="gemini/gemini-2.5-pro", params={"reasoning_effort": "xhigh"})
    assert _call_params(spec)["reasoning_effort"] == "high"


def test_reasoning_effort_minimal_downgraded_off_openai():
    params = _call_params(_spec(model="anthropic/claude", params={"reasoning_effort": "minimal"}))
    assert params["reasoning_effort"] == "low"  # anthropic has no 'minimal'


def test_reasoning_effort_minimal_kept_for_openai():
    params = _call_params(_spec(model="openai/gpt-5", params={"reasoning_effort": "minimal"}))
    assert params["reasoning_effort"] == "minimal"


def test_reasoning_effort_max_clamped_to_high():
    params = _call_params(_spec(model="openai/gpt-5", params={"reasoning_effort": "max"}))
    assert params["reasoning_effort"] == "high"


def test_drop_params_enabled_by_default():
    assert _call_params(_spec())["drop_params"] is True


# --- _usage: provider cache-token spellings -------------------------------------------------


def test_usage_none_returns_empty():
    assert _usage(None) == Usage()


def test_usage_basic_prompt_and_completion():
    u = _usage(SimpleNamespace(prompt_tokens=100, completion_tokens=10))
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 10
    assert u.cached_prompt_tokens == 0
    assert u.cache_write_tokens == 0


def test_usage_no_cache_fields_yields_zero():
    # A response object carrying no cache fields at all must record 0/0, not error.
    u = _usage(SimpleNamespace())
    assert u.cached_prompt_tokens == 0
    assert u.cache_write_tokens == 0
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0


def test_usage_openai_style_cached_tokens():
    # OpenAI / Gemini / xAI / Groq: cache reads under prompt_tokens_details.cached_tokens.
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    assert _usage(raw).cached_prompt_tokens == 40


def test_usage_reasoning_tokens_from_completion_details():
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
    )
    assert _usage(raw).reasoning_tokens == 30


def test_usage_anthropic_cache_read_and_write():
    # Anthropic-native reports both cache dimensions as top-level fields.
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        cache_read_input_tokens=40,
        cache_creation_input_tokens=25,
    )
    u = _usage(raw)
    assert u.cached_prompt_tokens == 40
    assert u.cache_write_tokens == 25


def test_usage_deepseek_prompt_cache_hit_tokens():
    raw = SimpleNamespace(prompt_tokens=100, completion_tokens=10, prompt_cache_hit_tokens=64)
    assert _usage(raw).cached_prompt_tokens == 64


def test_usage_bedrock_camelcase_read_and_write():
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        cacheReadInputTokens=40,
        cacheWriteInputTokens=25,
    )
    u = _usage(raw)
    assert u.cached_prompt_tokens == 40
    assert u.cache_write_tokens == 25


def test_usage_bedrock_cache_creation_camelcase():
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        cacheCreationInputTokens=25,
    )
    assert _usage(raw).cache_write_tokens == 25


def test_usage_cache_write_from_prompt_details():
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cache_creation_tokens=15),
    )
    assert _usage(raw).cache_write_tokens == 15


def test_usage_first_nonzero_spelling_wins():
    # A zero in the first spelling must not shadow a real value in a later one.
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        cache_read_input_tokens=40,
    )
    assert _usage(raw).cached_prompt_tokens == 40


def test_usage_none_valued_cache_fields_yield_zero():
    # litellm may set the attributes to None rather than omit them.
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=None,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    u = _usage(raw)
    assert u.cached_prompt_tokens == 0
    assert u.cache_write_tokens == 0


# --- Responses API normalization (api: responses members) --------------------------------------


def test_responses_to_completion_message():
    resp = SimpleNamespace(
        output_text="hello there",
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(text="hello there")])],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        _hidden_params={"response_cost": 0.002},
    )
    c = _responses_to_completion(resp, "openai/gpt-5.6-sol")
    assert c.content == "hello there"
    assert c.finish_reason == "stop"
    assert c.usage.prompt_tokens == 10
    assert c.usage.completion_tokens == 5
    assert c.usage.reasoning_tokens == 3
    assert c.usage.cached_prompt_tokens == 2
    assert c.cost_usd == 0.002


def test_responses_to_completion_tool_call():
    resp = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="c1",
                name="get_weather",
                arguments='{"city":"Paris"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        _hidden_params={"response_cost": 0.001},
    )
    c = _responses_to_completion(resp, "openai/gpt-5.6-sol")
    assert c.finish_reason == "tool_calls"
    assert c.tool_calls[0]["function"]["name"] == "get_weather"
    assert c.tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'
    assert c.tool_calls[0]["id"] == "c1"
    assert c.cost_usd == 0.001


def test_responses_event_text_delta():
    ev = SimpleNamespace(type="response.output_text.delta", delta="hi")
    assert _responses_event_to_chunk(ev, "openai/gpt-5.6-sol").content == "hi"


def test_responses_event_tool_call_added_then_args():
    added = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="function_call", call_id="c1", name="get_weather"),
    )
    started = _responses_event_to_chunk(added, "openai/gpt-5.6-sol")
    assert started.tool_call["id"] == "c1"
    assert started.tool_call["name"] == "get_weather"
    arg = SimpleNamespace(
        type="response.function_call_arguments.delta", output_index=0, delta='{"city":"Paris"}'
    )
    assert _responses_event_to_chunk(arg, "m").tool_call["arguments"] == '{"city":"Paris"}'


def test_responses_event_completed_usage_and_cost():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
        output=[SimpleNamespace(type="message")],
        _hidden_params={"response_cost": 0.003},
    )
    ev = SimpleNamespace(type="response.completed", response=resp)
    chunk = _responses_event_to_chunk(ev, "m")
    assert chunk.finish_reason == "stop"
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.reasoning_tokens == 2
    assert chunk.cost_usd == 0.003


def test_responses_event_lifecycle_ignored():
    assert _responses_event_to_chunk(SimpleNamespace(type="response.created"), "m") is None


def test_fallback_token_estimate_counts_chars_over_four():
    messages = [
        {"role": "user", "content": "x" * 40},
        {"role": "user", "content": [{"type": "text", "text": "yyyy"}, {"type": "image_url"}]},
    ]
    assert _fallback_token_estimate(messages) == (40 + 4) // 4  # 11


def test_token_estimator_returns_positive_int_and_never_raises():
    estimator = LiteLLMTokenEstimator()
    count = estimator.count(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hello"}])
    assert isinstance(count, int)
    assert count >= 1


def test_token_estimator_falls_back_when_tokenizer_unavailable(monkeypatch: pytest.MonkeyPatch):
    import litellm

    def _boom(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("no tokenizer / offline")

    monkeypatch.setattr(litellm, "token_counter", _boom)
    estimator = LiteLLMTokenEstimator()
    # Falls back to the chars/4 heuristic instead of propagating the tokenizer failure.
    count = estimator.count(model="gpt-4", messages=[{"role": "user", "content": "x" * 40}])
    assert count == 10


# ------------------------------------------------------------------------------------------------
# _classify / _scrub / _upstream_error — Phase 1 of the v2 remediation: turn opaque provider
# failures into a mom-owned, client-safe ErrorKind plus a scrubbed operator-facing detail.
# ------------------------------------------------------------------------------------------------


def test_classify_litellm_exception_types():
    import httpx
    import litellm.exceptions as le

    response = httpx.Response(403, request=httpx.Request("GET", "http://x"))
    cases = [
        (le.Timeout(message="t", model="m", llm_provider="openai"), "timeout"),
        (le.RateLimitError(message="r", llm_provider="openai", model="m"), "rate_limit"),
        (le.AuthenticationError(message="a", llm_provider="openai", model="m"), "auth"),
        (
            le.PermissionDeniedError(
                message="p", llm_provider="openai", model="m", response=response
            ),
            "auth",
        ),
        (
            le.ContextWindowExceededError(message="c", model="m", llm_provider="openai"),
            "context_length",
        ),
        (
            le.ContentPolicyViolationError(message="c", model="m", llm_provider="openai"),
            "content_filter",
        ),
        (le.APIConnectionError(message="c", llm_provider="openai", model="m"), "connection"),
        (le.InternalServerError(message="i", model="m", llm_provider="openai"), "server_error"),
        (
            le.ServiceUnavailableError(message="s", model="m", llm_provider="openai"),
            "server_error",
        ),
        (le.BadRequestError(message="b", model="m", llm_provider="openai"), "bad_request"),
    ]
    for exc, expected in cases:
        assert _classify(exc) == expected, exc.__class__.__name__


def test_classify_falls_back_to_status_code_when_not_a_litellm_exception():
    class _RawHttpError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("boom")
            self.status_code = status_code

    assert _classify(_RawHttpError(401)) == "auth"
    assert _classify(_RawHttpError(429)) == "rate_limit"
    assert _classify(_RawHttpError(408)) == "timeout"
    assert _classify(_RawHttpError(400)) == "bad_request"
    assert _classify(_RawHttpError(503)) == "server_error"


class _ConnectionResetLikeError(Exception):
    pass


class _TimeoutLikeError(Exception):
    pass


def test_classify_falls_back_to_class_name_when_no_status_code():
    assert _classify(_ConnectionResetLikeError("x")) == "connection"
    assert _classify(_TimeoutLikeError("x")) == "timeout"


def test_classify_unknown_defaults_to_unknown():
    assert _classify(ValueError("some bug")) == "unknown"


def test_scrub_redacts_openai_style_secret_key():
    assert "sk-abc123def456" not in _scrub("failed with key=sk-abc123def456")


def test_scrub_redacts_google_style_secret_key():
    assert "AIzaSyD1234567890" not in _scrub("key AIzaSyD1234567890 rejected")


def test_scrub_redacts_bearer_header():
    scrubbed = _scrub("Authorization: Bearer sk-live-abcdefghijklmnop")
    assert "abcdefghijklmnop" not in scrubbed


def test_scrub_strips_query_string_but_keeps_the_url():
    scrubbed = _scrub("GET https://api.example.com/v1/x?api_key=abc123&foo=bar failed")
    assert "https://api.example.com/v1/x" in scrubbed
    assert "abc123" not in scrubbed


def test_scrub_collapses_long_absolute_paths():
    scrubbed = _scrub("open failed: /home/deploy/secrets/prod/credentials.json")
    assert "credentials.json" not in scrubbed


def test_scrub_truncates():
    assert len(_scrub("x" * 1000, max_chars=50)) == 50


def test_upstream_error_safe_message_never_contains_the_raw_cause():
    spec = _spec()
    exc = ValueError("SECRET provider detail: key=sk-abc123def456")
    err = _upstream_error(spec, exc, "call")
    assert err.safe_message == "x call failed"
    assert "SECRET" not in err.safe_message
    assert err.kind == "unknown"  # ValueError isn't classified as anything specific
    assert "sk-abc123def456" not in (err.detail or "")


def test_upstream_error_classifies_and_scrubs_the_detail():
    import litellm.exceptions as le

    spec = _spec()
    exc = le.RateLimitError(
        message="rate limited, key=sk-abc123def456", llm_provider="x", model="m"
    )
    err = _upstream_error(spec, exc, "stream")
    assert err.safe_message == "x stream failed"
    assert err.kind == "rate_limit"
    assert err.detail is not None
    assert "sk-abc123def456" not in err.detail
    assert "rate limited" in err.detail  # non-secret context is preserved, not wiped wholesale
