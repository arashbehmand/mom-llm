"""LiteLLM adapter — the ONLY module that imports ``litellm``.

Normalizes provider responses/chunks into the domain's ``Completion`` / ``CompletionChunk`` and
usage into ``Usage`` (including provider prompt-cache tokens). This first version covers the chat
route; the Responses route and the provider-specific streaming quirks are layered on later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
from typing import Any
from urllib.parse import urlparse

from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.results import Usage


# Set before litellm is ever imported (lazily, below): use the bundled model-cost map instead of
# fetching it over the network at import time (which is slow, and pytest-socket would block it).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _first_int(*values: Any) -> int:
    """First truthy value coerced to int (each may be None/absent); 0 if none are present."""
    for value in values:
        coerced = int(value or 0)
        if coerced:
            return coerced
    return 0


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)
    reasoning = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0) if completion_details else 0
    )
    # Providers report cache-read/write prompt tokens under different field names; take the first
    # present non-zero spelling for each so cache-aware cost isn't undercounted.
    cached = _first_int(
        getattr(prompt_details, "cached_tokens", None) if prompt_details else None,
        getattr(raw, "cache_read_input_tokens", None),  # Anthropic-native, top-level
        getattr(raw, "prompt_cache_hit_tokens", None),  # DeepSeek, top-level
        getattr(raw, "cacheReadInputTokens", None),  # Bedrock camelCase, top-level
    )
    cache_write = _first_int(
        getattr(raw, "cache_creation_input_tokens", None),  # Anthropic, top-level
        getattr(prompt_details, "cache_creation_tokens", None) if prompt_details else None,
        getattr(raw, "cacheWriteInputTokens", None),  # Bedrock camelCase, top-level
        getattr(raw, "cacheCreationInputTokens", None),  # Bedrock camelCase, top-level
    )
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cached_prompt_tokens=cached,
        cache_write_tokens=cache_write,
    )


def _provider_cost(container: Any, usage_raw: Any) -> float | None:
    """The real cost the provider returned (OpenRouter usage accounting / litellm response_cost)."""
    hidden = getattr(container, "_hidden_params", None) or {}
    for candidate in (hidden.get("response_cost"), getattr(usage_raw, "cost", None)):
        if isinstance(candidate, int | float) and candidate:
            return float(candidate)
    return None


def _response_cost(response: Any, model: str, usage: Usage) -> float | None:
    """Real per-call cost: provider-returned cost first (OpenRouter), then litellm's cost map."""
    import litellm

    provider = _provider_cost(response, getattr(response, "usage", None))
    if provider is not None:
        return provider
    try:
        computed = litellm.completion_cost(completion_response=response)
    except Exception:
        computed = None
    if computed:
        return float(computed)
    return _cost_from_usage(model, usage)


def _cost_from_usage(model: str, usage: Usage) -> float | None:
    """Cost for a streamed call from its final usage, via litellm's cost-per-token map."""
    import litellm

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
    except Exception:
        return None
    total = float(prompt_cost or 0.0) + float(completion_cost or 0.0)
    return total or None


def _responses_usage(raw: Any) -> Usage:
    """Normalize the Responses-API usage shape (input_tokens/output_tokens) into ``Usage``."""
    if raw is None:
        return Usage()
    in_details = getattr(raw, "input_tokens_details", None)
    out_details = getattr(raw, "output_tokens_details", None)
    reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0) if out_details else 0
    cached = int(getattr(in_details, "cached_tokens", 0) or 0) if in_details else 0
    cache_write = int(getattr(in_details, "cache_write_tokens", 0) or 0) if in_details else 0
    return Usage(
        prompt_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        completion_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        reasoning_tokens=reasoning,
        cached_prompt_tokens=cached,
        cache_write_tokens=cache_write,
    )


def _chat_tools_to_responses(tools: Any) -> Any:
    """Reshape Chat-format tools to Responses-format for ``litellm.aresponses``.

    Chat wants ``{"type":"function","function":{"name",...}}``; the Responses API wants the name
    and schema flattened to the top level (``{"type":"function","name",...}``) — otherwise it
    400s with ``Missing required parameter: 'tools[0].name'``. Non-function blocks (e.g. relayed
    ``type: mcp``) are already Responses-shaped and pass through untouched.
    """
    if not isinstance(tools, list):
        return tools
    out: list[Any] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(fn, dict):
            out.append({"type": "function", **fn})
        else:
            out.append(tool)
    return out


def _chat_tool_choice_to_responses(choice: Any) -> Any:
    """Flatten a specific-function ``tool_choice`` to Responses shape (name at the top level)."""
    fn = choice.get("function") if isinstance(choice, dict) else None
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(fn, dict):
        return {"type": "function", "name": fn.get("name")}
    return choice


def _responses_content_part(part: dict[str, Any], *, text_type: str) -> dict[str, Any] | None:
    """Reshape one Chat content part to a Responses ``input`` part (or drop it)."""
    kind = part.get("type")
    if kind in ("text", "input_text", "output_text"):
        return {"type": text_type, "text": part.get("text", "")}
    if kind in ("image_url", "input_image"):
        url = part.get("image_url") or part.get("url") or ""
        if isinstance(url, dict):
            url = url.get("url", "")
        return {"type": "input_image", "image_url": url}
    return None


def _chat_messages_to_responses_input(messages: Any) -> Any:
    """Reshape Chat-format messages into Responses ``input`` items.

    The Responses API rejects Chat's content-part types: an input role needs ``input_text`` /
    ``input_image`` and an assistant reply needs ``output_text`` (Chat uses ``text`` for both).
    Assistant ``tool_calls`` become ``function_call`` items and ``role: tool`` messages become
    ``function_call_output`` items. String content is passed straight through (already valid).
    """
    if not isinstance(messages, list):
        return messages
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": message.get("content") or "",
                }
            )
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            items.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_type = "output_text" if role == "assistant" else "input_text"
            parts = [
                p
                for raw in content
                if isinstance(raw, dict)
                and (p := _responses_content_part(raw, text_type=text_type)) is not None
            ]
            if parts:
                items.append({"role": role, "content": parts})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.get("id"),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "",
                }
            )
    return items


def _responses_params(spec: CallSpec) -> dict[str, Any]:
    """Params for ``litellm.aresponses`` — the Responses API takes ``input``, not ``messages``."""
    params = dict(spec.params)
    params["model"] = spec.model
    params["input"] = _chat_messages_to_responses_input(spec.messages)
    if "tools" in params:
        params["tools"] = _chat_tools_to_responses(params["tools"])
    if "tool_choice" in params:
        params["tool_choice"] = _chat_tool_choice_to_responses(params["tool_choice"])
    if spec.timeout_seconds is not None:
        params["timeout"] = spec.timeout_seconds
    if spec.retries:
        params["num_retries"] = spec.retries
    api_key = _resolve_api_key(spec.key_env_candidates)
    if api_key is not None:
        params["api_key"] = api_key
    _normalize_reasoning_effort(params, spec.model)
    params.setdefault("drop_params", True)
    return params


def _responses_to_completion(response: Any, model: str) -> Completion:
    """Normalize a ``ResponsesAPIResponse`` into the domain ``Completion``."""
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            for summary in getattr(item, "summary", None) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_parts.append(text)
        elif kind == "function_call":
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", "") or "",
                        "arguments": getattr(item, "arguments", "") or "",
                    },
                }
            )
    usage = _responses_usage(getattr(response, "usage", None))
    return Completion(
        content=getattr(response, "output_text", None) or "",
        reasoning="".join(reasoning_parts) or None,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage,
        tool_calls=tuple(tool_calls),
        cost_usd=_provider_cost(response, getattr(response, "usage", None))
        or _cost_from_usage(model, usage),
    )


def _responses_event_to_chunk(event: Any, model: str) -> CompletionChunk | None:
    """Map one ``aresponses`` stream event to a ``CompletionChunk`` (or None to skip)."""
    etype = getattr(event, "type", "")
    etype = getattr(etype, "value", etype)  # unwrap a str-enum to its value
    if etype == "response.output_text.delta":
        return CompletionChunk(content=getattr(event, "delta", None))
    if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        return CompletionChunk(reasoning=getattr(event, "delta", None))
    if etype == "response.output_item.added":
        item = getattr(event, "item", None)
        if item is not None and getattr(item, "type", None) == "function_call":
            return CompletionChunk(
                tool_call={
                    "index": getattr(event, "output_index", 0) or 0,
                    "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                    "name": getattr(item, "name", None),
                    "arguments": None,
                }
            )
        return None
    if etype == "response.function_call_arguments.delta":
        return CompletionChunk(
            tool_call={
                "index": getattr(event, "output_index", 0) or 0,
                "arguments": getattr(event, "delta", None),
            }
        )
    if etype == "response.completed":
        resp = getattr(event, "response", None)
        usage = _responses_usage(getattr(resp, "usage", None))
        has_tools = any(
            getattr(item, "type", None) == "function_call"
            for item in (getattr(resp, "output", None) or [])
        )
        cost = _provider_cost(resp, getattr(resp, "usage", None)) or _cost_from_usage(model, usage)
        return CompletionChunk(
            finish_reason="tool_calls" if has_tools else "stop", usage=usage, cost_usd=cost
        )
    return None


def _tool_call_dict(tc: Any) -> dict[str, Any]:
    """A complete (non-streaming) tool call -> OpenAI wire dict."""
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(fn, "name", "") or "",
            "arguments": getattr(fn, "arguments", "") or "",
        },
    }


def _tool_call_fragment(tc: Any) -> dict[str, Any]:
    """A streamed tool-call delta -> a normalized fragment for the pipeline."""
    fn = getattr(tc, "function", None)
    return {
        "index": getattr(tc, "index", 0) or 0,
        "id": getattr(tc, "id", None),
        "name": getattr(fn, "name", None) if fn else None,
        "arguments": getattr(fn, "arguments", None) if fn else None,
    }


def _resolve_api_key(candidates: tuple[str, ...]) -> str | None:
    """The first set env var among the ordered candidates (matches config resolution order)."""
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value
    return None


# Our extended effort ladder clamped to values providers actually accept. litellm maps
# `reasoning_effort` onto each provider's native reasoning param, but the value must be valid:
# OpenAI accepts minimal/low/medium/high, most others cap at low/medium/high, and none accept our
# xhigh/max tiers. Together with drop_params (for models with no reasoning support at all), this
# replaces the raw passthrough that made Gemini/Anthropic 400 on an unsupported effort value.
_EFFORT_CLAMP: dict[str, str | None] = {
    "none": None,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _normalize_reasoning_effort(params: dict[str, Any], model: str) -> None:
    """Clamp a resolved ``reasoning_effort`` to what the model's provider accepts (or drop it)."""
    value = params.get("reasoning_effort")
    if not isinstance(value, str):
        return
    clamped = _EFFORT_CLAMP.get(value.lower().strip())
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if clamped is None:
        params.pop("reasoning_effort", None)
    elif provider in {"openai", "azure"}:
        params["reasoning_effort"] = clamped
    else:
        params["reasoning_effort"] = "low" if clamped == "minimal" else clamped


def _call_params(spec: CallSpec) -> dict[str, Any]:
    params = dict(spec.params)
    params["model"] = spec.model
    params["messages"] = spec.messages
    if spec.timeout_seconds is not None:
        params["timeout"] = spec.timeout_seconds
    if spec.retries:
        params["num_retries"] = spec.retries
    api_key = _resolve_api_key(spec.key_env_candidates)
    if api_key is not None:
        params["api_key"] = api_key
    _normalize_reasoning_effort(params, spec.model)
    # Drop provider-unsupported params (e.g. reasoning on a non-reasoning model) instead of 500.
    params.setdefault("drop_params", True)
    if spec.model.startswith("openrouter/"):
        # Ask OpenRouter to return real usage-based cost (it prices models litellm's map lacks).
        extra_body = dict(params.get("extra_body") or {})
        usage_acct = dict(extra_body.get("usage") or {})
        usage_acct["include"] = True
        extra_body["usage"] = usage_acct
        params["extra_body"] = extra_body
    return params


def _proxy_client(proxy_url_env: str | None, timeout: float | None, llm_name: str) -> Any:
    """Build a request-scoped litellm HTTP client that routes this model through its proxy.

    The guarantee ported verbatim from v1: a proxied model NEVER silently falls back to a direct
    connection. If the proxy is configured but its env var is unset or malformed we raise (so the
    call fails) rather than connect directly, and ``trust_env=False`` ignores ambient proxy env.
    Returns ``None`` when the model has no proxy configured.
    """
    if not proxy_url_env:
        return None
    proxy_url = os.getenv(proxy_url_env)
    if not proxy_url:
        raise UpstreamError(f"proxy env var {proxy_url_env!r} is required for {llm_name} but unset")
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UpstreamError(f"proxy env var {proxy_url_env!r} must be a valid http(s) URL")

    import httpx
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    class _ProxyHandler(AsyncHTTPHandler):
        """A request-scoped client that connects only through the proxy (no direct fallback)."""

        def create_client(
            self,
            timeout: Any,
            event_hooks: Any,
            ssl_verify: Any = None,
            shared_session: Any = None,
        ) -> Any:
            del ssl_verify, shared_session  # signature parity with the base; unused here
            return httpx.AsyncClient(
                proxy=proxy_url,
                timeout=timeout,
                event_hooks=event_hooks,
                follow_redirects=True,
                trust_env=False,
            )

    return _ProxyHandler(timeout=int(timeout) if timeout else 600, client_alias="mom-proxied-llm")


def _fallback_token_estimate(messages: list[dict[str, Any]]) -> int:
    """A provider-naive ~chars/4 count, used when litellm's tokenizer is unavailable."""
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(
                len(part["text"])
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    return max(1, chars // 4)


class LiteLLMTokenEstimator:
    """Model-aware input-token count via ``litellm.token_counter`` (chars/4 fallback).

    Defensive by contract: a missing tokenizer, an unknown model, or a blocked network download
    must never break a request, so every failure path falls back to the cheap heuristic.
    """

    def count(self, *, model: str, messages: list[dict[str, Any]]) -> int:
        import litellm

        try:
            return int(litellm.token_counter(model=model, messages=messages))
        except Exception:
            return _fallback_token_estimate(messages)


class LiteLLMClient:
    """A single-model transport backed by ``litellm.acompletion``."""

    async def complete(self, spec: CallSpec) -> Completion:
        if spec.api == "responses":
            return await self._complete_via_responses(spec)
        import litellm

        params = _call_params(spec)
        proxy = _proxy_client(spec.proxy_url_env, spec.timeout_seconds, spec.llm_name)
        if proxy is not None:
            params["client"] = proxy
        try:
            response = await litellm.acompletion(stream=False, **params)
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} call failed") from exc

        choice = response.choices[0]
        message = choice.message
        tool_calls = tuple(
            _tool_call_dict(tc) for tc in (getattr(message, "tool_calls", None) or [])
        )
        usage = _usage(getattr(response, "usage", None))
        return Completion(
            content=message.content or "",
            reasoning=getattr(message, "reasoning_content", None),
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            tool_calls=tool_calls,
            cost_usd=_response_cost(response, spec.model, usage),
        )

    async def _complete_via_responses(self, spec: CallSpec) -> Completion:
        """Non-streaming call over the provider's Responses API (for ``api: responses`` models)."""
        import litellm

        params = _responses_params(spec)
        proxy = _proxy_client(spec.proxy_url_env, spec.timeout_seconds, spec.llm_name)
        if proxy is not None:
            params["client"] = proxy
        try:
            response = await litellm.aresponses(**params)
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} call failed") from exc
        return _responses_to_completion(response, spec.model)

    async def _stream_via_responses(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        """Streaming call over the provider's Responses API (for an ``api: responses`` synth)."""
        import litellm

        params = _responses_params(spec)
        proxy = _proxy_client(spec.proxy_url_env, spec.timeout_seconds, spec.llm_name)
        if proxy is not None:
            params["client"] = proxy
        try:
            stream = await litellm.aresponses(stream=True, **params)
            async for event in stream:
                chunk = _responses_event_to_chunk(event, spec.model)
                if chunk is not None:
                    yield chunk
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} stream failed") from exc

    async def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        if spec.api == "responses":
            async for chunk in self._stream_via_responses(spec):
                yield chunk
            return
        import litellm

        params = _call_params(spec)
        params["stream_options"] = {"include_usage": True}
        proxy = _proxy_client(spec.proxy_url_env, spec.timeout_seconds, spec.llm_name)
        if proxy is not None:
            params["client"] = proxy
        try:
            stream = await litellm.acompletion(stream=True, **params)
            async for part in stream:
                choices = getattr(part, "choices", None) or []
                delta = choices[0].delta if choices else None
                finish = choices[0].finish_reason if choices else None
                usage_raw = getattr(part, "usage", None)
                for raw_tool in (getattr(delta, "tool_calls", None) or []) if delta else []:
                    yield CompletionChunk(tool_call=_tool_call_fragment(raw_tool))
                content = getattr(delta, "content", None) if delta else None
                reasoning = getattr(delta, "reasoning_content", None) if delta else None
                if content is not None or reasoning is not None or finish or usage_raw:
                    chunk_usage = _usage(usage_raw) if usage_raw else None
                    chunk_cost: float | None = None
                    if chunk_usage is not None:
                        chunk_cost = _provider_cost(part, usage_raw) or _cost_from_usage(
                            spec.model, chunk_usage
                        )
                    yield CompletionChunk(
                        content=content,
                        reasoning=reasoning,
                        finish_reason=finish,
                        usage=chunk_usage,
                        cost_usd=chunk_cost,
                    )
        except Exception as exc:
            raise UpstreamError(f"{spec.llm_name} stream failed") from exc
