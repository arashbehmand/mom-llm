"""Translate wire requests into the canonical IR."""

from __future__ import annotations

from typing import Any

from mom.api.schemas.openai_chat import ChatCompletionRequest, ChatMessageIn
from mom.domain.errors import InvalidRequestError
from mom.domain.request import (
    ChatRequestIR,
    ContentPart,
    ImagePart,
    MessageIR,
    Sampling,
    SpecificTool,
    TextPart,
    ToolCallIR,
    ToolChoice,
    ToolSpec,
)


_ROLES: frozenset[str] = frozenset({"system", "developer", "user", "assistant", "tool"})


def _content(raw: str | list[dict[str, Any]] | None) -> str | tuple[ContentPart, ...]:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    parts: list[ContentPart] = []
    for item in raw:
        kind = item.get("type")
        if kind == "text":
            parts.append(TextPart(text=item.get("text", "")))
        elif kind == "image_url":
            image = item.get("image_url") or {}
            url = image.get("url", "") if isinstance(image, dict) else str(image)
            detail = image.get("detail") if isinstance(image, dict) else None
            parts.append(ImagePart(url=url, detail=detail))
    return tuple(parts)


def _message(raw: ChatMessageIn) -> MessageIR:
    role = "system" if raw.role == "developer" else raw.role
    if role not in _ROLES:
        raise InvalidRequestError(f"unsupported message role {raw.role!r}")
    tool_calls = tuple(
        ToolCallIR(
            id=tc.get("id", ""),
            name=tc.get("function", {}).get("name", ""),
            arguments=tc.get("function", {}).get("arguments", ""),
        )
        for tc in (raw.tool_calls or [])
    )
    return MessageIR(
        role=role,  # type: ignore[arg-type]
        content=_content(raw.content),
        tool_calls=tool_calls,
        tool_call_id=raw.tool_call_id,
        name=raw.name,
    )


def _tools(raw: list[dict[str, Any]] | None) -> tuple[ToolSpec, ...]:
    specs: list[ToolSpec] = []
    for tool in raw or []:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        specs.append(
            ToolSpec(
                name=fn.get("name", ""),
                description=fn.get("description"),
                parameters=fn.get("parameters"),
                strict=fn.get("strict"),
            )
        )
    return tuple(specs)


def _tool_choice(raw: Any) -> ToolChoice:
    """Map OpenAI's ``tool_choice`` (a string or ``{type:function, function:{name}}``) to the IR."""
    if isinstance(raw, dict):
        if raw.get("type") == "function":
            fn = raw.get("function") or {}
            name = fn.get("name") or raw.get("name")
            if name:
                return SpecificTool(name=str(name))
        return "auto"
    if raw == "none":
        return "none"
    if raw == "required":
        return "required"
    return "auto"


def chat_request_to_ir(req: ChatCompletionRequest) -> ChatRequestIR:
    """Map an OpenAI Chat Completions request to the internal IR."""
    stop: tuple[str, ...] = ()
    if isinstance(req.stop, str):
        stop = (req.stop,)
    elif isinstance(req.stop, list):
        stop = tuple(req.stop)
    sampling = Sampling(
        temperature=req.temperature,
        top_p=req.top_p,
        max_output_tokens=req.max_completion_tokens or req.max_tokens,
        stop=stop,
        seed=req.seed,
    )
    return ChatRequestIR(
        model=req.model,
        messages=tuple(_message(m) for m in req.messages),
        tools=_tools(req.tools),
        tool_choice=_tool_choice(req.tool_choice),
        parallel_tool_calls=req.parallel_tool_calls,
        effort=req.reasoning_effort,
        web_search=bool(req.web_search or req.web_search_options),
        response_format=req.response_format,
        sampling=sampling,
        stream=req.stream,
        include_usage=bool(req.stream_options and req.stream_options.include_usage),
        user=req.user,
        metadata=req.metadata or {},
    )
