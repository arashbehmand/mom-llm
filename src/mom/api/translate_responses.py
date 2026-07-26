"""Translate OpenAI Responses requests into the canonical IR (stateless subset)."""

from __future__ import annotations

from typing import Any

from mom.api.schemas.responses import ResponsesRequest
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


_SUPPORTED_TOOL_TYPES = frozenset({"function", "web_search", "web_search_preview"})


def _content(raw: list[dict[str, Any]]) -> tuple[ContentPart, ...]:
    parts: list[ContentPart] = []
    for item in raw:
        kind = item.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append(TextPart(text=item.get("text", "")))
        elif kind in ("input_image", "image"):
            url = item.get("image_url") or item.get("url") or ""
            if isinstance(url, dict):
                url = url.get("url", "")
            parts.append(ImagePart(url=url))
    return tuple(parts)


def _input_messages(value: str | list[dict[str, Any]]) -> list[MessageIR]:
    if isinstance(value, str):
        return [MessageIR(role="user", content=value)]
    messages: list[MessageIR] = []
    for item in value:
        kind = item.get("type", "message")
        if kind == "message":
            role = "system" if item.get("role") == "developer" else item.get("role", "user")
            content = item.get("content", "")
            body: str | tuple[ContentPart, ...] = (
                content if isinstance(content, str) else _content(content)
            )
            messages.append(MessageIR(role=role, content=body))  # type: ignore[arg-type]
        elif kind == "function_call":
            messages.append(
                MessageIR(
                    role="assistant",
                    tool_calls=(
                        ToolCallIR(
                            id=item.get("call_id", item.get("id", "")),
                            name=item.get("name", ""),
                            arguments=item.get("arguments", ""),
                        ),
                    ),
                )
            )
        elif kind == "function_call_output":
            output = item.get("output", "")
            messages.append(
                MessageIR(
                    role="tool",
                    content=output if isinstance(output, str) else str(output),
                    tool_call_id=item.get("call_id"),
                )
            )
        # reasoning items are our own replayed output — dropped.
    return messages


def _tools(
    raw: list[dict[str, Any]] | None,
) -> tuple[tuple[ToolSpec, ...], bool, tuple[dict[str, Any], ...]]:
    specs: list[ToolSpec] = []
    web_search = False
    mcp_tools: list[dict[str, Any]] = []
    for tool in raw or []:
        kind = tool.get("type")
        if kind == "mcp":
            # Carried opaquely; plan.py forwards it to an MCP-capable synthesizer, else 400s.
            mcp_tools.append(tool)
            continue
        if kind and kind not in _SUPPORTED_TOOL_TYPES:
            raise InvalidRequestError(f"unsupported tool type {kind!r}")
        if kind in ("web_search", "web_search_preview"):
            web_search = True
            continue
        specs.append(
            ToolSpec(
                name=tool.get("name", ""),
                description=tool.get("description"),
                parameters=tool.get("parameters"),
                strict=tool.get("strict"),
            )
        )
    return tuple(specs), web_search, tuple(mcp_tools)


def _tool_choice(raw: Any) -> ToolChoice:
    if isinstance(raw, dict) and raw.get("type") == "function" and raw.get("name"):
        return SpecificTool(name=raw["name"])
    if raw == "none":
        return "none"
    if raw == "required":
        return "required"
    return "auto"


def responses_request_to_ir(req: ResponsesRequest, *, stream: bool) -> ChatRequestIR:
    """Map an OpenAI Responses request to the internal IR (raises on stateful features)."""
    if req.previous_response_id:
        raise InvalidRequestError(
            "mom-llm is stateless: resend the full input each turn (store=false operation)"
        )
    if req.background:
        raise InvalidRequestError("background responses are not supported")

    messages: list[MessageIR] = []
    if req.instructions:
        messages.append(MessageIR(role="system", content=req.instructions))
    messages.extend(_input_messages(req.input))

    tools, web_search, mcp_tools = _tools(req.tools)
    effort = None
    if req.reasoning:
        effort = req.reasoning.get("effort")
        if effort == "auto":
            effort = None
    response_format = (req.text or {}).get("format")
    sampling = Sampling(
        temperature=req.temperature,
        top_p=req.top_p,
        max_output_tokens=req.max_output_tokens,
    )
    return ChatRequestIR(
        model=req.model,
        messages=tuple(messages),
        tools=tools,
        mcp_tools=mcp_tools,
        tool_choice=_tool_choice(req.tool_choice),
        parallel_tool_calls=req.parallel_tool_calls,
        effort=effort,
        web_search=web_search,
        response_format=response_format if isinstance(response_format, dict) else None,
        sampling=sampling,
        stream=stream,
        metadata={k: str(v) for k, v in (req.metadata or {}).items()},
    )
