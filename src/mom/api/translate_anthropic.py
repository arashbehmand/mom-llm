"""Translate Anthropic Messages requests into the canonical IR."""

from __future__ import annotations

import json
from typing import Any

from mom.api.schemas.anthropic import AnthropicMessage, MessagesRequest
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


# Anthropic thinking budget_tokens -> effort bucket (calibrated to Claude Code's 4k/10k/31999).
_THINKING_BUCKETS = (
    (2048, "minimal"),
    (8192, "low"),
    (16384, "medium"),
    (32768, "high"),
    (65536, "xhigh"),
)


def _system_text(system: str | list[dict[str, Any]] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n\n".join(block.get("text", "") for block in system if block.get("type") == "text")


def _effort(thinking: dict[str, Any] | None) -> str | None:
    if not thinking:
        return None
    kind = thinking.get("type")
    if kind == "disabled":
        return "none"
    if kind == "enabled":
        budget = int(thinking.get("budget_tokens", 0) or 0)
        for limit, label in _THINKING_BUCKETS:
            if budget <= limit:
                return label
        return "max"
    return None


def _image_part(source: dict[str, Any]) -> ImagePart | None:
    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        return ImagePart(url=f"data:{media};base64,{source.get('data', '')}")
    if source.get("type") == "url":
        return ImagePart(url=source.get("url", ""))
    return None


def _translate_message(message: AnthropicMessage) -> list[MessageIR]:
    """One Anthropic message may expand into tool messages plus a user/assistant message."""
    if isinstance(message.content, str):
        return [MessageIR(role=message.role, content=message.content)]  # type: ignore[arg-type]

    out: list[MessageIR] = []
    parts: list[ContentPart] = []
    tool_calls: list[ToolCallIR] = []
    for block in message.content:
        kind = block.get("type")
        if kind == "text":
            parts.append(TextPart(text=block.get("text", "")))
        elif kind == "image":
            image = _image_part(block.get("source", {}))
            if image is not None:
                parts.append(image)
        elif kind == "tool_use":
            tool_calls.append(
                ToolCallIR(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                )
            )
        elif kind == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content if c.get("type") == "text")
            out.append(
                MessageIR(
                    role="tool",
                    content=content,
                    tool_call_id=block.get("tool_use_id"),
                )
            )
        # thinking / redacted_thinking blocks are our own replayed output — dropped.

    if message.role == "assistant":
        out.append(
            MessageIR(
                role="assistant",
                content=tuple(parts) if parts else "",
                tool_calls=tuple(tool_calls),
            )
        )
    elif parts:
        out.append(MessageIR(role="user", content=tuple(parts)))
    return out


def _tools(raw: list[dict[str, Any]] | None) -> tuple[ToolSpec, ...]:
    specs: list[ToolSpec] = []
    for tool in raw or []:
        if "name" not in tool:
            continue
        specs.append(
            ToolSpec(
                name=tool["name"],
                description=tool.get("description"),
                parameters=tool.get("input_schema"),
            )
        )
    return tuple(specs)


def _tool_choice(raw: dict[str, Any] | None) -> ToolChoice:
    if not raw:
        return "auto"
    kind = raw.get("type")
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and raw.get("name"):
        return SpecificTool(name=raw["name"])
    return "auto"


def messages_request_to_ir(req: MessagesRequest, *, stream: bool) -> ChatRequestIR:
    """Map an Anthropic Messages request to the internal IR."""
    messages: list[MessageIR] = []
    system = _system_text(req.system)
    if system:
        messages.append(MessageIR(role="system", content=system))
    for message in req.messages:
        messages.extend(_translate_message(message))

    sampling = Sampling(
        temperature=req.temperature,
        top_p=req.top_p,
        max_output_tokens=req.max_tokens,
        stop=tuple(req.stop_sequences or ()),
    )
    return ChatRequestIR(
        model=req.model,
        messages=tuple(messages),
        tools=_tools(req.tools),
        tool_choice=_tool_choice(req.tool_choice),
        effort=_effort(req.thinking),
        sampling=sampling,
        stream=stream,
        metadata={k: str(v) for k, v in (req.metadata or {}).items()},
    )
