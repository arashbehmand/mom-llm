"""The canonical internal request (IR).

All three protocol surfaces (chat completions, responses, messages) translate to and from this
one neutral shape, so the engine never sees a wire format. It is deliberately OpenAI-chat-shaped
because that is LiteLLM's lingua franca — the adapter forwards it upstream nearly verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    url: str
    detail: str | None = None


ContentPart = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class ToolCallIR:
    id: str
    name: str
    arguments: str  # raw JSON string, as the model emitted it


@dataclass(frozen=True, slots=True)
class MessageIR:
    role: Role
    content: str | tuple[ContentPart, ...] = ""
    tool_calls: tuple[ToolCallIR, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @property
    def text(self) -> str:
        """Best-effort plain-text view of the content."""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    @property
    def has_images(self) -> bool:
        return not isinstance(self.content, str) and any(
            isinstance(part, ImagePart) for part in self.content
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(frozen=True, slots=True)
class SpecificTool:
    name: str


ToolChoice = Literal["auto", "none", "required"] | SpecificTool


@dataclass(frozen=True, slots=True)
class Sampling:
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ChatRequestIR:
    model: str
    messages: tuple[MessageIR, ...]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool | None = None
    effort: str | None = None  # raw effort token from the client, resolved by plan.py
    response_format: dict[str, Any] | None = None
    sampling: Sampling = field(default_factory=Sampling)
    stream: bool = False
    include_usage: bool = False
    user: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def has_images(self) -> bool:
        return any(m.has_images for m in self.messages)
