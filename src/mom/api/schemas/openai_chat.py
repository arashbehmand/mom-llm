"""OpenAI Chat Completions wire models (request + response)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessageIn]
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None


# ---- response ----
class ChatMessageOut(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessageOut
    finish_reason: str


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = Field(default_factory=PromptTokensDetails)
    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails
    )


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage
