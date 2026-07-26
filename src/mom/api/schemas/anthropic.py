"""Anthropic Messages API request schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[dict[str, Any]]


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class CountTokensRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
