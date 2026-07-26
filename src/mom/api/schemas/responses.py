"""OpenAI Responses API request schema (Open Responses subset — stateless, function tools)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    text: dict[str, Any] | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    background: bool | None = None
    metadata: dict[str, Any] | None = None
