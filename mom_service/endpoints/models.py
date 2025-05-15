from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


# Shared Models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: Optional[List[str]] = None  # For multimodal models


class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None


class ThinkingContextItem(BaseModel):
    model: str
    content: str
    usage: UsageInfo


# OpenAI Specific Models
class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class OpenAIChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None
    thinking_context: Optional[List[ThinkingContextItem]] = None
    total_cost_usd: Optional[float] = None


# Ollama Specific Models
class OllamaChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    format: Optional[Literal["json"]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    keep_alive: Optional[str] = None


class OllamaModelDetails(BaseModel):
    format: Optional[str] = "MoM-internal"
    family: Optional[str] = "MoM"
    families: Optional[List[str]] = ["MoM"]
    parameter_size: Optional[str] = "N/A"
    quantization_level: Optional[str] = "N/A"


class OllamaTagInfo(BaseModel):
    name: str
    modified_at: str  # ISO 8601 timestamp
    size: int  # In bytes
    digest: str  # SHA256 hash
    details: OllamaModelDetails


class OllamaShowRequest(BaseModel):
    name: str  # e.g., "mom_model_name:latest"


class OllamaShowResponse(BaseModel):
    modelfile: Optional[str] = None  # String representation of MoM config
    parameters: Optional[str] = None  # List of query LLMs and concluding LLM
    template: Optional[str] = None  # Chat template or system prompt info
    details: OllamaModelDetails
    license: Optional[str] = None


class OllamaTagsResponse(BaseModel):
    models: List[OllamaTagInfo]


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatMessage
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
