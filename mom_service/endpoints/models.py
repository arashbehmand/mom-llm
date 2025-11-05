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

    @classmethod
    def from_litellm_usage(
        cls,
        usage: Any,
        response_obj: Any = None,
        is_cached: bool = False,
        pricing_config: Any = None
    ) -> "UsageInfo":
        """
        Create UsageInfo from LiteLLM usage object or response.

        Args:
            usage: LiteLLM usage object or dict
            response_obj: Optional full LiteLLM response object for cost calculation
            is_cached: Whether this is a cached response (cost should be 0.0)
            pricing_config: Optional PricingConfig to override default LiteLLM pricing

        Returns:
            UsageInfo instance with tokens and cost
        """
        if usage is None:
            return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0 if is_cached else None)

        # Extract token counts
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            # LiteLLM usage object
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

        # Calculate cost
        cost = None
        if is_cached:
            # Cached responses have zero cost
            cost = 0.0
        elif pricing_config is not None:
            # Use custom pricing configuration
            cost = pricing_config.calculate_cost(prompt_tokens, completion_tokens)
        elif response_obj is not None:
            try:
                import litellm
                calculated_cost = litellm.completion_cost(completion_response=response_obj)
                # Ensure cost is always a number or None, never something else
                cost = float(calculated_cost) if calculated_cost is not None else 0.0
            except Exception:
                # Cost calculation failed, default to 0.0 for safety
                cost = 0.0

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )


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
