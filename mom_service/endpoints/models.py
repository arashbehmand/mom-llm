from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# Multimodal Content Models (OpenAI Vision API format)
class TextContentPart(BaseModel):
    """Text content part for multimodal messages"""

    type: Literal["text"] = "text"
    text: str


class ImageUrlDetail(BaseModel):
    """Image URL with optional detail level"""

    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageContentPart(BaseModel):
    """Image content part for multimodal messages"""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlDetail


class ThinkingContentPart(BaseModel):
    """
    Thinking content part for extended thinking models (e.g., Claude with extended thinking).

    This represents the model's internal reasoning process. The thinking field contains
    the summarized reasoning, and the signature field is used for verification.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


# Union of all content part types
ContentPart = Union[TextContentPart, ImageContentPart, ThinkingContentPart]


# Shared Models
class ChatMessage(BaseModel):
    """
    Chat message supporting both simple text and multimodal content.

    For simple text messages:
        content: str

    For multimodal messages (OpenAI Vision API format):
        content: list[ContentPart]
    """

    role: Literal["system", "user", "assistant"]
    content: Union[str, list[ContentPart]]
    images: list[str] = Field(
        default_factory=list
    )  # For alternative multimodal format (empty list by default)


class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None

    @classmethod
    def from_litellm_usage(cls, usage: Any, **kwargs) -> "UsageInfo":
        """
        Create UsageInfo from LiteLLM usage object or response.

        Args:
            usage: LiteLLM usage object or dict
            response_obj: Optional full LiteLLM response object for cost calculation
            is_cached: Whether this is a cached response (cost should be 0.0)
            pricing_config: Optional PricingConfig to override default LiteLLM pricing
            model_name: Optional model name for cost calculation when response_obj is not available

        Returns:
            UsageInfo instance with tokens and cost
        """
        # Extract optional parameters from kwargs for backward compatibility
        response_obj = kwargs.get("response_obj")
        is_cached = kwargs.get("is_cached", False)
        pricing_config = kwargs.get("pricing_config")
        model_name = kwargs.get("model_name")

        if usage is None:
            return cls(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0 if is_cached else None,
            )

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
        elif model_name is not None:
            # Try to calculate cost using model name and token counts
            try:
                import logging

                import litellm

                logger = logging.getLogger(__name__)

                logger.info(
                    f"Attempting cost calculation for {model_name}: "
                    f"input={prompt_tokens}, output={completion_tokens}"
                )

                # Use cost_per_token which returns (prompt_cost, completion_cost)
                prompt_cost_usd, completion_cost_usd = litellm.cost_per_token(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                calculated_cost = (prompt_cost_usd or 0.0) + (completion_cost_usd or 0.0)

                logger.info(
                    f"LiteLLM cost_per_token returned: prompt=${prompt_cost_usd:.6f}, "
                    f"completion=${completion_cost_usd:.6f}, total=${calculated_cost:.6f}"
                )
                cost = float(calculated_cost) if calculated_cost is not None else 0.0
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Cost calculation failed for {model_name}: {e}", exc_info=True)
                # Cost calculation failed, default to 0.0 for safety
                cost = 0.0

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )


@dataclass
class LLMCallParams:
    """
    Dataclass to encapsulate LLM call parameters so they can be passed
    around as a single object instead of multiple positional arguments.
    """

    model: str
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False


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
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


class OpenAIChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None
    thinking_context: Optional[list[ThinkingContextItem]] = None
    total_cost_usd: Optional[float] = None
