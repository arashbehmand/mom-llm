"""
Cost calculation utilities for LLM calls with support for reasoning tokens.

Some models (e.g., Gemini 2.5 Flash, OpenAI o1) charge different rates for
reasoning tokens vs text tokens. This module handles granular cost breakdown.

Priority order for pricing:
1. Custom pricing from model config (llm_def.pricing)
2. LiteLLM's built-in pricing database
3. Return $0 if no pricing available
"""

import logging
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


def calculate_detailed_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    completion_tokens_details: Optional[dict] = None,
    pricing_config=None,  # PricingConfig from LLMDefinition
) -> tuple[float, dict[str, float]]:
    """
    Calculate cost breakdown handling reasoning tokens separately from text tokens.

    For models that support reasoning tokens (internal thinking), calculates separate
    costs for reasoning vs text output. Uses pricing from:
    1. Custom pricing config if provided
    2. LiteLLM's pricing database
    3. Returns $0 if no pricing available (with warning)

    Args:
        model_name: LiteLLM model identifier (e.g., "gemini/gemini-2.5-flash")
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Total completion tokens
        completion_tokens_details: Optional breakdown with 'reasoning_tokens', 'text_tokens'
        prompt_tokens_details: Optional prompt token breakdown (for future use)
        pricing_config: Optional PricingConfig from LLMDefinition

    Returns:
        Tuple of (total_cost, cost_breakdown_dict)
        - total_cost: Total USD cost for this call
        - cost_breakdown: Dict with keys like 'input', 'output_text', 'output_reasoning'

    Example with custom pricing:
        >>> pricing = PricingConfig(
        ...     prompt_cost_per_token=0.15 / 1_000_000,
        ...     completion_cost_per_token=0.60 / 1_000_000,
        ...     reasoning_cost_per_token=3.50 / 1_000_000,
        ... )
        >>> total, breakdown = calculate_detailed_cost(
        ...     "gemini/gemini-2.5-flash",
        ...     prompt_tokens=1000,
        ...     completion_tokens=825,
        ...     completion_tokens_details={"reasoning_tokens": 764, "text_tokens": 61},
        ...     pricing_config=pricing,
        ... )
        >>> print(f"Total: ${total:.6f}")
        Total: $0.003024
    """
    # Extract reasoning and text token counts if available
    reasoning_tokens = 0
    text_tokens = 0
    if completion_tokens_details:
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
        text_tokens = completion_tokens_details.get("text_tokens", 0)

        # Validate token counts
        if reasoning_tokens + text_tokens != completion_tokens:
            logger.warning(
                f"Token count mismatch for {model_name}: "
                f"reasoning({reasoning_tokens}) + text({text_tokens}) != total({completion_tokens})"
            )

    # Priority 1: Use custom pricing config if provided
    if pricing_config:
        try:
            total_cost, cost_breakdown = pricing_config.calculate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                text_tokens=text_tokens,
            )

            logger.info(
                f"Used custom pricing for {model_name}: "
                f"prompt={prompt_tokens}, completion={completion_tokens}, "
                f"reasoning={reasoning_tokens}, text={text_tokens}, "
                f"total=${total_cost:.6f}"
            )

            return total_cost, cost_breakdown

        except Exception as e:
            logger.error(f"Custom pricing calculation failed for {model_name}: {e}")
            # Fall through to LiteLLM pricing

    # Priority 2: Fall back to LiteLLM's cost_per_token
    try:
        prompt_cost_usd, completion_cost_usd = litellm.cost_per_token(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        total_cost = (prompt_cost_usd or 0.0) + (completion_cost_usd or 0.0)

        # Build cost breakdown
        # Note: LiteLLM doesn't separate reasoning vs text, so we report as single output
        cost_breakdown = {
            "input": prompt_cost_usd or 0.0,
            "output": completion_cost_usd or 0.0,
        }

        # If we have reasoning tokens but no custom pricing, warn user
        if reasoning_tokens > 0 and not pricing_config:
            logger.warning(
                f"Model {model_name} has {reasoning_tokens} reasoning tokens "
                f"but no custom pricing configured. LiteLLM may not apply "
                f"differential pricing for reasoning tokens. Consider adding "
                f"'reasoning_cost_per_token' to model config."
            )

        logger.debug(
            f"LiteLLM pricing for {model_name}: "
            f"prompt=${prompt_cost_usd:.6f}, "
            f"completion=${completion_cost_usd:.6f}, "
            f"total=${total_cost:.6f}"
        )

        return total_cost, cost_breakdown

    except Exception as e:
        logger.warning(
            f"Cost calculation failed for {model_name} (no pricing available): {e}. "
            f"Consider adding custom pricing config."
        )
        return 0.0, {}
