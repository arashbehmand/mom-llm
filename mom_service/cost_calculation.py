"""
Utilities for calculating detailed costs including reasoning tokens.
"""
import logging
from typing import Dict, Optional, Tuple

import litellm

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash pricing (per 1M tokens)
GEMINI_25_FLASH_PRICES = {
    "input": 0.15 / 1_000_000,  # $0.15 per 1M tokens
    "output_text": 0.60 / 1_000_000,  # $0.60 per 1M tokens
    "output_reasoning": 3.50 / 1_000_000,  # $3.50 per 1M tokens (thinking mode)
}


def calculate_detailed_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    completion_tokens_details: Optional[Dict] = None,
    prompt_tokens_details: Optional[Dict] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate detailed cost breakdown for LLM calls, handling reasoning tokens separately.

    Args:
        model_name: The LiteLLM model identifier
        prompt_tokens: Total prompt tokens
        completion_tokens: Total completion tokens
        completion_tokens_details: Detailed breakdown (e.g., {'reasoning_tokens': X, 'text_tokens': Y})
        prompt_tokens_details: Detailed prompt breakdown (e.g., {'text_tokens': X})

    Returns:
        Tuple of (total_cost, cost_breakdown_dict)
        cost_breakdown_dict has keys like: 'input', 'output_text', 'output_reasoning'
    """
    cost_breakdown = {}

    # Check if this is a Gemini 2.5 Flash model with reasoning tokens
    is_gemini_25_flash = "gemini-2.5-flash" in model_name.lower() or "gemini/gemini-2.5-flash" in model_name.lower()

    if is_gemini_25_flash and completion_tokens_details:
        # Gemini 2.5 Flash with detailed token breakdown
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
        text_tokens = completion_tokens_details.get("text_tokens", 0)

        logger.info(
            f"Using Gemini 2.5 Flash pricing for {model_name}: "
            f"input={prompt_tokens}, text_output={text_tokens}, reasoning_output={reasoning_tokens}"
        )

        # Calculate costs separately
        input_cost = prompt_tokens * GEMINI_25_FLASH_PRICES["input"]
        text_output_cost = text_tokens * GEMINI_25_FLASH_PRICES["output_text"]
        reasoning_output_cost = reasoning_tokens * GEMINI_25_FLASH_PRICES["output_reasoning"]

        cost_breakdown = {
            "input": input_cost,
            "output_text": text_output_cost,
            "output_reasoning": reasoning_output_cost,
        }

        total_cost = input_cost + text_output_cost + reasoning_output_cost

        logger.info(
            f"Detailed cost for {model_name}: input=${input_cost:.6f}, "
            f"text_output=${text_output_cost:.6f}, reasoning_output=${reasoning_output_cost:.6f}, "
            f"total=${total_cost:.6f}"
        )

    else:
        # Fall back to LiteLLM's cost_per_token for models without detailed breakdown
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            cost_breakdown = {
                "input": prompt_cost or 0.0,
                "output": completion_cost or 0.0,
            }

            total_cost = (prompt_cost or 0.0) + (completion_cost or 0.0)

            logger.info(
                f"Standard cost for {model_name}: input=${prompt_cost:.6f}, "
                f"output=${completion_cost:.6f}, total=${total_cost:.6f}"
            )

        except Exception as e:
            logger.error(f"Cost calculation failed for {model_name}: {e}", exc_info=True)
            total_cost = 0.0
            cost_breakdown = {}

    return total_cost, cost_breakdown
