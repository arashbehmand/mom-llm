"""
Utilities for unified pricing and cost tracking.

This module provides functionality to normalize token counts based on a unified
pricing model ($1/1M input tokens, $10/1M output tokens) while preserving the
actual cost of LLM calls.
"""

import logging

logger = logging.getLogger(__name__)

# Unified pricing model
UNIFIED_INPUT_COST_PER_TOKEN = 0.000001  # $1 per 1M tokens = $0.000001 per token
UNIFIED_OUTPUT_COST_PER_TOKEN = 0.00001  # $10 per 1M tokens = $0.00001 per token


def calculate_normalized_tokens(
    actual_cost: float,
    actual_input_tokens: int,
    actual_output_tokens: int,
) -> tuple[int, int]:
    """
    Calculate normalized token counts that match the actual cost using unified pricing.

    The unified pricing model is:
    - $1 per 1M input tokens
    - $10 per 1M output tokens

    This function adjusts the reported token counts so that when multiplied by the
    unified pricing, they equal the actual cost. The ratio between input and output
    tokens is preserved from the actual token counts.

    Args:
        actual_cost: The actual cost of the LLM call in USD
        actual_input_tokens: The actual number of input tokens
        actual_output_tokens: The actual number of output tokens

    Returns:
        Tuple of (normalized_input_tokens, normalized_output_tokens)

    Examples:
        >>> # If actual cost is $0.05 with 100 input and 200 output tokens
        >>> calculate_normalized_tokens(0.05, 100, 200)
        # Returns normalized tokens that sum to $0.05 with unified pricing
    """
    if actual_cost <= 0:
        # No cost, return actual tokens
        return actual_input_tokens, actual_output_tokens

    if actual_input_tokens == 0 and actual_output_tokens == 0:
        # No tokens, but there's a cost - distribute to output tokens
        # (output is more expensive in unified model)
        normalized_output = int(actual_cost / UNIFIED_OUTPUT_COST_PER_TOKEN)
        return 0, normalized_output

    # Calculate the ratio of input to output tokens
    total_actual_tokens = actual_input_tokens + actual_output_tokens
    input_ratio = actual_input_tokens / total_actual_tokens if total_actual_tokens > 0 else 0.5
    output_ratio = actual_output_tokens / total_actual_tokens if total_actual_tokens > 0 else 0.5

    # We need to solve for normalized_input and normalized_output such that:
    # normalized_input * UNIFIED_INPUT_COST_PER_TOKEN + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN = actual_cost
    #
    # We want to preserve the ratio:
    # normalized_input / (normalized_input + normalized_output) ≈ input_ratio
    #
    # Let's use the ratio to express normalized_output in terms of normalized_input:
    # normalized_input = input_ratio * total_normalized
    # normalized_output = output_ratio * total_normalized
    #
    # Substituting into cost equation:
    # input_ratio * total_normalized * UNIFIED_INPUT_COST_PER_TOKEN +
    # output_ratio * total_normalized * UNIFIED_OUTPUT_COST_PER_TOKEN = actual_cost
    #
    # total_normalized = actual_cost / (input_ratio * UNIFIED_INPUT_COST_PER_TOKEN +
    #                                     output_ratio * UNIFIED_OUTPUT_COST_PER_TOKEN)

    weighted_cost_per_token = (
        input_ratio * UNIFIED_INPUT_COST_PER_TOKEN + output_ratio * UNIFIED_OUTPUT_COST_PER_TOKEN
    )

    if weighted_cost_per_token <= 0:
        # Edge case: distribute all cost to output tokens
        normalized_output = int(actual_cost / UNIFIED_OUTPUT_COST_PER_TOKEN)
        return 0, normalized_output

    total_normalized = actual_cost / weighted_cost_per_token

    normalized_input = int(input_ratio * total_normalized)
    normalized_output = int(output_ratio * total_normalized)

    # Verify and adjust if needed to match the exact cost
    calculated_cost = (
        normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
        + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
    )

    # If there's a rounding error, adjust output tokens (since they're more expensive)
    if abs(calculated_cost - actual_cost) > 0.000001:  # Allow 0.1 cent tolerance
        # Recalculate output tokens to exactly match cost
        remaining_cost = actual_cost - (normalized_input * UNIFIED_INPUT_COST_PER_TOKEN)
        if remaining_cost >= 0:
            normalized_output = int(remaining_cost / UNIFIED_OUTPUT_COST_PER_TOKEN)
        else:
            # Cost was mostly from input, adjust
            normalized_input = int(actual_cost / UNIFIED_INPUT_COST_PER_TOKEN)
            normalized_output = 0

    logger.debug(
        f"Normalized tokens: actual_cost=${actual_cost:.6f}, "
        f"actual_tokens=({actual_input_tokens} in, {actual_output_tokens} out), "
        f"normalized_tokens=({normalized_input} in, {normalized_output} out), "
        f"verification_cost=${normalized_input * UNIFIED_INPUT_COST_PER_TOKEN + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN:.6f}"
    )

    return normalized_input, normalized_output


def calculate_cost_from_normalized_tokens(
    normalized_input_tokens: int, normalized_output_tokens: int
) -> float:
    """
    Calculate cost from normalized tokens using unified pricing.

    Args:
        normalized_input_tokens: Normalized input token count
        normalized_output_tokens: Normalized output token count

    Returns:
        Cost in USD
    """
    return (
        normalized_input_tokens * UNIFIED_INPUT_COST_PER_TOKEN
        + normalized_output_tokens * UNIFIED_OUTPUT_COST_PER_TOKEN
    )
