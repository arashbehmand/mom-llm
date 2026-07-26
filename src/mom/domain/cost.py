"""Cache-aware cost computation (pure).

Cost is derived from token usage and per-model pricing; there is no separate cost engine with its
own rounding rules. Cached prompt tokens are billed at the provider's cache-read rate and
cache-write tokens at the cache-write rate — the whole reason ``Usage`` carries them as
first-class fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from mom.domain.results import Usage


@dataclass(frozen=True, slots=True)
class Pricing:
    """Per-1M-token prices for one model. Zero means free / unpriced for that dimension."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    reasoning_per_1m: float = 0.0
    cache_read_per_1m: float = 0.0
    cache_write_per_1m: float = 0.0


def compute_cost(usage: Usage, pricing: Pricing | None) -> float:
    """Dollar cost of one call. Returns 0.0 when the model has no pricing configured.

    ``cached_prompt_tokens`` are treated as a subset of ``prompt_tokens`` (OpenAI/Gemini style:
    cached tokens counted *inside* the prompt) and billed at the cache-read rate; the remainder is
    billed at the full input rate. ``cache_write_tokens`` (Anthropic cache creation, counted
    *outside* the prompt) are billed additively at the cache-write rate.
    """
    if pricing is None:
        return 0.0
    billable_input = max(0, usage.prompt_tokens - usage.cached_prompt_tokens)
    total = (
        billable_input * pricing.input_per_1m
        + usage.cached_prompt_tokens * pricing.cache_read_per_1m
        + usage.cache_write_tokens * pricing.cache_write_per_1m
        + usage.completion_tokens * pricing.output_per_1m
        + usage.reasoning_tokens * pricing.reasoning_per_1m
    )
    return total / 1_000_000.0
