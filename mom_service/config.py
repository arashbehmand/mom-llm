"""
Configuration management for the Mixture of Models (MoM) service.

This module defines Pydantic models for all configuration entities and provides
the load_config() function to parse and validate the config.yaml file.

Configuration structure:
- LLMDefinition: Individual LLM provider configurations (model, API keys, parameters)
- ModelConfig: MoM model definitions (which LLMs to query, concluding LLM, prompts)
- ServiceConfig: Service-level settings (timeouts, caching, retries, exposed APIs)
- LangfuseConfig: Optional observability/tracing configuration
- MoMConfig: Top-level configuration combining all of the above

The configuration file is loaded from one of these locations (in order):
1. Path explicitly provided to load_config()
2. MOM_CONFIG_PATH environment variable
3. ./config.yaml (current directory)
4. ../config.yaml (parent directory, relative to this module)
"""

import os
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ValidationError


class PricingConfig(BaseModel):
    """Custom pricing configuration for an LLM model"""

    prompt_cost_per_token: Optional[float] = (
        None  # Cost per prompt token (e.g., 0.00003 for $0.03/1K tokens)
    )
    completion_cost_per_token: Optional[float] = None  # Cost per completion token (text output)
    reasoning_cost_per_token: Optional[float] = (
        None  # Cost per reasoning token (thinking/internal reasoning)
    )

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        text_tokens: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate total cost based on token counts with optional reasoning token breakdown.

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Total completion tokens (used if no text/reasoning breakdown)
            reasoning_tokens: Number of reasoning tokens (optional, for models with thinking mode)
            text_tokens: Number of text output tokens (optional, for models with thinking mode)

        Returns:
            Tuple of (total_cost, cost_breakdown_dict)
        """
        prompt_cost = (
            (prompt_tokens * self.prompt_cost_per_token) if self.prompt_cost_per_token else 0.0
        )

        # If reasoning token pricing is configured and we have breakdown, use it
        if self.reasoning_cost_per_token and (reasoning_tokens > 0 or text_tokens > 0):
            text_cost = (
                (text_tokens * self.completion_cost_per_token)
                if self.completion_cost_per_token
                else 0.0
            )
            reasoning_cost = reasoning_tokens * self.reasoning_cost_per_token

            return prompt_cost + text_cost + reasoning_cost, {
                "input": prompt_cost,
                "output_text": text_cost,
                "output_reasoning": reasoning_cost,
            }
        # Standard pricing without reasoning breakdown
        completion_cost = (
            (completion_tokens * self.completion_cost_per_token)
            if self.completion_cost_per_token
            else 0.0
        )
        return prompt_cost + completion_cost, {
            "input": prompt_cost,
            "output": completion_cost,
        }


class LLMDefinition(BaseModel):
    name: str  # Unique identifier for this LLM definition
    model: str
    api_key_env: str
    params: Optional[dict[str, Any]] = None
    pricing: Optional[PricingConfig] = None  # Custom pricing override


class PromptDefinition(BaseModel):
    name: str
    content: str


class ModelConfig(BaseModel):
    name: str
    llms_to_query: list[str]
    concluding_llm: str
    concluding_prompt: Optional[str] = None  # Name of the PromptDefinition to use
    include_thinking_context: bool = False  # Default to false if not specified


class ServiceConfig(BaseModel):
    timeout_seconds: int = 30
    exposed_apis: list[str] = ["openai"]  # Default to only openai if not specified
    cache_enabled: bool = False  # Add cache enabled flag
    max_llm_retries: int = 3  # Max retries for individual LLM calls
    llm_retry_delay_seconds: int = 2  # Delay between LLM call retries


class LangfuseConfig(BaseModel):
    public_key_env: str
    secret_key_env: str
    host_env: str


class MoMConfig(BaseModel):
    llm_definitions: list[LLMDefinition]
    prompt_definitions: Optional[list[PromptDefinition]] = None
    models: list[ModelConfig]
    service: ServiceConfig
    langfuse: Optional[LangfuseConfig] = None


def load_config(config_path: str = None) -> MoMConfig:
    # If config_path is provided, use it directly.
    # Otherwise, check the MOM_CONFIG_PATH environment variable.
    # Fallback to default search paths if neither is set.
    path_to_load = config_path or os.getenv("MOM_CONFIG_PATH")

    if path_to_load:
        if not os.path.isfile(path_to_load):
            raise FileNotFoundError(f"Config file not found at specified path: {path_to_load}")
        search_paths = [path_to_load]
    else:
        # Default search paths
        search_paths = [
            os.path.join(os.getcwd(), "config.yaml"),
            os.path.join(
                os.path.dirname(__file__), "..", "config.yaml"
            ),  # Adjusted for being in mom_service/
        ]

    for path in search_paths:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            try:
                config = MoMConfig(**raw)
                # Log the path that was successfully loaded
                # Use a simple print here as logger might not be configured yet
                print(f"--- Config loaded successfully from: {path} ---")
                return config
            except ValidationError as e:
                raise RuntimeError(f"Invalid configuration in {path}: {e}") from e

    raise FileNotFoundError(f"config.yaml not found in any of the search paths: {search_paths}")
