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

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError


class PricingConfig(BaseModel):
    """Custom pricing configuration for an LLM model"""

    prompt_cost_per_token: float | None = (
        None  # Cost per prompt token (e.g., 0.00003 for $0.03/1K tokens)
    )
    completion_cost_per_token: float | None = None  # Cost per completion token (text output)
    reasoning_cost_per_token: float | None = (
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
    api_mode: Literal["auto", "completion", "responses"] = "auto"
    params: dict[str, Any] | None = None
    pricing: PricingConfig | None = None  # Custom pricing override


class PromptDefinition(BaseModel):
    name: str
    content: str


class ModelConfig(BaseModel):
    name: str
    llms_to_query: list[str]
    concluding_llm: str
    concluding_prompt: str | None = None  # Name of the PromptDefinition to use
    include_thinking_context: bool = False  # Default to false if not specified


class ServiceConfig(BaseModel):
    timeout_seconds: int = 30
    exposed_apis: list[str] = ["openai"]  # Default to only openai if not specified
    cache_enabled: bool = False  # Add cache enabled flag
    max_llm_retries: int = 3  # Max retries for individual LLM calls
    llm_retry_delay_seconds: int = 2  # Delay between LLM call retries
    enable_mom_debug_model: bool = False  # Auto-create mom-debug model with all LLMs


class LangfuseConfig(BaseModel):
    public_key_env: str
    secret_key_env: str
    host_env: str


class MoMConfig(BaseModel):
    llm_definitions: list[LLMDefinition]
    prompt_definitions: list[PromptDefinition] | None = None
    models: list[ModelConfig]
    service: ServiceConfig
    langfuse: LangfuseConfig | None = None

    def __init__(self, **data: Any) -> None:
        # Pre-process llm_definitions if they are raw dictionaries (from yaml)
        # The validation logic _expand_llm_definitions handles base_name/variants expansion
        # converting them into flat dictionary definitions that can be parsed into LLMDefinition
        if (
            "llm_definitions" in data
            and isinstance(data["llm_definitions"], list)
            and data["llm_definitions"]
            and isinstance(data["llm_definitions"][0], dict)
        ):
            # Check if expansion is needed (i.e. if we have dicts, not objects yet)
            data["llm_definitions"] = _expand_llm_definitions(data["llm_definitions"])

        super().__init__(**data)


DEFAULT_API_KEY_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

MOM_DEBUG_MODEL_NAME = "mom-debug"
MOM_DEBUG_DEFAULT_PROMPT = "synth_default"


def _infer_api_key_env(model: str | None) -> str | None:
    if not isinstance(model, str) or not model:
        return None

    provider, _, _ = model.partition("/")
    if provider:
        return DEFAULT_API_KEY_ENV_BY_PROVIDER.get(provider)
    return None


def _validate_entry_keys(entry: dict[str, Any], allowed_keys: set[str], path: str) -> None:
    unknown_keys = set(entry.keys()) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"{path} contains unsupported keys: {unknown}")


def _validate_optional_mapping(value: Any, field_name: str, path: str) -> None:
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{path}.{field_name} must be a mapping when provided.")


def _deep_merge_dict(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if base is None and override is None:
        return None

    result = deepcopy(base) if base is not None else {}
    for key, value in (override or {}).items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _build_llm_definition(
    name: str,
    state: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    model = state.get("model")
    api_key_env = state.get("api_key_env")

    if not isinstance(name, str) or not name:
        raise ValueError(f"{path} resolved to an invalid LLM name.")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path} is missing required field 'model'.")
    if api_key_env is None:
        api_key_env = _infer_api_key_env(model)
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError(
            f"{path} is missing field 'api_key_env' and no provider default could be inferred "
            f"from model '{model}'."
        )

    llm_definition = {
        "name": name,
        "model": model,
        "api_key_env": api_key_env,
    }
    if state.get("api_mode") is not None:
        llm_definition["api_mode"] = state["api_mode"]
    if state.get("params") is not None:
        llm_definition["params"] = state["params"]
    if state.get("pricing") is not None:
        llm_definition["pricing"] = state["pricing"]
    return llm_definition


def _expand_llm_definitions(raw_definitions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_definitions, list):
        raise ValueError("llm_definitions must be a list.")

    expanded: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    explicit_allowed_keys = {"name", "model", "api_key_env", "api_mode", "params", "pricing"}
    base_allowed_keys = {
        "base_name",
        "model",
        "api_key_env",
        "api_mode",
        "params",
        "pricing",
        "variants",
    }
    variant_allowed_keys = {
        "name",
        "suffix",
        "model",
        "api_key_env",
        "api_mode",
        "params",
        "pricing",
        "variants",
    }

    def append_definition(definition: dict[str, Any], path: str) -> None:
        name = definition["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate llm_definitions name '{name}' found at {path}.")
        seen_names.add(name)
        expanded.append(definition)

    def expand_variants(
        variants: Any,
        parent_name: str,
        inherited_state: dict[str, Any],
        path: str,
    ) -> None:
        if not isinstance(variants, list):
            raise ValueError(f"{path}.variants must be a list.")

        for index, variant in enumerate(variants):
            variant_path = f"{path}.variants[{index}]"
            if not isinstance(variant, dict):
                raise ValueError(f"{variant_path} must be a mapping.")

            _validate_entry_keys(variant, variant_allowed_keys, variant_path)

            if "name" in variant and "suffix" in variant:
                raise ValueError(f"{variant_path} cannot contain both 'name' and 'suffix'.")
            if "name" not in variant and "suffix" not in variant:
                raise ValueError(f"{variant_path} must contain either 'name' or 'suffix'.")

            if "name" in variant:
                variant_name = variant["name"]
                if not isinstance(variant_name, str) or not variant_name:
                    raise ValueError(f"{variant_path}.name must be a non-empty string.")
            else:
                suffix = variant["suffix"]
                if not isinstance(suffix, str) or not suffix.strip():
                    raise ValueError(f"{variant_path}.suffix must be a non-empty string.")
                normalized_suffix = suffix.strip().lstrip(":-_")
                if not normalized_suffix:
                    raise ValueError(
                        f"{variant_path}.suffix must contain non-separator characters."
                    )
                variant_name = f"{parent_name}:{normalized_suffix}"

            _validate_optional_mapping(variant.get("params"), "params", variant_path)
            _validate_optional_mapping(variant.get("pricing"), "pricing", variant_path)

            variant_state = {
                "model": variant.get("model", inherited_state.get("model")),
                "api_key_env": variant.get("api_key_env", inherited_state.get("api_key_env")),
                "api_mode": variant.get("api_mode", inherited_state.get("api_mode")),
                "params": _deep_merge_dict(inherited_state.get("params"), variant.get("params")),
                "pricing": _deep_merge_dict(inherited_state.get("pricing"), variant.get("pricing")),
            }

            append_definition(
                _build_llm_definition(variant_name, variant_state, variant_path), variant_path
            )

            nested_variants = variant.get("variants")
            if nested_variants is not None:
                expand_variants(nested_variants, variant_name, variant_state, variant_path)

    for index, raw_definition in enumerate(raw_definitions):
        path = f"llm_definitions[{index}]"
        if not isinstance(raw_definition, dict):
            raise ValueError(f"{path} must be a mapping.")

        has_base_name = "base_name" in raw_definition
        has_variants = "variants" in raw_definition

        if has_base_name or has_variants:
            if "name" in raw_definition:
                raise ValueError(f"{path} cannot contain 'name' when using base/variant syntax.")
            _validate_entry_keys(raw_definition, base_allowed_keys, path)

            base_name = raw_definition.get("base_name")
            if not isinstance(base_name, str) or not base_name:
                raise ValueError(f"{path}.base_name must be a non-empty string.")

            _validate_optional_mapping(raw_definition.get("params"), "params", path)
            _validate_optional_mapping(raw_definition.get("pricing"), "pricing", path)

            base_state = {
                "model": raw_definition.get("model"),
                "api_key_env": raw_definition.get("api_key_env"),
                "api_mode": raw_definition.get("api_mode"),
                "params": raw_definition.get("params"),
                "pricing": raw_definition.get("pricing"),
            }

            variants = raw_definition.get("variants")
            append_definition(_build_llm_definition(base_name, base_state, path), path)
            if variants is not None:
                expand_variants(variants, base_name, base_state, path)
            continue

        _validate_entry_keys(raw_definition, explicit_allowed_keys, path)
        _validate_optional_mapping(raw_definition.get("params"), "params", path)
        _validate_optional_mapping(raw_definition.get("pricing"), "pricing", path)

        definition = _build_llm_definition(raw_definition.get("name"), raw_definition, path)
        append_definition(definition, path)

    return expanded


def _normalize_raw_config(raw_config: Any) -> dict[str, Any]:
    if raw_config is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(raw_config, dict):
        raise ValueError("Top-level configuration must be a mapping.")

    normalized = dict(raw_config)
    if "llm_definitions" in normalized:
        normalized["llm_definitions"] = _expand_llm_definitions(normalized["llm_definitions"])
    _apply_mom_debug_model_if_enabled(normalized)
    return normalized


def _apply_mom_debug_model_if_enabled(normalized: dict[str, Any]) -> None:
    service = normalized.get("service")
    if not isinstance(service, dict) or not service.get("enable_mom_debug_model", False):
        return

    llm_definitions = normalized.get("llm_definitions")
    if not isinstance(llm_definitions, list) or not llm_definitions:
        raise ValueError("service.enable_mom_debug_model requires non-empty llm_definitions.")

    llm_names: list[str] = []
    for index, llm_definition in enumerate(llm_definitions):
        if not isinstance(llm_definition, dict):
            raise ValueError(f"llm_definitions[{index}] must be a mapping.")
        llm_name = llm_definition.get("name")
        if not isinstance(llm_name, str) or not llm_name:
            raise ValueError(f"llm_definitions[{index}] is missing required field 'name'.")
        llm_names.append(llm_name)

    debug_model = {
        "name": MOM_DEBUG_MODEL_NAME,
        "llms_to_query": llm_names,
        "concluding_llm": llm_names[0],
        "concluding_prompt": MOM_DEBUG_DEFAULT_PROMPT,
        "include_thinking_context": True,
    }

    models = normalized.get("models")
    if models is None:
        normalized["models"] = [debug_model]
        return
    if not isinstance(models, list):
        raise ValueError("models must be a list.")

    for index, model in enumerate(models):
        if isinstance(model, dict) and model.get("name") == MOM_DEBUG_MODEL_NAME:
            models[index] = debug_model
            return

    models.append(debug_model)


def _validate_model_references(config: MoMConfig) -> None:
    # Handle both dicts and LLMDefinition objects
    known_llm_names = set()
    for llm in config.llm_definitions:
        if isinstance(llm, dict):
            known_llm_names.add(llm["name"])
        else:
            known_llm_names.add(llm.name)

    errors: list[str] = []

    for model_cfg in config.models:
        if model_cfg.concluding_llm not in known_llm_names:
            errors.append(
                f"models[{model_cfg.name}].concluding_llm='{model_cfg.concluding_llm}' "
                "does not match any llm_definitions name."
            )

        missing_query_llms = [
            name for name in model_cfg.llms_to_query if name not in known_llm_names
        ]
        if missing_query_llms:
            missing_csv = ", ".join(missing_query_llms)
            errors.append(
                f"models[{model_cfg.name}].llms_to_query contains unknown llm_definitions: "
                f"{missing_csv}"
            )

    if errors:
        raise ValueError("Invalid model references: " + " | ".join(errors))


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
                normalized_raw = _normalize_raw_config(raw)
                config = MoMConfig(**normalized_raw)
                _validate_model_references(config)
                # Log the path that was successfully loaded
                # Use a simple print here as logger might not be configured yet
                print(f"--- Config loaded successfully from: {path} ---")
                return config
            except ValueError as e:
                raise RuntimeError(f"Invalid configuration in {path}: {e}") from e
            except ValidationError as e:
                raise RuntimeError(f"Invalid configuration in {path}: {e}") from e

    raise FileNotFoundError(f"config.yaml not found in any of the search paths: {search_paths}")
