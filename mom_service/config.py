import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ValidationError


class LLMDefinition(BaseModel):
    name: str  # Unique identifier for this LLM definition
    model: str
    api_key_env: str
    params: Optional[Dict[str, Any]] = None


class PromptDefinition(BaseModel):
    name: str
    content: str


class ModelConfig(BaseModel):
    name: str
    llms_to_query: List[str]
    concluding_llm: str
    concluding_prompt: Optional[str] = None  # Name of the PromptDefinition to use
    include_thinking_context: bool = False  # Default to false if not specified


class ServiceConfig(BaseModel):
    timeout_seconds: int = 30
    exposed_apis: List[str] = ["openai"]  # Default to only openai if not specified
    cache_enabled: bool = False # Add cache enabled flag
    max_retries: int = 3 # Add max retries for LLM calls
    retry_delay_seconds: int = 5 # Add delay between retries


class LangfuseConfig(BaseModel):
    public_key_env: str
    secret_key_env: str
    host_env: str


class MoMConfig(BaseModel):
    llm_definitions: List[LLMDefinition]
    prompt_definitions: Optional[List[PromptDefinition]] = None
    models: List[ModelConfig]
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
            os.path.join(os.path.dirname(__file__), "..", "config.yaml"), # Adjusted for being in mom_service/
        ]

    for path in search_paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
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
