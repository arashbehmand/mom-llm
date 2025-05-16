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
    # Try current working directory first, then fallback to mom_service/config.yaml
    search_paths = []
    if config_path:
        search_paths.append(config_path)
    else:
        search_paths.append(os.path.join(os.getcwd(), "config.yaml"))
        search_paths.append(os.path.join(os.path.dirname(__file__), "config.yaml"))
    for path in search_paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            try:
                config = MoMConfig(**raw)
            except ValidationError as e:
                raise RuntimeError(f"Invalid config.yaml: {e}")
            return config
    raise FileNotFoundError(f"config.yaml not found in any of: {search_paths}")
