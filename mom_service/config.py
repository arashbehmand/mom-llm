import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, ValidationError

from typing import List, Optional, Dict, Any # Added Dict, Any

class LLMConfig(BaseModel):
    name: str
    model: str
    api_key_env: str
    params: Optional[Dict[str, Any]] = None # Added optional params field

class ConcludingLLMConfig(BaseModel):
    name: str
    model: str
    api_key_env: str
    params: Optional[Dict[str, Any]] = None # Added optional params field

class ServiceConfig(BaseModel):
    timeout_seconds: int = 30

class LangfuseConfig(BaseModel):
    public_key_env: str
    secret_key_env: str
    host_env: str

class MoMConfig(BaseModel):
    llms_to_query: List[LLMConfig]
    concluding_llm: ConcludingLLMConfig
    service: ServiceConfig
    concluding_llm_user_prompt: Optional[str] = None # Renamed from consolidation_system_prompt
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
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
            try:
                config = MoMConfig(**raw)
            except ValidationError as e:
                raise RuntimeError(f"Invalid config.yaml: {e}")
            return config
    raise FileNotFoundError(f"config.yaml not found in any of: {search_paths}")
