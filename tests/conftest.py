"""
Pytest configuration and shared fixtures for MoM Service tests
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_langfuse_client():
    """Fixture providing a mocked Langfuse client"""
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_generation = MagicMock()

    mock_client.trace.return_value = mock_trace
    mock_trace.generation.return_value = mock_generation

    return mock_client


@pytest.fixture
def sample_llm_definition():
    """Fixture providing a sample LLM definition"""
    from mom_service.config import LLMDefinition

    return LLMDefinition(
        name="test-llm",
        model="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        params={"temperature": 0.7},
    )


@pytest.fixture
def sample_mom_config():
    """Fixture providing a sample MoM configuration"""
    from mom_service.config import LLMDefinition, ModelConfig, MoMConfig, ServiceConfig

    return MoMConfig(
        llm_definitions=[
            LLMDefinition(name="gpt4", model="gpt-4", api_key_env="OPENAI_API_KEY", params={}),
            LLMDefinition(
                name="gpt35", model="gpt-3.5-turbo", api_key_env="OPENAI_API_KEY", params={}
            ),
        ],
        models=[
            ModelConfig(
                name="test-mom-model",
                llms_to_query=["gpt4", "gpt35"],
                concluding_llm="gpt4",
                include_thinking_context=True,
            )
        ],
        service=ServiceConfig(timeout_seconds=30, cache_enabled=False, max_llm_retries=2),
    )


@pytest.fixture
def sample_request_messages():
    """Fixture providing sample chat messages"""
    return [{"role": "user", "content": "What is the capital of France?"}]


@pytest.fixture
def mock_litellm_response():
    """Fixture providing a mocked LiteLLM response"""
    from litellm.utils import Choices, Message, ModelResponse, Usage

    return ModelResponse(
        id="test-response-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Paris is the capital of France.", role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        object="chat.completion",
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture creating a temporary config file"""
    config_content = """
llm_definitions:
  - name: test-gpt4
    model: gpt-4
    api_key_env: OPENAI_API_KEY
    params:
      temperature: 0.7

models:
  - name: test-model
    llms_to_query:
      - test-gpt4
    concluding_llm: test-gpt4
    include_thinking_context: false

service:
  timeout_seconds: 30
  cache_enabled: false
  max_llm_retries: 2
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture setting mock environment variables"""
    monkeypatch.setenv("API_TOKEN", "test-token-123")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ALLOWED_CORS_ORIGINS", "*")
    return {
        "API_TOKEN": "test-token-123",
        "OPENAI_API_KEY": "test-openai-key",
    }


@pytest.fixture
def temp_metrics_db(tmp_path):
    """Fixture creating a temporary metrics database"""
    db_path = tmp_path / "test_metrics.db"
    # Patch the METRICS_DB_PATH before importing metrics_db
    from mom_service import metrics_db

    original_path = metrics_db.METRICS_DB_PATH
    metrics_db.METRICS_DB_PATH = str(db_path)
    metrics_db.init_metrics_db()

    yield str(db_path)

    # Cleanup
    metrics_db.METRICS_DB_PATH = original_path
