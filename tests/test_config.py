"""
Unit tests for mom_service.config module
"""

import pytest
from pydantic import ValidationError

from mom_service.config import LLMDefinition, ModelConfig, MoMConfig, ServiceConfig, load_config


class TestLLMDefinition:
    """Tests for LLMDefinition model"""

    def test_valid_llm_definition(self):
        """Test creating a valid LLM definition"""
        llm_def = LLMDefinition(
            name="test-llm",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
            params={"temperature": 0.7},
        )
        assert llm_def.name == "test-llm"
        assert llm_def.model == "gpt-4"
        assert llm_def.api_key_env == "OPENAI_API_KEY"
        assert llm_def.params == {"temperature": 0.7}

    def test_llm_definition_without_params(self):
        """Test LLM definition with no params (should default to None)"""
        llm_def = LLMDefinition(name="test-llm", model="gpt-4", api_key_env="OPENAI_API_KEY")
        assert llm_def.params is None

    def test_invalid_llm_definition_missing_required_field(self):
        """Test that missing required fields raise ValidationError"""
        with pytest.raises(ValidationError):
            LLMDefinition(name="test-llm", model="gpt-4")


class TestModelConfig:
    """Tests for ModelConfig model"""

    def test_valid_model_config(self):
        """Test creating a valid model configuration"""
        model_config = ModelConfig(
            name="test-model",
            llms_to_query=["gpt4", "gpt35"],
            concluding_llm="gpt4",
            concluding_prompt="synthesis-prompt",
            include_thinking_context=True,
        )
        assert model_config.name == "test-model"
        assert model_config.llms_to_query == ["gpt4", "gpt35"]
        assert model_config.concluding_llm == "gpt4"
        assert model_config.include_thinking_context is True

    def test_model_config_defaults(self):
        """Test default values for optional fields"""
        model_config = ModelConfig(name="test-model", llms_to_query=["gpt4"], concluding_llm="gpt4")
        assert model_config.concluding_prompt is None
        assert model_config.include_thinking_context is False


class TestServiceConfig:
    """Tests for ServiceConfig model"""

    def test_service_config_defaults(self):
        """Test default values for service configuration"""
        service_config = ServiceConfig()
        assert service_config.timeout_seconds == 30
        assert service_config.exposed_apis == ["openai"]
        assert service_config.cache_enabled is False
        assert service_config.max_llm_retries == 3
        assert service_config.llm_retry_delay_seconds == 2

    def test_service_config_custom_values(self):
        """Test custom service configuration values"""
        service_config = ServiceConfig(timeout_seconds=60, cache_enabled=True, max_llm_retries=5)
        assert service_config.timeout_seconds == 60
        assert service_config.cache_enabled is True
        assert service_config.max_llm_retries == 5


class TestMoMConfig:
    """Tests for MoMConfig model"""

    def test_valid_mom_config(self, sample_mom_config):
        """Test creating a valid MoM configuration"""
        assert len(sample_mom_config.llm_definitions) == 2
        assert len(sample_mom_config.models) == 1
        assert sample_mom_config.service.timeout_seconds == 30

    def test_mom_config_missing_required_fields(self):
        """Test that missing required fields raise ValidationError"""
        with pytest.raises(ValidationError):
            MoMConfig(llm_definitions=[], models=[])


class TestLoadConfig:
    """Tests for load_config function"""

    def test_load_config_from_valid_file(self, temp_config_file):
        """Test loading configuration from a valid YAML file"""
        config = load_config(config_path=temp_config_file)
        assert config is not None
        assert len(config.llm_definitions) == 1
        assert len(config.models) == 1
        assert config.llm_definitions[0].name == "test-gpt4"

    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config file"""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test that RuntimeError is raised for invalid YAML"""
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parsing error
            load_config(config_path=str(invalid_config))

    def test_load_config_invalid_schema(self, tmp_path):
        """Test that RuntimeError is raised for invalid schema"""
        invalid_config = tmp_path / "invalid_schema.yaml"
        invalid_config.write_text(
            """
llm_definitions:
  - name: test
    # Missing required fields
models: []
service:
  timeout_seconds: 30
"""
        )

        with pytest.raises(RuntimeError) as exc_info:
            load_config(config_path=str(invalid_config))
        assert "Invalid configuration" in str(exc_info.value)

    def test_load_config_from_env_variable(self, temp_config_file, monkeypatch):
        """Test loading configuration from MOM_CONFIG_PATH environment variable"""
        monkeypatch.setenv("MOM_CONFIG_PATH", temp_config_file)
        config = load_config()
        assert config is not None
        assert len(config.llm_definitions) == 1
