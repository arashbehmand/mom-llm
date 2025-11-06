"""
Unit tests for mom_service.config module
"""

import pytest
import yaml
from pydantic import ValidationError

from mom_service.config import (
    LLMDefinition,
    ModelConfig,
    MoMConfig,
    PricingConfig,
    ServiceConfig,
    load_config,
)


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

        with pytest.raises(yaml.YAMLError):  # YAML parsing error
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


class TestPricingConfig:
    """Tests for PricingConfig model"""

    def test_pricing_config_with_all_costs(self):
        """Test PricingConfig with all cost fields specified"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.00003,
            completion_cost_per_token=0.00006,
            reasoning_cost_per_token=0.0001,
        )
        assert pricing.prompt_cost_per_token == 0.00003
        assert pricing.completion_cost_per_token == 0.00006
        assert pricing.reasoning_cost_per_token == 0.0001

    def test_pricing_config_optional_fields(self):
        """Test that pricing config fields are optional"""
        pricing = PricingConfig()
        assert pricing.prompt_cost_per_token is None
        assert pricing.completion_cost_per_token is None
        assert pricing.reasoning_cost_per_token is None

    def test_pricing_config_partial_specification(self):
        """Test PricingConfig with only some fields specified"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.00001,
            completion_cost_per_token=0.00002,
        )
        assert pricing.prompt_cost_per_token == 0.00001
        assert pricing.completion_cost_per_token == 0.00002
        assert pricing.reasoning_cost_per_token is None

    def test_calculate_cost_standard_completion(self):
        """Test cost calculation for standard completion (no reasoning)"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.00003,
            completion_cost_per_token=0.00006,
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # 1000 * 0.00003 + 500 * 0.00006 = 0.03 + 0.03 = 0.06
        assert total_cost == pytest.approx(0.06, rel=1e-6)
        assert cost_breakdown == {"input": pytest.approx(0.03), "output": pytest.approx(0.03)}

    def test_calculate_cost_with_reasoning_breakdown(self):
        """Test cost calculation with reasoning token breakdown"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.15 / 1_000_000,  # $0.15/1M
            completion_cost_per_token=0.60 / 1_000_000,  # $0.60/1M
            reasoning_cost_per_token=3.50 / 1_000_000,  # $3.50/1M
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=825,
            reasoning_tokens=764,
            text_tokens=61,
        )

        # Input: 1000 * 0.15/1M = 0.00015
        # Text: 61 * 0.60/1M = 0.0000366
        # Reasoning: 764 * 3.50/1M = 0.002674
        assert total_cost == pytest.approx(0.0028606, rel=1e-4)
        assert "input" in cost_breakdown
        assert "output_text" in cost_breakdown
        assert "output_reasoning" in cost_breakdown

    def test_calculate_cost_with_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001,
            completion_cost_per_token=0.002,
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=0,
            completion_tokens=0,
        )

        assert total_cost == 0.0
        assert cost_breakdown == {"input": 0.0, "output": 0.0}

    def test_calculate_cost_only_reasoning_tokens(self):
        """Test cost calculation when all output is reasoning tokens"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
            reasoning_cost_per_token=0.1 / 1_000_000,
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=100,
            completion_tokens=1000,
            reasoning_tokens=1000,
            text_tokens=0,
        )

        # All output cost should come from reasoning
        assert total_cost > 0
        assert cost_breakdown["output_text"] == 0.0
        assert cost_breakdown["output_reasoning"] > 0

    def test_calculate_cost_without_reasoning_pricing(self):
        """Test that standard pricing is used when no reasoning pricing configured"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001,
            completion_cost_per_token=0.002,
            # No reasoning_cost_per_token
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=100,
            completion_tokens=200,
            reasoning_tokens=50,  # These will be ignored
            text_tokens=150,
        )

        # Should use standard completion pricing for all output tokens
        assert total_cost == pytest.approx(0.5, rel=1e-6)  # 100*0.001 + 200*0.002
        assert "input" in cost_breakdown
        assert "output" in cost_breakdown
        assert "output_text" not in cost_breakdown  # No reasoning breakdown

    def test_calculate_cost_with_none_costs(self):
        """Test cost calculation with None values"""
        pricing = PricingConfig(
            prompt_cost_per_token=None,
            completion_cost_per_token=None,
        )

        total_cost, cost_breakdown = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert total_cost == 0.0
        assert cost_breakdown == {"input": 0.0, "output": 0.0}

    def test_llm_definition_with_pricing(self):
        """Test LLMDefinition with pricing configuration"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.00003,
            completion_cost_per_token=0.00006,
        )

        llm_def = LLMDefinition(
            name="test-llm",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
            pricing=pricing,
        )

        assert llm_def.pricing is not None
        assert llm_def.pricing.prompt_cost_per_token == 0.00003

    def test_llm_definition_without_pricing(self):
        """Test LLMDefinition without pricing (should default to None)"""
        llm_def = LLMDefinition(
            name="test-llm",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
        )

        assert llm_def.pricing is None

    def test_pricing_config_from_yaml(self, tmp_path):
        """Test loading PricingConfig from YAML"""
        config_content = """
llm_definitions:
  - name: test-gpt4
    model: gpt-4
    api_key_env: OPENAI_API_KEY
    pricing:
      prompt_cost_per_token: 0.00003
      completion_cost_per_token: 0.00006
      reasoning_cost_per_token: 0.0001

models:
  - name: test-model
    llms_to_query:
      - test-gpt4
    concluding_llm: test-gpt4

service:
  timeout_seconds: 30
"""
        config_file = tmp_path / "config_with_pricing.yaml"
        config_file.write_text(config_content)

        config = load_config(config_path=str(config_file))

        assert config.llm_definitions[0].pricing is not None
        assert config.llm_definitions[0].pricing.prompt_cost_per_token == 0.00003
        assert config.llm_definitions[0].pricing.completion_cost_per_token == 0.00006
        assert config.llm_definitions[0].pricing.reasoning_cost_per_token == 0.0001
