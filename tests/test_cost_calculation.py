"""
Unit tests for mom_service.cost_calculation module

Tests detailed cost calculations with support for reasoning tokens,
custom pricing configurations, and LiteLLM fallback pricing.
"""

from unittest.mock import patch

import pytest

from mom_service.config import PricingConfig
from mom_service.cost_calculation import calculate_detailed_cost


class TestCalculateDetailedCostWithCustomPricing:
    """Tests for calculate_detailed_cost with custom pricing config"""

    def test_custom_pricing_without_reasoning_tokens(self):
        """Test custom pricing for standard completion (no reasoning breakdown)"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.00003,  # $30/1M tokens
            completion_cost_per_token=0.00006,  # $60/1M tokens
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            pricing_config=pricing,
        )

        # Expected: 1000 * 0.00003 + 500 * 0.00006 = 0.03 + 0.03 = 0.06
        assert total_cost == pytest.approx(0.06, rel=1e-6)
        assert cost_breakdown == {"input": pytest.approx(0.03), "output": pytest.approx(0.03)}

    def test_custom_pricing_with_reasoning_tokens(self):
        """Test custom pricing with reasoning token breakdown"""
        # Simulating Gemini 2.5 Flash pricing
        pricing = PricingConfig(
            prompt_cost_per_token=0.15 / 1_000_000,  # $0.15/1M
            completion_cost_per_token=0.60 / 1_000_000,  # $0.60/1M for text
            reasoning_cost_per_token=3.50 / 1_000_000,  # $3.50/1M for reasoning
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="gemini/gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=825,
            completion_tokens_details={"reasoning_tokens": 764, "text_tokens": 61},
            pricing_config=pricing,
        )

        # Expected:
        # Input: 1000 * (0.15/1M) = 0.00015
        # Text: 61 * (0.60/1M) = 0.0000366
        # Reasoning: 764 * (3.50/1M) = 0.002674
        # Total: 0.0028606
        assert total_cost == pytest.approx(0.0028606, rel=1e-4)
        assert "input" in cost_breakdown
        assert "output_text" in cost_breakdown
        assert "output_reasoning" in cost_breakdown
        assert cost_breakdown["input"] == pytest.approx(0.00015, rel=1e-6)
        assert cost_breakdown["output_text"] == pytest.approx(0.0000366, rel=1e-6)
        assert cost_breakdown["output_reasoning"] == pytest.approx(0.002674, rel=1e-4)

    def test_custom_pricing_with_zero_text_tokens(self):
        """Test custom pricing when all output is reasoning tokens"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
            reasoning_cost_per_token=0.1 / 1_000_000,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=100,
            completion_tokens=1000,
            completion_tokens_details={"reasoning_tokens": 1000, "text_tokens": 0},
            pricing_config=pricing,
        )

        # All output cost should be from reasoning tokens
        assert total_cost > 0
        assert cost_breakdown["output_text"] == 0.0
        assert cost_breakdown["output_reasoning"] > 0

    def test_custom_pricing_token_mismatch_warning(self):
        """Test that token count mismatch generates a warning"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
            reasoning_cost_per_token=0.1 / 1_000_000,
        )

        with patch("mom_service.cost_calculation.logger.warning") as mock_warning:
            # Provide mismatched token counts
            calculate_detailed_cost(
                model_name="test-model",
                prompt_tokens=100,
                completion_tokens=1000,
                completion_tokens_details={
                    "reasoning_tokens": 500,
                    "text_tokens": 400,  # 500 + 400 = 900 != 1000
                },
                pricing_config=pricing,
            )

            # Should log a warning about mismatch
            assert mock_warning.called
            warning_msg = mock_warning.call_args[0][0]
            assert "mismatch" in warning_msg.lower()

    def test_custom_pricing_with_none_values(self):
        """Test custom pricing with None values returns 0 cost"""
        # Create a pricing config with None values
        pricing = PricingConfig(
            prompt_cost_per_token=None,
            completion_cost_per_token=None,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            pricing_config=pricing,
        )

        # With None costs, calculate_cost returns 0 (valid result)
        assert total_cost == 0.0
        assert cost_breakdown == {"input": 0.0, "output": 0.0}


class TestCalculateDetailedCostWithLiteLLMPricing:
    """Tests for calculate_detailed_cost using LiteLLM's pricing database"""

    def test_litellm_pricing_standard_model(self):
        """Test cost calculation using LiteLLM's pricing database"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            # Mock LiteLLM returning costs for GPT-4
            mock_litellm_cost.return_value = (0.03, 0.06)  # $30/1M input, $60/1M output

            total_cost, cost_breakdown = calculate_detailed_cost(
                model_name="gpt-4",
                prompt_tokens=1000,
                completion_tokens=500,
                pricing_config=None,  # Use LiteLLM pricing
            )

            assert total_cost == 0.09  # 0.03 + 0.06
            assert cost_breakdown == {"input": 0.03, "output": 0.06}
            mock_litellm_cost.assert_called_once_with(
                model="gpt-4", prompt_tokens=1000, completion_tokens=500
            )

    def test_litellm_pricing_with_reasoning_tokens_warning(self):
        """Test that reasoning tokens without custom pricing generates warning"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            mock_litellm_cost.return_value = (0.01, 0.02)

            with patch("mom_service.cost_calculation.logger.warning") as mock_warning:
                calculate_detailed_cost(
                    model_name="gemini/gemini-2.5-flash",
                    prompt_tokens=1000,
                    completion_tokens=825,
                    completion_tokens_details={"reasoning_tokens": 764, "text_tokens": 61},
                    pricing_config=None,  # No custom pricing
                )

                # Should warn about reasoning tokens without custom pricing
                assert mock_warning.called
                warning_msg = mock_warning.call_args[0][0]
                assert "reasoning tokens" in warning_msg.lower()
                assert "custom pricing" in warning_msg.lower()

    def test_litellm_pricing_returns_none(self):
        """Test handling when LiteLLM returns None for costs"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            # LiteLLM returns None when pricing is not available
            mock_litellm_cost.return_value = (None, None)

            total_cost, cost_breakdown = calculate_detailed_cost(
                model_name="unknown-model",
                prompt_tokens=1000,
                completion_tokens=500,
                pricing_config=None,
            )

            # When LiteLLM returns None which causes an error, empty dict is returned
            assert total_cost == 0.0
            # The implementation returns empty dict on error
            assert cost_breakdown == {}

    def test_litellm_pricing_partial_none(self):
        """Test handling when LiteLLM returns partial None"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            # One cost is available, one is None - this will also cause an error in formatting
            mock_litellm_cost.return_value = (0.01, None)

            total_cost, cost_breakdown = calculate_detailed_cost(
                model_name="test-model",
                prompt_tokens=1000,
                completion_tokens=500,
                pricing_config=None,
            )

            # Error in handling None will cause fallback to $0 cost
            assert total_cost == 0.0
            assert cost_breakdown == {}


class TestCalculateDetailedCostErrorHandling:
    """Tests for error handling in calculate_detailed_cost"""

    def test_no_pricing_available(self):
        """Test when no pricing is available (neither custom nor LiteLLM)"""
        with patch(
            "mom_service.cost_calculation.litellm.cost_per_token",
            side_effect=Exception("No pricing available"),
        ):
            with patch("mom_service.cost_calculation.logger.warning") as mock_warning:
                total_cost, cost_breakdown = calculate_detailed_cost(
                    model_name="unknown-model",
                    prompt_tokens=1000,
                    completion_tokens=500,
                    pricing_config=None,
                )

                # Should return $0 cost
                assert total_cost == 0.0
                assert cost_breakdown == {}

                # Should log warning
                assert mock_warning.called
                warning_msg = mock_warning.call_args[0][0]
                assert "failed" in warning_msg.lower()

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=0,
            completion_tokens=0,
            pricing_config=pricing,
        )

        assert total_cost == 0.0
        assert cost_breakdown == {"input": 0.0, "output": 0.0}

    def test_none_completion_tokens_details(self):
        """Test that None completion_tokens_details is handled gracefully"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            completion_tokens_details=None,  # Explicitly None
            pricing_config=pricing,
        )

        # Should calculate cost without reasoning breakdown
        assert total_cost > 0
        assert "input" in cost_breakdown
        assert "output" in cost_breakdown

    def test_empty_completion_tokens_details(self):
        """Test that empty completion_tokens_details dict is handled"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            completion_tokens_details={},  # Empty dict
            pricing_config=pricing,
        )

        # Should calculate cost without reasoning breakdown
        assert total_cost > 0
        assert "input" in cost_breakdown
        assert "output" in cost_breakdown


class TestCalculateDetailedCostRealWorldScenarios:
    """Real-world scenario tests"""

    def test_openai_gpt4_typical_call(self):
        """Test typical OpenAI GPT-4 call"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            # GPT-4 pricing: $30/1M input, $60/1M output
            mock_litellm_cost.return_value = (0.03, 0.06)

            total_cost, _ = calculate_detailed_cost(
                model_name="gpt-4",
                prompt_tokens=1500,
                completion_tokens=800,
            )

            # Mock returns costs per request, not per token
            # 0.03 for all input tokens + 0.06 for all output tokens = 0.09
            assert total_cost == pytest.approx(0.09, rel=1e-6)

    def test_gemini_flash_with_reasoning(self):
        """Test Gemini 2.5 Flash with reasoning tokens"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.15 / 1_000_000,
            completion_cost_per_token=0.60 / 1_000_000,
            reasoning_cost_per_token=3.50 / 1_000_000,
        )

        total_cost, cost_breakdown = calculate_detailed_cost(
            model_name="gemini/gemini-2.5-flash",
            prompt_tokens=2000,
            completion_tokens=1500,
            completion_tokens_details={"reasoning_tokens": 1200, "text_tokens": 300},
            pricing_config=pricing,
        )

        # Input: 2000 * 0.15/1M = 0.0003
        # Text: 300 * 0.60/1M = 0.00018
        # Reasoning: 1200 * 3.50/1M = 0.0042
        # Total: 0.00468
        assert total_cost == pytest.approx(0.00468, rel=1e-4)
        assert cost_breakdown["input"] == pytest.approx(0.0003, rel=1e-6)
        assert cost_breakdown["output_text"] == pytest.approx(0.00018, rel=1e-6)
        assert cost_breakdown["output_reasoning"] == pytest.approx(0.0042, rel=1e-4)

    def test_anthropic_claude_typical_call(self):
        """Test typical Anthropic Claude call"""
        with patch("mom_service.cost_calculation.litellm.cost_per_token") as mock_litellm_cost:
            # Claude pricing varies, but let's use typical values
            mock_litellm_cost.return_value = (0.024, 0.072)

            total_cost, _ = calculate_detailed_cost(
                model_name="claude-3-opus",
                prompt_tokens=2500,
                completion_tokens=1000,
            )

            # Mock returns costs per request: 0.024 + 0.072 = 0.096
            assert total_cost == pytest.approx(0.096, rel=1e-6)

    def test_very_large_token_counts(self):
        """Test with very large token counts"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
        )

        total_cost, _ = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=10_000_000,  # 10M tokens
            completion_tokens=5_000_000,  # 5M tokens
            pricing_config=pricing,
        )

        # 10M * (0.001/1M) + 5M * (0.01/1M) = 10*0.001 + 5*0.01 = 0.01 + 0.05 = 0.06
        assert total_cost == pytest.approx(0.06, rel=1e-6)

    def test_very_small_token_counts(self):
        """Test with very small token counts"""
        pricing = PricingConfig(
            prompt_cost_per_token=0.001 / 1_000_000,
            completion_cost_per_token=0.01 / 1_000_000,
        )

        total_cost, _ = calculate_detailed_cost(
            model_name="test-model",
            prompt_tokens=1,
            completion_tokens=1,
            pricing_config=pricing,
        )

        # Should still calculate correctly even with tiny values
        assert total_cost > 0
        assert total_cost < 0.000001  # Very small cost
