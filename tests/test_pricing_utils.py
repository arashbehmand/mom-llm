"""
Unit tests for mom_service.pricing_utils module

Tests the normalized token calculation functionality that converts actual costs
to normalized tokens using a unified pricing model ($1/1M input, $10/1M output).
"""

import pytest

from mom_service.pricing_utils import (
    UNIFIED_INPUT_COST_PER_TOKEN,
    UNIFIED_OUTPUT_COST_PER_TOKEN,
    calculate_cost_from_normalized_tokens,
    calculate_normalized_tokens,
)


class TestCalculateNormalizedTokens:
    """Tests for calculate_normalized_tokens function"""

    def test_normalized_tokens_with_typical_usage(self):
        """Test normalized token calculation with typical token counts"""
        # Simulate a call with actual cost of $0.05
        # With 1000 input tokens and 500 output tokens
        actual_cost = 0.05
        actual_input = 1000
        actual_output = 500

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        # Verify the normalized tokens produce the correct cost
        calculated_cost = (
            normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
            + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
        )

        # Allow tolerance for integer rounding (implementation uses int())
        assert abs(calculated_cost - actual_cost) < 0.00001  # Within 0.01 cent tolerance
        assert normalized_input > 0
        assert normalized_output > 0

        # Verify ratio is preserved (approximately)
        original_ratio = actual_input / (actual_input + actual_output)
        normalized_ratio = normalized_input / (normalized_input + normalized_output)
        assert abs(original_ratio - normalized_ratio) < 0.1  # Within 10%

    def test_normalized_tokens_with_zero_cost(self):
        """Test that zero cost returns actual tokens unchanged"""
        actual_cost = 0.0
        actual_input = 100
        actual_output = 200

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        assert normalized_input == actual_input
        assert normalized_output == actual_output

    def test_normalized_tokens_with_zero_tokens_but_positive_cost(self):
        """Test handling when there are no tokens but there's a cost"""
        actual_cost = 0.01
        actual_input = 0
        actual_output = 0

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        # Cost should be attributed to output tokens (more expensive)
        assert normalized_input == 0
        assert normalized_output > 0

        # Verify cost matches (allow rounding tolerance)
        calculated_cost = normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
        assert abs(calculated_cost - actual_cost) < 0.00001

    def test_normalized_tokens_only_input_tokens(self):
        """Test with only input tokens (no output)"""
        actual_cost = 0.001
        actual_input = 1000
        actual_output = 0

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        assert normalized_input > 0
        assert normalized_output == 0

        # Verify cost (allow rounding tolerance)
        calculated_cost = normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
        assert abs(calculated_cost - actual_cost) < 0.00001

    def test_normalized_tokens_only_output_tokens(self):
        """Test with only output tokens (no input)"""
        actual_cost = 0.01
        actual_input = 0
        actual_output = 1000

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        assert normalized_input == 0
        assert normalized_output > 0

        # Verify cost (allow rounding tolerance)
        calculated_cost = normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
        assert abs(calculated_cost - actual_cost) < 0.00001

    def test_normalized_tokens_preserves_cost_with_different_ratios(self):
        """Test that cost is preserved across different input/output ratios"""
        test_cases = [
            # (actual_cost, actual_input, actual_output)
            (0.05, 5000, 1000),  # Heavy input
            (0.05, 1000, 5000),  # Heavy output
            (0.05, 3000, 3000),  # Balanced
            (0.001, 100, 50),  # Small values
            (1.0, 100000, 50000),  # Large values
        ]

        for actual_cost, actual_input, actual_output in test_cases:
            normalized_input, normalized_output = calculate_normalized_tokens(
                actual_cost, actual_input, actual_output
            )

            calculated_cost = (
                normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
                + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
            )

            # Allow rounding tolerance from integer conversion
            assert abs(calculated_cost - actual_cost) < 0.00001, (
                f"Cost mismatch for case ({actual_cost}, {actual_input}, {actual_output}): "
                f"expected {actual_cost}, got {calculated_cost}"
            )

    def test_normalized_tokens_with_negative_cost(self):
        """Test handling of negative cost (edge case)"""
        actual_cost = -0.01
        actual_input = 100
        actual_output = 200

        # Negative cost should return actual tokens
        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        assert normalized_input == actual_input
        assert normalized_output == actual_output

    def test_normalized_tokens_reasoning_heavy_scenario(self):
        """Test scenario simulating reasoning tokens (expensive)"""
        # Simulate Gemini 2.5 Flash pricing:
        # Input: $0.15/1M, Text output: $0.60/1M, Reasoning: $3.50/1M
        # Actual call: 1000 input, 61 text output, 764 reasoning tokens
        # Expected cost: 1000*0.15 + 61*0.60 + 764*3.50 = 0.15 + 0.0366 + 2.674 = $2.8606/1M = $0.0028606

        actual_cost = 0.0028606
        actual_input = 1000
        actual_output = 825  # 61 text + 764 reasoning

        normalized_input, normalized_output = calculate_normalized_tokens(
            actual_cost, actual_input, actual_output
        )

        calculated_cost = (
            normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
            + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
        )

        # Allow rounding tolerance from integer conversion
        assert abs(calculated_cost - actual_cost) < 0.00001
        assert normalized_input > 0
        assert normalized_output > 0


class TestCalculateCostFromNormalizedTokens:
    """Tests for calculate_cost_from_normalized_tokens function"""

    def test_cost_calculation_with_typical_tokens(self):
        """Test cost calculation with typical token counts"""
        normalized_input = 1000
        normalized_output = 500

        expected_cost = (
            normalized_input * UNIFIED_INPUT_COST_PER_TOKEN
            + normalized_output * UNIFIED_OUTPUT_COST_PER_TOKEN
        )

        cost = calculate_cost_from_normalized_tokens(normalized_input, normalized_output)

        assert cost == expected_cost
        assert cost == pytest.approx(0.006, rel=1e-6)  # 1000*0.000001 + 500*0.00001

    def test_cost_calculation_with_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        cost = calculate_cost_from_normalized_tokens(0, 0)
        assert cost == 0.0

    def test_cost_calculation_only_input(self):
        """Test cost calculation with only input tokens"""
        cost = calculate_cost_from_normalized_tokens(1000000, 0)
        assert cost == pytest.approx(1.0, rel=1e-6)  # $1 per 1M tokens

    def test_cost_calculation_only_output(self):
        """Test cost calculation with only output tokens"""
        cost = calculate_cost_from_normalized_tokens(0, 1000000)
        assert cost == pytest.approx(10.0, rel=1e-6)  # $10 per 1M tokens

    def test_unified_pricing_model(self):
        """Verify the unified pricing constants are correct"""
        assert pytest.approx(0.000001, rel=1e-9) == UNIFIED_INPUT_COST_PER_TOKEN  # $1/1M
        assert pytest.approx(0.00001, rel=1e-9) == UNIFIED_OUTPUT_COST_PER_TOKEN  # $10/1M
        assert (
            pytest.approx(10 * UNIFIED_INPUT_COST_PER_TOKEN, rel=1e-9)
            == UNIFIED_OUTPUT_COST_PER_TOKEN
        )


class TestRoundTripConversion:
    """Test that cost -> normalized tokens -> cost works correctly"""

    def test_round_trip_preserves_cost(self):
        """Test that converting cost to tokens and back preserves the cost"""
        original_cost = 0.12345
        actual_input = 5000
        actual_output = 3000

        # Convert cost to normalized tokens
        normalized_input, normalized_output = calculate_normalized_tokens(
            original_cost, actual_input, actual_output
        )

        # Convert back to cost
        calculated_cost = calculate_cost_from_normalized_tokens(normalized_input, normalized_output)

        # Should match within tolerance (allow rounding from int conversion)
        assert abs(calculated_cost - original_cost) < 0.00001

    def test_round_trip_with_various_costs(self):
        """Test round-trip with various cost values"""
        test_costs = [0.001, 0.01, 0.1, 1.0, 10.0]

        for original_cost in test_costs:
            normalized_input, normalized_output = calculate_normalized_tokens(
                original_cost, 1000, 1000
            )

            calculated_cost = calculate_cost_from_normalized_tokens(
                normalized_input, normalized_output
            )

            # Allow rounding tolerance from integer conversion
            # Use relative tolerance of 2% for better accuracy
            tolerance = max(0.00002, original_cost * 0.02)
            assert (
                abs(calculated_cost - original_cost) < tolerance
            ), f"Round-trip failed for cost {original_cost}: got {calculated_cost}"
