"""
Unit tests for mom_service.core_logic module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mom_service.core_logic import (
    _calculate_and_log_costs,
    _prepare_concluding_messages,
)
from mom_service.endpoints.models import ThinkingContextItem, UsageInfo


class TestCalculateAndLogCosts:
    """Tests for _calculate_and_log_costs function"""

    def test_calculate_costs_with_all_items(self):
        """Test cost calculation with fanout and concluding costs"""
        fanout_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response 1",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Response 2",
                usage=UsageInfo(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.0005)
            ),
        ]
        concluding_usage = UsageInfo(
            prompt_tokens=50, completion_tokens=100, total_tokens=150, cost=0.005
        )

        total_cost = _calculate_and_log_costs(fanout_context, concluding_usage)
        # 0.001 + 0.0005 + 0.005 = 0.0065
        assert total_cost == 0.0065

    def test_calculate_costs_with_none_costs(self):
        """Test cost calculation when some costs are None"""
        fanout_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response 1",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=None)
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Response 2",
                usage=UsageInfo(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.001)
            ),
        ]
        concluding_usage = UsageInfo(
            prompt_tokens=50, completion_tokens=100, total_tokens=150, cost=None
        )

        total_cost = _calculate_and_log_costs(fanout_context, concluding_usage)
        assert total_cost == 0.001

    def test_calculate_costs_with_zero_costs(self):
        """Test cost calculation with zero costs (e.g., cached responses)"""
        fanout_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response 1",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.0)
            ),
        ]
        concluding_usage = UsageInfo(
            prompt_tokens=50, completion_tokens=100, total_tokens=150, cost=0.0
        )

        total_cost = _calculate_and_log_costs(fanout_context, concluding_usage)
        assert total_cost == 0.0

    def test_calculate_costs_empty_fanout_context(self):
        """Test cost calculation with empty fanout context"""
        fanout_context = []
        concluding_usage = UsageInfo(
            prompt_tokens=50, completion_tokens=100, total_tokens=150, cost=0.005
        )

        total_cost = _calculate_and_log_costs(fanout_context, concluding_usage)
        assert total_cost == 0.005

    def test_calculate_costs_no_concluding_usage(self):
        """Test cost calculation without concluding usage"""
        fanout_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response 1",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.002)
            ),
        ]

        total_cost = _calculate_and_log_costs(fanout_context, None)
        assert total_cost == 0.002


class TestPrepareConcludingMessages:
    """Tests for _prepare_concluding_messages function"""

    def test_prepare_messages_with_successful_fanout(self, sample_mom_config):
        """Test message preparation with successful fanout responses"""
        request_messages = [
            {"role": "user", "content": "What is AI?"}
        ]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="AI stands for Artificial Intelligence.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="AI is a branch of computer science.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Should have: original message + context separator + 2 assistant messages
        assert len(result) >= 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is AI?"
        assert "llm responses" in result[1]["content"]
        assert result[2]["role"] == "assistant"

    def test_prepare_messages_with_failed_fanout(self, sample_mom_config):
        """Test message preparation when all fanout calls failed"""
        request_messages = [
            {"role": "user", "content": "What is AI?"}
        ]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Error: Call to gpt-4 failed.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Warning: Call to gpt-3.5 returned empty response.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Should have original message + failure notification
        assert len(result) >= 2
        assert "all initial llm consultations failed or returned no usable content" in result[1]["content"].lower()

    def test_prepare_messages_with_empty_context(self, sample_mom_config):
        """Test message preparation with no intermediate context"""
        request_messages = [
            {"role": "user", "content": "What is AI?"}
        ]
        intermediate_context = []

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Should have original message + failure notification
        assert len(result) >= 2
        assert result[0] == request_messages[0]

    def test_prepare_messages_with_mixed_results(self, sample_mom_config):
        """Test message preparation with both successful and failed responses"""
        request_messages = [
            {"role": "user", "content": "What is AI?"}
        ]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="AI stands for Artificial Intelligence.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Error: Call failed.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Should have original + separator + only the successful response
        assert len(result) >= 3
        # Count assistant messages (should be 1, not 2)
        assistant_messages = [msg for msg in result if msg["role"] == "assistant"]
        assert len(assistant_messages) == 1
        assert "AI stands for Artificial Intelligence" in assistant_messages[0]["content"]

    def test_prepare_messages_with_concluding_prompt(self):
        """Test message preparation with a custom concluding prompt"""
        from mom_service.config import ModelConfig, MoMConfig, ServiceConfig, PromptDefinition

        config = MoMConfig(
            llm_definitions=[],
            prompt_definitions=[
                PromptDefinition(
                    name="synthesis",
                    content="Please synthesize the responses."
                )
            ],
            models=[
                ModelConfig(
                    name="test-model",
                    llms_to_query=["gpt4"],
                    concluding_llm="gpt4",
                    concluding_prompt="synthesis"
                )
            ],
            service=ServiceConfig()
        )

        request_messages = [{"role": "user", "content": "Test"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            ),
        ]

        result = _prepare_concluding_messages(
            request_messages, intermediate_context, config.models[0], config
        )

        # Should include the custom prompt
        assert any("synthesize" in msg["content"].lower() for msg in result)
