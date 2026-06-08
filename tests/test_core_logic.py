"""
Unit tests for mom_service.core_logic module
"""

import asyncio
from builtins import anext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mom_service.config import LLMDefinition, ModelConfig
from mom_service.core_logic import (
    _calculate_and_log_costs,
    _perform_fanout_calls,
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
                usage=UsageInfo(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001
                ),
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Response 2",
                usage=UsageInfo(
                    prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.0005
                ),
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
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=None),
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Response 2",
                usage=UsageInfo(
                    prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.001
                ),
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
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.0),
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
                usage=UsageInfo(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.002
                ),
            ),
        ]

        total_cost = _calculate_and_log_costs(fanout_context, None)
        assert total_cost == 0.002


class TestPrepareConcludingMessages:
    """Tests for _prepare_concluding_messages function"""

    def test_prepare_messages_with_successful_fanout(self, sample_mom_config):
        """Test message preparation with successful fanout responses"""
        request_messages = [{"role": "user", "content": "What is AI?"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="AI stands for Artificial Intelligence.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="AI is a branch of computer science.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Responses are embedded in a single user message with numbered delimiters.
        assert len(result) >= 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is AI?"
        assert result[1]["role"] == "user"
        assert "2 independent LLM responses" in result[1]["content"]
        assert "===== RESPONSE 1 of 2 =====" in result[1]["content"]
        assert "===== RESPONSE 2 of 2 =====" in result[1]["content"]
        assert "AI stands for Artificial Intelligence." in result[1]["content"]
        assert "AI is a branch of computer science." in result[1]["content"]

    def test_prepare_messages_with_failed_fanout(self, sample_mom_config):
        """Test message preparation when all fanout calls failed"""
        request_messages = [{"role": "user", "content": "What is AI?"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Error: Call to gpt-4 failed.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Warning: Call to gpt-3.5 returned empty response.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Should have original message + failure notification
        assert len(result) >= 2
        assert (
            "all initial llm consultations failed or returned no usable content"
            in result[1]["content"].lower()
        )

    def test_prepare_messages_with_empty_context(self, sample_mom_config):
        """Test message preparation with no intermediate context"""
        request_messages = [{"role": "user", "content": "What is AI?"}]
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
        request_messages = [{"role": "user", "content": "What is AI?"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="AI stands for Artificial Intelligence.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            ),
            ThinkingContextItem(
                model="gpt-3.5",
                content="Error: Call failed.",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
        ]

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages, intermediate_context, model_conf, sample_mom_config
        )

        # Only the successful response should be embedded; the failed one excluded.
        assert len(result) >= 2
        combined = result[1]["content"]
        assert "1 independent LLM responses" in combined
        assert "===== RESPONSE 1 of 1 =====" in combined
        assert "AI stands for Artificial Intelligence" in combined
        assert "Error: Call failed." not in combined

    def test_prepare_messages_with_concluding_prompt(self):
        """Test message preparation with a custom concluding prompt"""
        from mom_service.config import MoMConfig, PromptDefinition, ServiceConfig

        config = MoMConfig(
            llm_definitions=[],
            prompt_definitions=[
                PromptDefinition(name="synthesis", content="Please synthesize the responses.")
            ],
            models=[
                ModelConfig(
                    name="test-model",
                    llms_to_query=["gpt4"],
                    concluding_llm="gpt4",
                    concluding_prompt="synthesis",
                )
            ],
            service=ServiceConfig(),
        )

        request_messages = [{"role": "user", "content": "Test"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="Response",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            ),
        ]

        result = _prepare_concluding_messages(
            request_messages, intermediate_context, config.models[0], config
        )

        # Should include the custom prompt
        assert any("synthesize" in msg["content"].lower() for msg in result)

    def test_prepare_messages_with_concluding_instruction(self, sample_mom_config):
        """Test message preparation with a concluding instruction"""
        request_messages = [{"role": "user", "content": "What is AI?"}]
        intermediate_context = [
            ThinkingContextItem(
                model="gpt-4",
                content="AI stands for Artificial Intelligence.",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            ),
        ]
        concluding_instruction = "Please provide a very short answer."

        model_conf = sample_mom_config.models[0]
        result = _prepare_concluding_messages(
            request_messages,
            intermediate_context,
            model_conf,
            sample_mom_config,
            concluding_instruction,
        )

        # Should include the concluding instruction as the last user message
        # (or second to last if there is a concluding prompt)
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == concluding_instruction


class TestFanoutProgressEvents:  # pylint: disable=too-few-public-methods
    """Tests for fanout progress event behavior."""

    @pytest.mark.asyncio
    async def test_fanout_completion_events_continue_after_consumer_disconnect(
        self, sample_mom_config
    ):
        """Fanout completion events should still publish after stream consumer disconnects."""
        llm_map = {
            "llm1": LLMDefinition(name="llm1", model="gpt-4", api_key_env="OPENAI_API_KEY"),
            "llm2": LLMDefinition(name="llm2", model="gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
        }
        model_conf = ModelConfig(
            name="test-model",
            llms_to_query=["llm1", "llm2"],
            concluding_llm="llm1",
            include_thinking_context=True,
        )
        request_messages = [{"role": "user", "content": "What is AI?"}]
        redis_publisher = SimpleNamespace(enabled=True, publish=AsyncMock())

        def _mock_response(content: str, total_tokens: int) -> MagicMock:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = content
            response.usage = MagicMock()
            response.usage.prompt_tokens = max(total_tokens // 3, 1)
            response.usage.completion_tokens = max(total_tokens - response.usage.prompt_tokens, 1)
            response.usage.total_tokens = total_tokens
            response._is_cached = False  # pylint: disable=protected-access
            return response

        async def _fast_response_gen(*_, **__):
            await asyncio.sleep(0.01)
            yield _mock_response("fast-response", 30)

        async def _slow_response_gen(*_, **__):
            await asyncio.sleep(0.15)
            yield _mock_response("slow-response", 45)

        with patch("mom_service.core_logic._call_lite_llm") as mock_call:

            def _side_effect(llm_def, *_, **__):
                if llm_def.name == "llm1":
                    return _fast_response_gen()
                return _slow_response_gen()

            mock_call.side_effect = _side_effect

            fanout_gen = _perform_fanout_calls(
                model_conf,
                llm_map,
                request_messages,
                timeout=30,
                config=sample_mom_config,
                options={"request_id": "rid-disconnect", "redis_publisher": redis_publisher},
            )

            # Simulate a client consuming one streamed item and then disconnecting.
            await anext(fanout_gen)
            await fanout_gen.aclose()

            # Allow detached fanout tasks to finish and publish completion events.
            await asyncio.sleep(0.25)

        published_events = [call.args[0] for call in redis_publisher.publish.await_args_list]
        fanout_complete = [event for event in published_events if event.type == "fanout_complete"]

        assert len(fanout_complete) == 2
        assert {event.data["model"] for event in fanout_complete} == {"llm1", "llm2"}
