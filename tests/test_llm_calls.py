"""
Integration tests for mom_service.llm_calls module
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import respx
from httpx import Response

from mom_service.llm_calls import (
    _init_cache_db,
    _generate_cache_key,
    _cache_response,
    _get_cached_response,
    _call_lite_llm,
)
from mom_service.endpoints.models import UsageInfo


class TestCacheDatabase:
    """Tests for cache database functions"""

    def test_init_cache_db(self):
        """Test that cache database initializes successfully"""
        # This should not raise an exception
        _init_cache_db()

    def test_generate_cache_key_consistency(self, sample_llm_definition, sample_request_messages):
        """Test that cache key generation is consistent"""
        params = {"temperature": 0.7}

        key1 = _generate_cache_key(sample_llm_definition, sample_request_messages, params)
        key2 = _generate_cache_key(sample_llm_definition, sample_request_messages, params)

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    def test_generate_cache_key_different_for_different_inputs(
        self, sample_llm_definition, sample_request_messages
    ):
        """Test that different inputs produce different cache keys"""
        params1 = {"temperature": 0.7}
        params2 = {"temperature": 0.9}

        key1 = _generate_cache_key(sample_llm_definition, sample_request_messages, params1)
        key2 = _generate_cache_key(sample_llm_definition, sample_request_messages, params2)

        assert key1 != key2


class TestCacheOperations:
    """Tests for cache get/set operations"""

    def test_cache_and_retrieve_response(self, mock_litellm_response):
        """Test caching and retrieving a response"""
        cache_key = "test_cache_key_12345"
        messages = [{"role": "user", "content": "Test message"}]

        # Cache the response
        _cache_response(cache_key, messages, mock_litellm_response)

        # Retrieve it
        cached = _get_cached_response(cache_key)

        assert cached is not None
        assert hasattr(cached, '_is_cached')
        assert cached._is_cached is True
        assert cached.id == mock_litellm_response.id
        assert cached.model == mock_litellm_response.model
        assert cached.choices[0].message.content == "Paris is the capital of France."

    def test_get_cached_response_miss(self):
        """Test retrieving a non-existent cache entry"""
        cached = _get_cached_response("nonexistent_key_9999")
        assert cached is None


@pytest.mark.asyncio
class TestCallLiteLLM:
    """Tests for _call_lite_llm function"""

    @respx.mock
    async def test_call_litellm_success(
        self, sample_llm_definition, sample_request_messages, sample_mom_config, mock_litellm_response
    ):
        """Test successful LLM call"""
        with patch('mom_service.llm_calls.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            result_gen = _call_lite_llm(
                sample_llm_definition,
                sample_request_messages,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-request-123",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                }
            )

            result = await anext(result_gen)

            assert result is not None
            assert result.id == "test-response-id"
            assert result.choices[0].message.content == "Paris is the capital of France."
            mock_acompletion.assert_called_once()

    @respx.mock
    async def test_call_litellm_with_cache_hit(
        self, sample_llm_definition, sample_request_messages, sample_mom_config, mock_litellm_response
    ):
        """Test LLM call with cache hit"""
        # Enable caching
        sample_mom_config.service.cache_enabled = True

        # First call - should cache
        with patch('mom_service.llm_calls.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            result_gen = _call_lite_llm(
                sample_llm_definition,
                sample_request_messages,
                timeout=30,
                config=sample_mom_config,
                options={"request_id": "test-1", "mom_model_name": "test"}
            )
            await anext(result_gen)

        # Second call - should hit cache
        with patch('mom_service.llm_calls.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion_2:
            result_gen2 = _call_lite_llm(
                sample_llm_definition,
                sample_request_messages,
                timeout=30,
                config=sample_mom_config,
                options={"request_id": "test-2", "mom_model_name": "test"}
            )
            cached_result = await anext(result_gen2)

            # Should not have called the API again
            mock_acompletion_2.assert_not_called()

            # Should have the cached flag
            assert hasattr(cached_result, '_is_cached')
            assert cached_result._is_cached is True

    @respx.mock
    async def test_call_litellm_failure(
        self, sample_llm_definition, sample_request_messages, sample_mom_config
    ):
        """Test LLM call failure handling"""
        with patch('mom_service.llm_calls.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API Error")

            result_gen = _call_lite_llm(
                sample_llm_definition,
                sample_request_messages,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-request-fail",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                }
            )

            with pytest.raises(Exception) as exc_info:
                await anext(result_gen)

            assert "API Error" in str(exc_info.value)

    @respx.mock
    async def test_call_litellm_with_trace(
        self, sample_llm_definition, sample_request_messages, sample_mom_config,
        mock_litellm_response, mock_langfuse_client
    ):
        """Test LLM call with Langfuse tracing"""
        with patch('mom_service.llm_calls.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            result_gen = _call_lite_llm(
                sample_llm_definition,
                sample_request_messages,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-trace",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                    "trace": mock_langfuse_client.trace.return_value,
                    "generation_name": "test-generation",
                }
            )

            result = await anext(result_gen)

            assert result is not None
            # Verify that trace.generation was called
            mock_langfuse_client.trace.return_value.generation.assert_called_once()


# Import anext for Python 3.9 compatibility
try:
    from builtins import anext
except ImportError:
    async def anext(ait):
        return await ait.__anext__()
