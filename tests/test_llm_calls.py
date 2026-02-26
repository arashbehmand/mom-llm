"""
Integration tests for mom_service.llm_calls module
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import respx

from mom_service.config import LLMDefinition
from mom_service.core_logic import call_llm
from mom_service.endpoints.models import LLMCallParams
from mom_service.llm_calls import (
    _cache_response,
    _generate_cache_key,
    _get_cached_response,
    _init_cache_db,
)


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

    def test_generate_cache_key_different_for_different_messages(self, sample_llm_definition):
        """Different message content must always produce different cache keys."""
        params = {"temperature": 0.7}
        messages1 = [{"role": "user", "content": "hello"}]
        messages2 = [{"role": "user", "content": "hello there"}]

        key1 = _generate_cache_key(sample_llm_definition, messages1, params)
        key2 = _generate_cache_key(sample_llm_definition, messages2, params)

        assert key1 != key2

    def test_generate_cache_key_ignores_runtime_and_sensitive_params(
        self, sample_llm_definition, sample_request_messages
    ):
        """Runtime metadata and secrets should not affect cache-key identity."""
        params1 = {
            "temperature": 0.7,
            "timeout": 30,
            "num_retries": 2,
            "api_key": "sk-first",
            "messages": [{"role": "user", "content": "shadow copy"}],
            "_api_route": "completion",
            "extra_headers": {
                "Authorization": "Bearer top-secret-1",
                "X-Client": "mom-service",
            },
        }
        params2 = {
            "temperature": 0.7,
            "timeout": 120,
            "num_retries": 9,
            "api_key": "sk-second",
            "messages": [{"role": "user", "content": "different shadow copy"}],
            "_api_route": "completion",
            "extra_headers": {
                "Authorization": "Bearer top-secret-2",
                "X-Client": "mom-service",
            },
        }

        key1 = _generate_cache_key(sample_llm_definition, sample_request_messages, params1)
        key2 = _generate_cache_key(sample_llm_definition, sample_request_messages, params2)

        assert key1 == key2


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
        assert hasattr(cached, "_is_cached")
        assert cached._is_cached is True  # pylint: disable=protected-access
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
        self,
        sample_llm_definition,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
    ):
        """Test successful LLM call"""
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            params = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=False,
            )

            result_gen = call_llm(
                sample_llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-request-123",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                },
            )

            result = await anext(result_gen)

            assert result is not None
            assert result.id == "test-response-id"
            assert result.choices[0].message.content == "Paris is the capital of France."
            mock_acompletion.assert_called_once()

    @respx.mock
    async def test_call_litellm_with_cache_hit(
        self,
        sample_llm_definition,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
    ):
        """Test LLM call with cache hit"""
        # Enable caching
        sample_mom_config.service.cache_enabled = True

        # First call - should cache
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            params1 = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=False,
            )
            result_gen = call_llm(
                sample_llm_definition,
                params1,
                timeout=30,
                config=sample_mom_config,
                options={"request_id": "test-1", "mom_model_name": "test"},
            )
            await anext(result_gen)

        # Second call - should hit cache
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion_2:
            params2 = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=False,
            )
            result_gen2 = call_llm(
                sample_llm_definition,
                params2,
                timeout=30,
                config=sample_mom_config,
                options={"request_id": "test-2", "mom_model_name": "test"},
            )
            cached_result = await anext(result_gen2)

            # Should not have called the API again
            mock_acompletion_2.assert_not_called()

            # Should have the cached flag
            assert hasattr(cached_result, "_is_cached")
            assert cached_result._is_cached is True  # pylint: disable=protected-access

    @respx.mock
    async def test_call_litellm_failure(
        self, sample_llm_definition, sample_request_messages, sample_mom_config
    ):
        """Test LLM call failure handling"""
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API Error")

            params = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=False,
            )
            result_gen = call_llm(
                sample_llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-request-fail",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                },
            )

            with pytest.raises(Exception) as exc_info:
                await anext(result_gen)

            assert "API Error" in str(exc_info.value)

    @respx.mock
    async def test_call_litellm_streaming_error_chunk_marks_langfuse_error(
        self,
        sample_llm_definition,
        sample_request_messages,
        sample_mom_config,
        mock_langfuse_client,
    ):
        """Provider error chunks in streaming mode must be marked as Langfuse errors."""

        class _ErrorChunkStream:
            def __init__(self):
                self._sent_error = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._sent_error:
                    raise StopAsyncIteration
                self._sent_error = True
                return {"error": {"type": "provider_error", "message": "upstream outage"}}

        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = _ErrorChunkStream()

            params = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=True,
            )
            result_gen = call_llm(
                sample_llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-stream-error",
                    "mom_model_name": "test-model",
                    "call_type": "concluding",
                    "trace": mock_langfuse_client.trace.return_value,
                    "generation_name": "test-concluding-stream",
                },
            )

            with pytest.raises(RuntimeError, match="upstream outage"):
                await anext(result_gen)

            generation = mock_langfuse_client.trace.return_value.generation.return_value
            generation.end.assert_called_once()
            end_kwargs = generation.end.call_args.kwargs
            assert end_kwargs["level"] == "ERROR"
            assert "upstream outage" in end_kwargs["status_message"]

    @respx.mock
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def test_call_litellm_with_trace(
        self,
        sample_llm_definition,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
        mock_langfuse_client,
    ):
        """Test LLM call with Langfuse tracing"""
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            params = LLMCallParams(
                model=sample_llm_definition.model,
                messages=sample_request_messages,
                temperature=(
                    sample_llm_definition.params.get("temperature")
                    if isinstance(sample_llm_definition.params, dict)
                    else None
                ),
                stream=False,
            )
            result_gen = call_llm(
                sample_llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-trace",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                    "trace": mock_langfuse_client.trace.return_value,
                    "generation_name": "test-generation",
                },
            )

            result = await anext(result_gen)

            assert result is not None
            # Verify that trace.generation was called
            mock_langfuse_client.trace.return_value.generation.assert_called_once()

    @respx.mock
    async def test_call_litellm_trace_does_not_include_api_key(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
        mock_langfuse_client,
    ):
        """Langfuse model parameters must not leak credentials."""
        llm_definition = LLMDefinition(
            name="trace-safe-llm",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
            params={
                "temperature": 0.2,
                "extra_headers": {
                    "Authorization": "Bearer secret-header-token",
                    "X-Client": "mom-service",
                },
            },
        )

        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response

            params = LLMCallParams(
                model=llm_definition.model,
                messages=sample_request_messages,
                stream=False,
            )
            result_gen = call_llm(
                llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "trace-safety-test",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                    "trace": mock_langfuse_client.trace.return_value,
                    "generation_name": "trace-safe-generation",
                },
            )

            result = await anext(result_gen)
            assert result is not None

            generation_kwargs = mock_langfuse_client.trace.return_value.generation.call_args.kwargs
            model_parameters = generation_kwargs["model_parameters"]

            assert "api_key" not in model_parameters
            assert "extra_headers" in model_parameters
            assert "secret-header-token" not in model_parameters["extra_headers"]
            assert "[REDACTED]" in model_parameters["extra_headers"]

    @respx.mock
    async def test_call_auto_api_mode_with_xai_tools_uses_completion_api(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
    ):
        """Test api_mode='auto' stays on chat completion path even with xAI tools."""
        xai_llm_definition = LLMDefinition(
            name="grok41fr:o",
            model="xai/grok-4-1-fast-reasoning",
            api_key_env="XAI_API_KEY",
            params={
                "tools": [
                    {"type": "web_search"},
                    {"type": "x_search"},
                ]
            },
        )

        with (
            patch(
                "mom_service.responses_api.litellm.aresponses", new_callable=AsyncMock
            ) as mock_aresponses,
            patch(
                "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion,
        ):
            mock_acompletion.return_value = mock_litellm_response

            params = LLMCallParams(
                model=xai_llm_definition.model,
                messages=sample_request_messages,
                stream=False,
            )
            result_gen = call_llm(
                xai_llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-xai-tools",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                },
            )

            result = await anext(result_gen)

            assert result.id == "test-response-id"
            mock_acompletion.assert_called_once()
            mock_aresponses.assert_not_called()

    @respx.mock
    async def test_call_explicit_responses_api_mode_routes_to_responses_api(
        self,
        sample_request_messages,
        sample_mom_config,
    ):
        """Test api_mode='responses' routes to Responses API without tools."""
        llm_definition = LLMDefinition(
            name="grok41fr:responses",
            model="xai/grok-4-1-fast-reasoning",
            api_key_env="XAI_API_KEY",
            api_mode="responses",
            params={"temperature": 0.3},
        )

        mock_usage = SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            total_tokens=19,
            input_tokens_details=SimpleNamespace(text_tokens=12, cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3, text_tokens=4),
            num_sources_used=0,
        )
        mock_responses_obj = SimpleNamespace(
            id="resp-test-456",
            created_at=1234567891,
            output_text="Responses mode output.",
            usage=mock_usage,
            output=[],
        )

        with (
            patch(
                "mom_service.responses_api.litellm.aresponses", new_callable=AsyncMock
            ) as mock_aresponses,
            patch(
                "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion,
        ):
            mock_aresponses.return_value = mock_responses_obj

            params = LLMCallParams(
                model=llm_definition.model,
                messages=sample_request_messages,
                stream=False,
            )
            result_gen = call_llm(
                llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-explicit-responses",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                },
            )
            result = await anext(result_gen)

            assert result.choices[0].message.content == "Responses mode output."
            mock_aresponses.assert_called_once()
            mock_acompletion.assert_not_called()

    @respx.mock
    async def test_call_explicit_completion_api_mode_keeps_completion_api(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
    ):
        """Test api_mode='completion' keeps chat completion path even with responses-style tools."""
        llm_definition = LLMDefinition(
            name="grok41fr:completion",
            model="xai/grok-4-1-fast-reasoning",
            api_key_env="XAI_API_KEY",
            api_mode="completion",
            params={"tools": [{"type": "web_search"}]},
        )

        with (
            patch(
                "mom_service.responses_api.litellm.aresponses", new_callable=AsyncMock
            ) as mock_aresponses,
            patch(
                "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion,
        ):
            mock_acompletion.return_value = mock_litellm_response

            params = LLMCallParams(
                model=llm_definition.model,
                messages=sample_request_messages,
                stream=False,
            )
            result_gen = call_llm(
                llm_definition,
                params,
                timeout=30,
                config=sample_mom_config,
                options={
                    "request_id": "test-explicit-completion",
                    "mom_model_name": "test-model",
                    "call_type": "fanout",
                },
            )
            result = await anext(result_gen)

            assert result.id == "test-response-id"
            mock_acompletion.assert_called_once()
            mock_aresponses.assert_not_called()


# Import anext for Python 3.9 compatibility
try:
    from builtins import anext
except ImportError:

    async def anext(ait):
        # Use getattr to avoid a direct dunder call which pylint flags as unnecessary-dunder-call
        func = ait.__anext__
        return await func()
