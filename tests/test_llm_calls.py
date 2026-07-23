"""
Integration tests for mom_service.llm_calls module
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web
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
    _ProxyAsyncHTTPHandler,
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

    def test_generate_cache_key_distinguishes_invocation_aliases(
        self, sample_llm_definition, sample_request_messages
    ):
        base = sample_llm_definition.model_copy(update={"name": "g36f:h"})
        alias_a = sample_llm_definition.model_copy(update={"name": "g36f:h+a"})
        alias_b = sample_llm_definition.model_copy(update={"name": "g36f:h+b"})

        keys = {
            _generate_cache_key(llm_def, sample_request_messages, {"reasoning_effort": "high"})
            for llm_def in (base, alias_a, alias_b)
        }

        assert len(keys) == 3

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
async def test_proxy_http_handler_routes_requests_through_configured_proxy():
    proxy_requests = []

    async def proxy_handler(request):
        proxy_requests.append(request.raw_path)
        return web.json_response({"proxied": True})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", proxy_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    client = _ProxyAsyncHTTPHandler(proxy_url=f"http://127.0.0.1:{port}", timeout=5)
    try:
        response = await client.post(
            "http://upstream.example/v1/chat/completions",
            json={"model": "test"},
        )
    finally:
        await client.close()
        await runner.cleanup()

    assert response.json() == {"proxied": True}
    assert proxy_requests
    assert "upstream.example" in proxy_requests[0]


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

    async def test_call_litellm_uses_and_closes_configured_proxy_session(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
        monkeypatch,
    ):
        proxy_url = "http://proxy-user:proxy-password@us-proxy.example:8080"
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", proxy_url)
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
            params={"reasoning_effort": "high"},
        )
        captured_client = None

        async def fake_acompletion(**kwargs):
            nonlocal captured_client
            captured_client = kwargs["client"]
            return mock_litellm_response

        with patch("mom_service.llm_calls.litellm.acompletion", side_effect=fake_acompletion):
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=False,
                ),
                timeout=30,
                config=sample_mom_config,
            )

            result = await anext(result_gen)
            await result_gen.aclose()

        assert result.id == "test-response-id"
        assert captured_client is not None
        assert captured_client.client.is_closed is True

    async def test_call_litellm_requires_configured_proxy_environment_variable(
        self,
        sample_request_messages,
        sample_mom_config,
        monkeypatch,
    ):
        monkeypatch.delenv("MUSE_SPARK_PROXY_URL", raising=False)
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )

        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=False,
                ),
                timeout=30,
                config=sample_mom_config,
            )

            with pytest.raises(ValueError, match="MUSE_SPARK_PROXY_URL"):
                await anext(result_gen)

            mock_acompletion.assert_not_called()

    async def test_proxy_url_is_not_added_to_litellm_params(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_litellm_response,
        monkeypatch,
    ):
        proxy_url = "http://proxy-user:proxy-password@us-proxy.example:8080"
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", proxy_url)
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )

        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_litellm_response
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=False,
                ),
                timeout=30,
                config=sample_mom_config,
            )

            await anext(result_gen)
            await result_gen.aclose()

        call_kwargs = mock_acompletion.call_args.kwargs
        assert proxy_url not in repr(call_kwargs)
        assert "proxy_url" not in call_kwargs
        assert "proxy_url_env" not in call_kwargs

    async def test_proxy_client_closes_when_provider_call_fails(
        self,
        sample_request_messages,
        sample_mom_config,
        monkeypatch,
    ):
        proxy_url = "http://proxy-user:proxy-password@us-proxy.example:8080"
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", proxy_url)
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )
        captured_client = None

        async def failing_acompletion(**kwargs):
            nonlocal captured_client
            captured_client = kwargs["client"]
            raise RuntimeError(f"proxy connection failed: {proxy_url}")

        with patch("mom_service.llm_calls.litellm.acompletion", side_effect=failing_acompletion):
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=False,
                ),
                timeout=30,
                config=sample_mom_config,
            )

            with pytest.raises(RuntimeError):
                await anext(result_gen)

        assert captured_client is not None
        assert captured_client.client.is_closed is True

    async def test_proxied_error_does_not_leak_proxy_url_to_observability(
        self,
        sample_request_messages,
        sample_mom_config,
        mock_langfuse_client,
        monkeypatch,
        caplog,
    ):
        proxy_url = "http://proxy-user:proxy-password@us-proxy.example:8080"
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", proxy_url)
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )

        with (
            patch(
                "mom_service.llm_calls.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError(f"proxy connection failed: {proxy_url}"),
            ),
            patch("mom_service.llm_calls.metrics_db.insert_metric_record") as insert_metric,
        ):
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=False,
                ),
                timeout=30,
                config=sample_mom_config,
                options={
                    "trace": mock_langfuse_client.start_observation.return_value,
                    "generation_name": "proxied-call",
                },
            )

            with pytest.raises(RuntimeError) as exc_info:
                await anext(result_gen)

        metric_record = insert_metric.call_args.args[0]
        generation = (
            mock_langfuse_client.start_observation.return_value.start_observation.return_value
        )
        trace_status = generation.update.call_args.kwargs["status_message"]

        assert proxy_url not in caplog.text
        assert "proxy-password" not in caplog.text
        assert proxy_url not in metric_record.error_message
        assert "proxy-password" not in metric_record.error_message
        assert proxy_url not in trace_status
        assert "proxy-password" not in trace_status
        assert str(exc_info.value) == "Provider request failed through configured proxy."
        assert proxy_url not in str(exc_info.value)
        assert exc_info.value.__context__ is None
        assert exc_info.value.__cause__ is None

    async def test_proxied_cache_hit_does_not_require_proxy_environment_variable(
        self,
        sample_mom_config,
        mock_litellm_response,
        monkeypatch,
    ):
        proxy_url = "http://proxy-user:proxy-password@us-proxy.example:8080"
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", proxy_url)
        sample_mom_config.service.cache_enabled = True
        llm_definition = LLMDefinition(
            name="muse11:cache-test",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )
        messages = [{"role": "user", "content": "unique proxied cache test"}]
        call_params = LLMCallParams(
            model=llm_definition.model,
            messages=messages,
            stream=False,
        )

        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as first_call:
            first_call.return_value = mock_litellm_response
            first_result_gen = call_llm(
                llm_definition,
                call_params,
                timeout=30,
                config=sample_mom_config,
            )
            await anext(first_result_gen)
            await first_result_gen.aclose()

        monkeypatch.delenv("MUSE_SPARK_PROXY_URL")
        with patch(
            "mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock
        ) as second_call:
            second_result_gen = call_llm(
                llm_definition,
                call_params,
                timeout=30,
                config=sample_mom_config,
            )
            cached_result = await anext(second_result_gen)

        second_call.assert_not_called()
        assert cached_result._is_cached is True

    async def test_streaming_proxy_client_closes_on_early_generator_close(
        self,
        sample_request_messages,
        sample_mom_config,
        monkeypatch,
    ):
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", "http://us-proxy.example:8080")
        llm_definition = LLMDefinition(
            name="muse11:h",
            model="openrouter/meta/muse-spark-1.1",
            api_key_env="OPENROUTER_API_KEY",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
        )
        captured_client = None

        async def response_stream():
            yield {"choices": [{"delta": {"content": "partial"}}]}
            yield {"choices": [{"delta": {"content": " response"}}]}

        async def fake_acompletion(**kwargs):
            nonlocal captured_client
            captured_client = kwargs["client"]
            return response_stream()

        with patch("mom_service.llm_calls.litellm.acompletion", side_effect=fake_acompletion):
            result_gen = call_llm(
                llm_definition,
                LLMCallParams(
                    model=llm_definition.model,
                    messages=sample_request_messages,
                    stream=True,
                ),
                timeout=30,
                config=sample_mom_config,
            )

            first_chunk = await anext(result_gen)
            assert captured_client.client.is_closed is False
            await result_gen.aclose()

        assert first_chunk["choices"][0]["delta"]["content"] == "partial"
        assert captured_client.client.is_closed is True

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
                    "trace": mock_langfuse_client.start_observation.return_value,
                    "generation_name": "test-concluding-stream",
                },
            )

            with pytest.raises(RuntimeError, match="upstream outage"):
                await anext(result_gen)

            generation = (
                mock_langfuse_client.start_observation.return_value.start_observation.return_value
            )
            generation.end.assert_called_once()
            update_kwargs = generation.update.call_args.kwargs
            assert update_kwargs["level"] == "ERROR"
            assert "upstream outage" in update_kwargs["status_message"]

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
                    "trace": mock_langfuse_client.start_observation.return_value,
                    "generation_name": "test-generation",
                },
            )

            result = await anext(result_gen)

            assert result is not None
            # Verify that trace.start_generation was called
            mock_langfuse_client.start_observation.return_value.start_observation.assert_called_once()

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
                    "trace": mock_langfuse_client.start_observation.return_value,
                    "generation_name": "trace-safe-generation",
                },
            )

            result = await anext(result_gen)
            assert result is not None

            generation_kwargs = (
                mock_langfuse_client.start_observation.return_value.start_observation.call_args.kwargs
            )
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
        monkeypatch,
    ):
        """Test api_mode='responses' routes to Responses API without tools."""
        monkeypatch.setenv("MUSE_SPARK_PROXY_URL", "http://us-proxy.example:8080")
        llm_definition = LLMDefinition(
            name="grok41fr:responses",
            model="xai/grok-4-1-fast-reasoning",
            api_key_env="XAI_API_KEY",
            api_mode="responses",
            proxy_url_env="MUSE_SPARK_PROXY_URL",
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
            responses_client = mock_aresponses.call_args.kwargs["client"]
            assert responses_client.client.is_closed is True

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
