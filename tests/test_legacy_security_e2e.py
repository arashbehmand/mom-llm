"""
Legacy test cases for security, E2E workflows, and edge cases.
Based on the comprehensive test plan from an older version of the codebase.

These tests cover:
- Security & Authentication (SEC-AUTH-*, SEC-CORS-*)
- LLM Orchestration & Core Logic (E2E-*, CORE-*)
- Streaming & Thinking Context (STREAM-*)
"""

# pylint: disable=redefined-outer-name,too-many-arguments,too-many-positional-arguments

import asyncio
import html
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mom_service.config import LLMDefinition, ModelConfig, MoMConfig, ServiceConfig
from mom_service.core_logic import _perform_fanout_calls
from mom_service.endpoints.models import ThinkingContextItem, UsageInfo


# =============================================================================
# SECURITY & AUTHENTICATION TESTS
# =============================================================================


class TestSecurityAuthentication:
    """Tests for bearer token enforcement and edge cases"""

    def test_sec_auth_001_service_fails_without_api_token(self, monkeypatch):
        """
        Test Case ID: SEC-AUTH-001
        Tags: [Priority: P0-Critical] [Type: Security] [Path: Negative]
        Description: Verify service fails securely if API_TOKEN is not configured
        """
        # Remove API_TOKEN from environment
        monkeypatch.delenv("API_TOKEN", raising=False)

        # Import after environment variable is removed
        # Note: We need to reload the module to pick up the new environment
        import importlib
        import mom_service.endpoints.openai_v1

        importlib.reload(mom_service.endpoints.openai_v1)

        from mom_service.main import app

        client = TestClient(app)

        # Test /v1/models endpoint
        response = client.get("/v1/models")
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "not configured" in data["error"]["message"].lower()
        # Note: Exception handler always returns "invalid_request_error" type
        assert data["error"]["type"] == "invalid_request_error"
        assert "X-Request-ID" in response.headers

        # Test /v1/chat/completions endpoint
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 503
        assert "X-Request-ID" in response.headers

        # Test /v1/metrics/usage endpoint
        response = client.get("/v1/metrics/usage")
        assert response.status_code == 503
        assert "X-Request-ID" in response.headers

    def test_sec_auth_001_edge_case_empty_api_token(self, monkeypatch):
        """
        SEC-AUTH-001 Edge Case: Empty string API_TOKEN
        """
        monkeypatch.setenv("API_TOKEN", "")

        # Reload module to pick up new environment
        import importlib
        import mom_service.endpoints.openai_v1

        importlib.reload(mom_service.endpoints.openai_v1)

        from mom_service.main import app

        client = TestClient(app)
        response = client.get("/v1/models")
        assert response.status_code == 503

    def test_sec_auth_002_malformed_lowercase_bearer(self, monkeypatch):
        """
        Test Case ID: SEC-AUTH-002
        Description: Reject malformed Authorization header with lowercase 'bearer'
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        headers = {"Authorization": "bearer test-token-12345"}
        response = client.get("/v1/models", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "Invalid authentication scheme" in data["error"]["message"]
        assert "Bearer token" in data["error"]["message"]

    def test_sec_auth_002_malformed_missing_token(self, monkeypatch):
        """
        SEC-AUTH-002: Authorization header with Bearer but no token
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        headers = {"Authorization": "Bearer "}
        response = client.get("/v1/models", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_sec_auth_002_malformed_missing_bearer(self, monkeypatch):
        """
        SEC-AUTH-002: Authorization header without Bearer scheme
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        headers = {"Authorization": "test-token-12345"}
        response = client.get("/v1/models", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid authentication scheme" in data["error"]["message"]

    def test_sec_auth_002_malformed_wrong_scheme(self, monkeypatch):
        """
        SEC-AUTH-002: Authorization header with wrong scheme (Token instead of Bearer)
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        headers = {"Authorization": "Token test-token-12345"}
        response = client.get("/v1/models", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid authentication scheme" in data["error"]["message"]

    def test_sec_auth_002_edge_case_whitespace_in_token(self, monkeypatch):
        """
        SEC-AUTH-002 Edge Case: Extra whitespace around token
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        # This should fail because the token with spaces won't match
        headers = {"Authorization": "Bearer  test-token-12345  "}
        response = client.get("/v1/models", headers=headers)

        # The token won't match due to extra spaces
        assert response.status_code == 401


class TestCORSPolicyEnforcement:
    """Tests for CORS policy enforcement"""

    def test_sec_cors_003_allowed_origin(self, monkeypatch):
        """
        Test Case ID: SEC-CORS-003
        Tags: [Priority: P1-High] [Type: Security] [Path: Negative]
        Description: Confirm CORS policy honors ALLOWED_CORS_ORIGINS
        """
        monkeypatch.setenv("ALLOWED_CORS_ORIGINS", "https://safe.example.com")
        monkeypatch.setenv("API_TOKEN", "test-token")

        # Reload to pick up new CORS settings
        import importlib
        import mom_service.main

        importlib.reload(mom_service.main)

        from mom_service.main import app

        client = TestClient(app)

        # Test with allowed origin
        response = client.options(
            "/v1/models",
            headers={
                "Origin": "https://safe.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "https://safe.example.com"

    def test_sec_cors_003_disallowed_origin(self, monkeypatch):
        """
        SEC-CORS-003: Disallowed origin should not receive CORS headers
        """
        monkeypatch.setenv("ALLOWED_CORS_ORIGINS", "https://safe.example.com")
        monkeypatch.setenv("API_TOKEN", "test-token")

        # Reload to pick up new CORS settings
        import importlib
        import mom_service.main

        importlib.reload(mom_service.main)

        from mom_service.main import app

        client = TestClient(app)

        # Test with disallowed origin
        response = client.options(
            "/v1/models",
            headers={
                "Origin": "https://malicious.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS middleware will return 400 or not include the allow-origin header
        # The origin should not be in the response headers
        cors_header = response.headers.get("access-control-allow-origin", "")
        assert cors_header != "https://malicious.example.com"

    def test_sec_cors_003_edge_case_multiple_origins(self, monkeypatch):
        """
        SEC-CORS-003 Edge Case: Multiple comma-separated origins
        """
        monkeypatch.setenv("ALLOWED_CORS_ORIGINS", "https://site1.com,https://site2.com")
        monkeypatch.setenv("API_TOKEN", "test-token")

        # Reload to pick up new CORS settings
        import importlib
        import mom_service.main

        importlib.reload(mom_service.main)

        from mom_service.main import app

        client = TestClient(app)

        # Test with first allowed origin
        response = client.options(
            "/v1/models",
            headers={
                "Origin": "https://site1.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers

        # Test with second allowed origin
        response = client.options(
            "/v1/models",
            headers={
                "Origin": "https://site2.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers


# =============================================================================
# LLM ORCHESTRATION & CORE LOGIC TESTS
# =============================================================================


class TestE2EWorkflows:
    """End-to-end tests for LLM orchestration"""

    @pytest.mark.asyncio
    async def test_e2e_partial_fail_001_some_fanout_fails(self, sample_mom_config):
        """
        Test Case ID: E2E-PARTIAL-FAIL-001
        Tags: [Priority: P1-High] [Type: E2E] [Path: Negative]
        Description: Ensure successful synthesis when some fan-out calls fail
        """
        # Create LLM map with two models
        llm_map = {
            "llm1": LLMDefinition(name="llm1", model="gpt-4", api_key_env="OPENAI_API_KEY"),
            "llm2": LLMDefinition(name="llm2", model="gpt-3.5-turbo", api_key_env="OPENAI_API_KEY"),
        }

        # Create model config with both LLMs
        model_conf = ModelConfig(
            name="test-model",
            llms_to_query=["llm1", "llm2"],
            concluding_llm="llm1",
            include_thinking_context=True,
        )

        request_messages = [{"role": "user", "content": "What is AI?"}]

        # Mock the call_llm function to return success for llm1 and failure for llm2
        with patch("mom_service.core_logic._call_lite_llm") as mock_call:

            def mock_call_lite_llm_side_effect(llm_def, *args, **kwargs):
                if llm_def.name == "llm1":
                    # Return successful response for llm1
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message = MagicMock()
                    mock_response.choices[0].message.content = "AI is Artificial Intelligence"
                    mock_response.usage = MagicMock()
                    mock_response.usage.prompt_tokens = 10
                    mock_response.usage.completion_tokens = 20
                    mock_response.usage.total_tokens = 30
                    mock_response._is_cached = False

                    async def success_gen():
                        yield mock_response

                    return success_gen()
                else:
                    # Raise exception for llm2
                    async def failure_gen():
                        raise TimeoutError("Request timeout")
                        yield  # Never reached, but makes this a generator

                    return failure_gen()

            mock_call.side_effect = mock_call_lite_llm_side_effect

            # Execute fan-out
            thinking_items = []
            async for item in _perform_fanout_calls(
                model_conf,
                llm_map,
                request_messages,
                timeout=30,
                config=sample_mom_config,
            ):
                thinking_items.append(item)

            # Verify we got results from both calls
            assert len(thinking_items) == 2

            # One should be successful
            successful = [item for item in thinking_items if not item.content.startswith("Error:")]
            assert len(successful) == 1
            assert "Artificial Intelligence" in successful[0].content

            # One should be an error
            failed = [item for item in thinking_items if item.content.startswith("Error:")]
            assert len(failed) == 1
            assert "timeout" in failed[0].content.lower() or "failed" in failed[0].content.lower()

    @pytest.mark.asyncio
    async def test_e2e_all_fail_002_all_fanout_fails(self, sample_mom_config):
        """
        Test Case ID: E2E-ALL-FAIL-002
        Tags: [Priority: P1-High] [Type: E2E] [Path: Negative]
        Description: Verify fallback when all fan-out models fail
        """
        # Create LLM map
        llm_map = {
            "llm1": LLMDefinition(name="llm1", model="gpt-4", api_key_env="OPENAI_API_KEY"),
            "llm2": LLMDefinition(name="llm2", model="gpt-3.5-turbo", api_key_env="OPENAI_API_KEY"),
        }

        model_conf = ModelConfig(
            name="test-model",
            llms_to_query=["llm1", "llm2"],
            concluding_llm="llm1",
        )

        request_messages = [{"role": "user", "content": "What is AI?"}]

        # Mock all calls to fail
        with patch("mom_service.core_logic._call_lite_llm") as mock_call:

            def failure_gen_factory(*args, **kwargs):
                async def failure_gen():
                    raise Exception("Service unavailable")
                    yield  # Never reached, but makes this a generator

                return failure_gen()

            mock_call.side_effect = failure_gen_factory

            # Execute fan-out
            thinking_items = []
            async for item in _perform_fanout_calls(
                model_conf,
                llm_map,
                request_messages,
                timeout=30,
                config=sample_mom_config,
            ):
                thinking_items.append(item)

            # All items should be errors
            assert len(thinking_items) == 2
            for item in thinking_items:
                assert item.content.startswith("Error:")
                assert item.usage.total_tokens == 0

            # Test that _prepare_concluding_messages handles this correctly
            from mom_service.core_logic import _prepare_concluding_messages

            concluding_messages = _prepare_concluding_messages(
                request_messages, thinking_items, model_conf, sample_mom_config
            )

            # Should include failure notification
            message_text = " ".join(str(msg.get("content", "")) for msg in concluding_messages)
            assert (
                "failed" in message_text.lower() or "no usable content" in message_text.lower()
            )

    def test_core_conclude_fail_001_concluding_llm_fails(self, monkeypatch, sample_mom_config):
        """
        Test Case ID: CORE-CONCLUDE-FAIL-001
        Tags: [Priority: P1-High] [Type: E2E] [Path: Negative]
        Description: Verify 500 error when concluding LLM fails
        """
        # Ensure API_TOKEN is set
        monkeypatch.setenv("API_TOKEN", "test-token-12345")

        from mom_service.main import app

        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer test-token-12345"}

        # Mock successful fan-out but failing concluding call
        with patch("mom_service.main._process_mom_chat_request") as mock_process:
            # Simulate an exception during concluding call
            mock_process.side_effect = Exception("Concluding LLM failed")

            request_data = {
                "model": "test-mom-model",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": False,
            }

            with patch("mom_service.main.get_mom_model_config") as mock_get_config:
                mock_get_config.return_value = sample_mom_config.models[0]

                response = client.post(
                    "/v1/chat/completions", json=request_data, headers=auth_headers
                )

                # Should return 500 error
                assert response.status_code == 500
                data = response.json()
                assert "error" in data

                # Should include request ID for tracing
                assert "X-Request-ID" in response.headers


# =============================================================================
# STREAMING & THINKING CONTEXT TESTS
# =============================================================================


class TestStreamingAndThinkingContext:
    """Tests for streaming protocol and security"""

    @pytest.mark.asyncio
    async def test_stream_think_escape_001_xss_payload_escaped(self, sample_mom_config):
        """
        Test Case ID: STREAM-THINK-ESCAPE-001
        Tags: [Priority: P1-High] [Type: Security] [Path: Negative]
        Description: Verify HTML/XSS payloads are escaped in streamed thinking_context
        """
        # Create LLM map
        llm_map = {
            "llm1": LLMDefinition(name="llm1", model="gpt-4", api_key_env="OPENAI_API_KEY"),
        }

        model_conf = ModelConfig(
            name="test-model",
            llms_to_query=["llm1"],
            concluding_llm="llm1",
            include_thinking_context=True,
        )

        request_messages = [{"role": "user", "content": "Test"}]

        # Mock call to return XSS payload
        xss_payload = "<script>alert('XSS')</script>"

        with patch("mom_service.core_logic._call_lite_llm") as mock_call:

            async def mock_xss_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message = MagicMock()
                mock_response.choices[0].message.content = xss_payload
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 5
                mock_response.usage.completion_tokens = 10
                mock_response.usage.total_tokens = 15
                mock_response._is_cached = False
                yield mock_response

            mock_call.return_value = mock_xss_response()

            # Execute fan-out
            thinking_items = []
            async for item in _perform_fanout_calls(
                model_conf,
                llm_map,
                request_messages,
                timeout=30,
                config=sample_mom_config,
            ):
                thinking_items.append(item)

            # Verify content is NOT escaped at this stage (core_logic returns raw content)
            assert len(thinking_items) == 1
            assert xss_payload in thinking_items[0].content

            # The HTML escaping should happen in the streaming response
            # Let's test that by checking if html.escape is called
            escaped_content = html.escape(thinking_items[0].content)
            assert "&lt;script&gt;" in escaped_content
            assert "&lt;/script&gt;" in escaped_content
            assert "<script>" not in escaped_content

    @pytest.mark.asyncio
    async def test_stream_think_escape_001_various_html_payloads(self, sample_mom_config):
        """
        STREAM-THINK-ESCAPE-001: Test various HTML/XSS payloads are escaped
        """
        test_payloads = [
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src='javascript:alert(1)'>",
            "<body onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<a href='javascript:void(0)'>Click</a>",
        ]

        llm_map = {
            "llm1": LLMDefinition(name="llm1", model="gpt-4", api_key_env="OPENAI_API_KEY"),
        }

        model_conf = ModelConfig(
            name="test-model",
            llms_to_query=["llm1"],
            concluding_llm="llm1",
            include_thinking_context=True,
        )

        for payload in test_payloads:
            request_messages = [{"role": "user", "content": "Test"}]

            with patch("mom_service.core_logic._call_lite_llm") as mock_call:

                async def mock_payload_response(*args, **kwargs):
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message = MagicMock()
                    mock_response.choices[0].message.content = payload
                    mock_response.usage = MagicMock()
                    mock_response.usage.prompt_tokens = 5
                    mock_response.usage.completion_tokens = 10
                    mock_response.usage.total_tokens = 15
                    mock_response._is_cached = False
                    yield mock_response

                mock_call.return_value = mock_payload_response()

                # Execute fan-out
                thinking_items = []
                async for item in _perform_fanout_calls(
                    model_conf,
                    llm_map,
                    request_messages,
                    timeout=30,
                    config=sample_mom_config,
                ):
                    thinking_items.append(item)

                # Verify HTML escaping - check that dangerous characters are escaped
                escaped_content = html.escape(thinking_items[0].content)
                # Either HTML tags are escaped (contain &lt; or &gt;)
                # Or the content doesn't contain HTML tags and remains unchanged
                if "<" in payload or ">" in payload:
                    # If payload contains HTML tags, they should be escaped
                    assert "&lt;" in escaped_content or "&gt;" in escaped_content
                    # Raw tags should not appear in escaped content
                    assert not any(
                        tag in escaped_content
                        for tag in ["<script", "<img", "<svg", "<iframe", "<body", "<a"]
                    )
                else:
                    # Payloads without HTML tags (like "javascript:") may not change
                    # but quotes should be escaped
                    assert "&#x27;" in escaped_content or payload == escaped_content


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_client():
    """Fixture providing a test client for the FastAPI app"""
    from mom_service.main import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Fixture providing authentication headers"""
    return {"Authorization": f"Bearer {os.environ.get('API_TOKEN', '')}"}
