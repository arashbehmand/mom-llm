"""
Integration tests for mom_service API endpoints
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Fixture providing a test client for the FastAPI app"""
    # Import after setting env vars
    from mom_service.main import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Fixture providing authentication headers"""
    return {"Authorization": f"Bearer {os.environ.get('API_TOKEN', '')}"}


class TestOpenAIEndpoints:
    """Tests for OpenAI-compatible API endpoints"""

    def test_get_models_success(self, test_client, auth_headers):
        """Test GET /v1/models endpoint"""
        response = test_client.get("/v1/models", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_models_unauthorized(self, test_client):
        """Test GET /v1/models without authentication"""
        response = test_client.get("/v1/models")

        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_models_invalid_token(self, test_client):
        """Test GET /v1/models with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = test_client.get("/v1/models", headers=headers)

        assert response.status_code == 401

    @patch("mom_service.main.get_mom_model_config")
    @patch("mom_service.main._process_mom_chat_request")
    def test_chat_completions_non_streaming(self, mock_process, mock_get_config, test_client, auth_headers):
        """Test POST /v1/chat/completions for non-streaming"""
        # Mock the response
        mock_process.return_value = (
            "Paris is the capital of France.",  # final_content
            None,  # thinking_context
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
            },  # usage
            0.001,  # total_cost
            "test-mom-model",  # mom_model_name
            False,  # thinking_embedded
            None,  # trace_obj
        )

        # Mock the get_mom_model_config function to return a valid config
        mock_get_config.return_value = MagicMock(
            name="test-mom-model",
            llms_to_query=["test-llm"],
            concluding_llm="test-concluding-llm",
            include_thinking_context=True,
        )

        request_data = {
            "model": "test-mom-model",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        }

        response = test_client.post("/v1/chat/completions", json=request_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."
        assert "usage" in data
        assert data["usage"]["total_tokens"] == 30

    def test_chat_completions_model_not_found(self, test_client, auth_headers):
        """Test POST /v1/chat/completions with non-existent model"""
        request_data = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = test_client.post("/v1/chat/completions", json=request_data, headers=auth_headers)

        assert response.status_code == 404
        assert "error" in response.json()

    def test_chat_completions_unauthorized(self, test_client):
        """Test POST /v1/chat/completions without authentication"""
        request_data = {
            "model": "test-mom-model",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = test_client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 401


class TestMetricsEndpoints:
    """Tests for metrics API endpoints"""

    def test_get_usage_metrics_success(self, test_client, auth_headers):
        """Test GET /v1/metrics/usage endpoint"""
        response = test_client.get("/v1/metrics/usage", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "metrics" in data
        assert "total_requests" in data["metrics"]

    def test_get_usage_metrics_with_filters(self, test_client, auth_headers):
        """Test GET /v1/metrics/usage with query parameters"""
        response = test_client.get(
            "/v1/metrics/usage?model_name=test-model&start_time=1000000000", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["model_name"] == "test-model"
        assert data["filters"]["start_time"] == 1000000000.0

    def test_get_usage_metrics_unauthorized(self, test_client):
        """Test GET /v1/metrics/usage without authentication"""
        response = test_client.get("/v1/metrics/usage")

        assert response.status_code == 401

    def test_get_raw_usage_metrics(self, test_client, auth_headers):
        """Test GET /v1/metrics/usage/raw endpoint"""
        response = test_client.get("/v1/metrics/usage/raw", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "records" in data
        assert isinstance(data["records"], list)
        assert "count" in data


# pylint: disable=too-few-public-methods
class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, test_client):
        """Test GET /health endpoint"""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRequestIDMiddleware:
    """Tests for request ID middleware"""

    def test_request_id_in_response_headers(self, test_client):
        """Test that X-Request-ID header is present in responses"""
        response = test_client.get("/health")

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID length with dashes

    def test_request_id_in_error_responses(self, test_client):
        """Test that X-Request-ID is present even in error responses"""
        response = test_client.get("/v1/models")  # Will fail due to no auth

        assert response.status_code == 401
        assert "X-Request-ID" in response.headers
