"""
Tests for health check endpoints and utilities
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mom_service.health import (
    check_database_health,
    check_llm_connectivity,
    perform_comprehensive_health_check,
)


class TestDatabaseHealth:
    """Tests for database health check function"""

    @pytest.mark.asyncio
    async def test_check_database_health_success(self, temp_metrics_db):
        """Test successful database health check"""
        result = await check_database_health(temp_metrics_db, "test_db")

        assert result["status"] == "healthy"
        assert result["name"] == "test_db"
        assert "size_bytes" in result
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_database_health_nonexistent(self):
        """Test health check with nonexistent database"""
        result = await check_database_health("/nonexistent/path/db.db", "missing_db")

        assert result["status"] == "unavailable"
        assert result["name"] == "missing_db"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_check_database_health_corrupted(self, tmp_path):
        """Test health check with corrupted database file"""
        # Create a corrupted database file
        corrupted_db = tmp_path / "corrupted.db"
        corrupted_db.write_text("not a valid sqlite database")

        result = await check_database_health(str(corrupted_db), "corrupted_db")

        assert result["status"] == "unhealthy"
        assert "error" in result


class TestLLMConnectivity:
    """Tests for LLM connectivity check function"""

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_success(self, sample_mom_config, mock_litellm_response):
        """Test successful LLM connectivity check"""
        with patch(
            "mom_service.health.litellm.acompletion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.return_value = mock_litellm_response

            result = await check_llm_connectivity(sample_mom_config, timeout=10)

            assert result["status"] == "healthy"
            assert "llm_tested" in result
            assert "model" in result
            assert "response_time_ms" in result
            mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_failure(self, sample_mom_config):
        """Test LLM connectivity check with API failure"""
        with patch(
            "mom_service.health.litellm.acompletion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.side_effect = Exception("API connection failed")

            result = await check_llm_connectivity(sample_mom_config, timeout=10)

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "API connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_empty_response(self, sample_mom_config):
        """Test LLM connectivity check with empty response"""
        with patch(
            "mom_service.health.litellm.acompletion", new_callable=AsyncMock
        ) as mock_completion:
            # Create a response with no choices
            empty_response = MagicMock()
            empty_response.choices = []
            mock_completion.return_value = empty_response

            result = await check_llm_connectivity(sample_mom_config, timeout=10)

            assert result["status"] == "unhealthy"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_no_llms_configured(self):
        """Test LLM connectivity check when no LLMs are configured"""
        from mom_service.config import MoMConfig, ServiceConfig

        empty_config = MoMConfig(llm_definitions=[], models=[], service=ServiceConfig())

        result = await check_llm_connectivity(empty_config)

        assert result["status"] == "skipped"
        assert "reason" in result


class TestComprehensiveHealthCheck:
    """Tests for comprehensive health check function"""

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_without_llm(self, sample_mom_config, temp_metrics_db):
        """Test comprehensive health check without LLM connectivity test"""
        with patch("mom_service.health.os.path.join") as mock_join:
            # Mock database paths to use our test database
            mock_join.side_effect = lambda *args: (
                temp_metrics_db if "metrics" in str(args) else temp_metrics_db
            )

            result = await perform_comprehensive_health_check(sample_mom_config, check_llm=False)

            assert "status" in result
            assert "timestamp" in result
            assert "checks" in result
            assert "cache_db" in result["checks"]
            assert "metrics_db" in result["checks"]
            assert "configuration" in result["checks"]
            assert "llm_connectivity" not in result["checks"]

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_with_llm(
        self, sample_mom_config, temp_metrics_db, mock_litellm_response
    ):
        """Test comprehensive health check with LLM connectivity test"""
        with (
            patch("mom_service.health.os.path.join") as mock_join,
            patch(
                "mom_service.health.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion,
        ):
            mock_join.side_effect = lambda *args: temp_metrics_db
            mock_completion.return_value = mock_litellm_response

            result = await perform_comprehensive_health_check(sample_mom_config, check_llm=True)

            assert "status" in result
            assert "llm_connectivity" in result["checks"]
            assert result["checks"]["llm_connectivity"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_degraded_status(self, sample_mom_config):
        """Test that degraded status is returned when a component is unhealthy"""
        with patch(
            "mom_service.health.check_database_health", new_callable=AsyncMock
        ) as mock_db_check:
            # Simulate unhealthy cache database
            mock_db_check.return_value = {
                "status": "unhealthy",
                "name": "cache",
                "error": "Connection failed",
            }

            result = await perform_comprehensive_health_check(sample_mom_config, check_llm=False)

            assert result["status"] == "degraded"


class TestHealthEndpoints:
    """Tests for health check HTTP endpoints"""

    @pytest.fixture
    def test_client(self):
        """Fixture providing a test client for the FastAPI app"""
        from fastapi.testclient import TestClient

        from mom_service.main import app

        return TestClient(app)

    def test_basic_health_endpoint(self, test_client):
        """Test GET /health endpoint"""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_detailed_health_endpoint(self, test_client):
        """Test GET /health/detailed endpoint"""
        response = test_client.get("/health/detailed")

        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data

    def test_detailed_health_with_llm_check(self, test_client):
        """Test GET /health/detailed with LLM check parameter"""
        with patch(
            "mom_service.main.perform_comprehensive_health_check", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": 1234567890,
                "checks": {"llm_connectivity": {"status": "healthy"}},
            }

            response = test_client.get("/health/detailed?check_llm=true")

            assert response.status_code == 200
            # Verify that check_llm parameter was passed
            mock_health.assert_called_once()
            call_args = mock_health.call_args
            assert call_args.kwargs.get("check_llm") is True
