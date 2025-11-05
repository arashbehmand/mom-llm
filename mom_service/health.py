"""
Health check utilities for MoM Service
"""

import logging
import os
import sqlite3
import time
from typing import Any

import litellm
from litellm.utils import ModelResponse

from . import metrics_db
from .config import MoMConfig

logger = logging.getLogger(__name__)


async def check_database_health(db_path: str, db_name: str) -> dict[str, Any]:
    """
    Check SQLite database connectivity and basic health.

    Args:
        db_path: Path to the SQLite database file
        db_name: Friendly name for the database (e.g., "cache", "metrics")

    Returns:
        Dict with status and details
    """
    try:
        start_time = time.time()

        # Check if file exists
        if not os.path.exists(db_path):
            return {
                "status": "unavailable",
                "name": db_name,
                "error": "Database file does not exist",
                "path": db_path,
            }

        # Try to connect and execute a simple query
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

            # Get database size
            cursor.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            size_bytes = cursor.fetchone()[0]

        duration_ms = (time.time() - start_time) * 1000

        return {
            "status": "healthy",
            "name": db_name,
            "path": db_path,
            "size_bytes": size_bytes,
            "response_time_ms": round(duration_ms, 2),
        }
    except Exception as e:
        logger.error(f"Database health check failed for {db_name}: {e}")
        return {"status": "unhealthy", "name": db_name, "error": str(e), "path": db_path}


async def check_llm_connectivity(config: MoMConfig, timeout: int = 10) -> dict[str, Any]:
    """
    Test LLM connectivity by attempting a simple completion call.
    Uses the first available LLM definition from config.

    Args:
        config: MoM configuration
        timeout: Timeout in seconds for the test call

    Returns:
        Dict with status and details
    """
    if not config.llm_definitions:
        return {"status": "skipped", "reason": "No LLM definitions configured"}

    # Use the first LLM definition for testing
    test_llm = config.llm_definitions[0]

    try:
        start_time = time.time()

        # Make a minimal completion call; guard against params being None
        response: ModelResponse = await litellm.acompletion(
            model=test_llm.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            timeout=timeout,
            **(test_llm.params or {}),
        )

        duration_ms = (time.time() - start_time) * 1000

        # Check if response is valid
        if response and response.choices and len(response.choices) > 0:
            return {
                "status": "healthy",
                "llm_tested": test_llm.name,
                "model": test_llm.model,
                "response_time_ms": round(duration_ms, 2),
            }
        return {
            "status": "unhealthy",
            "llm_tested": test_llm.name,
            "model": test_llm.model,
            "error": "Empty or invalid response from LLM",
        }

    except Exception as e:
        logger.error(f"LLM connectivity check failed for {test_llm.name}: {e}")
        return {
            "status": "unhealthy",
            "llm_tested": test_llm.name,
            "model": test_llm.model,
            "error": str(e)[:200],  # Truncate long error messages
        }


async def perform_comprehensive_health_check(
    config: MoMConfig, check_llm: bool = False
) -> dict[str, Any]:
    """
    Perform a comprehensive health check of all system components.

    Args:
        config: MoM configuration
        check_llm: Whether to perform LLM connectivity check (slower)

    Returns:
        Dict with overall status and component details
    """
    health_data = {"status": "healthy", "timestamp": time.time(), "checks": {}}

    # Check cache database
    cache_db_path = os.path.join(os.path.dirname(__file__), "llm_cache.db")
    cache_health = await check_database_health(cache_db_path, "cache")
    health_data["checks"]["cache_db"] = cache_health

    if cache_health["status"] != "healthy":
        health_data["status"] = "degraded"

    # Check metrics database (use canonical path from metrics_db module)
    metrics_db_path = metrics_db.METRICS_DB_PATH
    metrics_health = await check_database_health(metrics_db_path, "metrics")
    health_data["checks"]["metrics_db"] = metrics_health

    if metrics_health["status"] != "healthy":
        health_data["status"] = "degraded"

    # Check configuration
    try:
        config_check = {
            "status": "healthy",
            "models_configured": len(config.models),
            "llms_configured": len(config.llm_definitions),
            "cache_enabled": config.service.cache_enabled,
        }
    except Exception as e:
        config_check = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "unhealthy"

    health_data["checks"]["configuration"] = config_check

    # Optional LLM connectivity check
    if check_llm:
        llm_health = await check_llm_connectivity(config)
        health_data["checks"]["llm_connectivity"] = llm_health

        if llm_health["status"] == "unhealthy":
            health_data["status"] = "degraded"

    return health_data
