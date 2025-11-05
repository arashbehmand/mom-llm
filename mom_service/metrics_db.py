"""
Metrics Database Module

This module provides persistent storage for LLM call metrics using SQLite.
It tracks token usage, costs, duration, and status for all LLM calls.
"""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# SQLite database file path
METRICS_DB_PATH = os.path.join(os.path.dirname(__file__), "usage_metrics.db")
logger.info(f"Metrics DB path: {METRICS_DB_PATH}")


def _init_metrics_db():
    """Initialize the SQLite database and create the usage_metrics table if it doesn't exist."""
    try:
        with sqlite3.connect(METRICS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    mom_model_name TEXT NOT NULL,
                    llm_name TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost REAL,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    cache_hit BOOLEAN DEFAULT 0
                )
                """
            )
            # Create index on request_id for faster lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_id
                ON usage_metrics(request_id)
                """
            )
            # Create index on timestamp for time-range queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON usage_metrics(timestamp)
                """
            )
            # Create index on mom_model_name for model-specific queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mom_model_name
                ON usage_metrics(mom_model_name)
                """
            )
            conn.commit()
        logger.info(f"SQLite metrics database initialized at {METRICS_DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing SQLite metrics database: {e}", exc_info=True)


# Public alias for tests / external callers to avoid protected-access lint warnings
init_metrics_db = _init_metrics_db


# pylint: disable=too-many-instance-attributes
@dataclass
class MetricRecord:
    request_id: str
    mom_model_name: str
    llm_name: str
    call_type: str  # "fanout" or "concluding"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "SUCCESS"  # SUCCESS, FAILED, CACHED
    error_message: Optional[str] = None
    cache_hit: bool = False


@dataclass
class Timestamps:
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None


def insert_metric_record(record: MetricRecord) -> None:
    """
    Insert a MetricRecord into the usage_metrics table.

    Args:
        record: MetricRecord instance containing all metric fields.
    """
    try:
        with sqlite3.connect(METRICS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO usage_metrics (
                    request_id, timestamp, mom_model_name, llm_name, call_type,
                    prompt_tokens, completion_tokens, total_tokens, cost, duration_ms,
                    status, error_message, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    time.time(),
                    record.mom_model_name,
                    record.llm_name,
                    record.call_type,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.cost,
                    record.duration_ms,
                    record.status,
                    record.error_message,
                    1 if record.cache_hit else 0,
                ),
            )
            conn.commit()
            logger.debug(
                f"Inserted metric record: request_id={record.request_id}, llm={record.llm_name}, "
                f"type={record.call_type}, status={record.status}, tokens={record.total_tokens}, cost=${record.cost}"
            )
    except Exception as e:
        logger.error(
            f"Error inserting metric record for request_id={getattr(record, 'request_id', None)}: {e}",
            exc_info=True,
        )


def query_metrics(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    model_name: Optional[str] = None,
    call_type: Optional[str] = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Query metrics from the database.

    Args:
        start_time: Optional start timestamp (Unix time)
        end_time: Optional end timestamp (Unix time)
        model_name: Optional filter by MoM model name
        call_type: Optional filter by call type ("fanout" or "concluding")
        limit: Maximum number of records to return

    Returns:
        List of metric records as dictionaries
    """
    try:
        with sqlite3.connect(METRICS_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM usage_metrics WHERE 1=1"
            params = []

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if model_name is not None:
                query += " AND mom_model_name = ?"
                params.append(model_name)

            if call_type is not None:
                query += " AND call_type = ?"
                params.append(call_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error querying metrics: {e}", exc_info=True)
        return []


def get_aggregated_metrics(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get aggregated statistics from the metrics database.

    Args:
        start_time: Optional start timestamp (Unix time)
        end_time: Optional end timestamp (Unix time)
        model_name: Optional filter by MoM model name

    Returns:
        Dictionary with aggregated statistics
    """
    try:
        with sqlite3.connect(METRICS_DB_PATH) as conn:
            cursor = conn.cursor()

            # Build WHERE clause
            where_clause = "WHERE 1=1"
            params = []

            if start_time is not None:
                where_clause += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                where_clause += " AND timestamp <= ?"
                params.append(end_time)

            if model_name is not None:
                where_clause += " AND mom_model_name = ?"
                params.append(model_name)

            # Get overall statistics
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cached_requests,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost) as total_cost,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(cost) as avg_cost
                FROM usage_metrics
                {where_clause}
                """,
                params,
            )

            overall_stats = cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cached_requests,
                    COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0.0) as total_cost,
                    COALESCE(AVG(duration_ms), 0.0) as avg_duration_ms,
                    COALESCE(AVG(cost), 0.0) as avg_cost
                FROM usage_metrics
                {where_clause}
                """,
                params,
            ).fetchone()

            # Get stats by call type
            cursor.execute(
                f"""
                SELECT
                    call_type,
                    COUNT(*) as count,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0.0) as total_cost,
                    COALESCE(AVG(duration_ms), 0.0) as avg_duration_ms
                FROM usage_metrics
                {where_clause}
                GROUP BY call_type
                """,
                params,
            )

            by_call_type = {}
            for row in cursor.fetchall():
                by_call_type[row[0]] = {
                    "count": row[1],
                    "total_tokens": row[2],
                    "total_cost": row[3],
                    "avg_duration_ms": row[4],
                }

            # Get stats by model
            cursor.execute(
                f"""
                SELECT
                    mom_model_name,
                    COUNT(*) as count,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0.0) as total_cost
                FROM usage_metrics
                {where_clause}
                GROUP BY mom_model_name
                """,
                params,
            )

            by_model = {}
            for row in cursor.fetchall():
                by_model[row[0]] = {
                    "count": row[1],
                    "total_tokens": row[2],
                    "total_cost": row[3],
                }

            # Calculate cache hit rate
            total_reqs = overall_stats[0] or 0
            cached_reqs = overall_stats[3] or 0
            cache_hit_rate = (cached_reqs / total_reqs * 100) if total_reqs > 0 else 0.0

            return {
                "total_requests": overall_stats[0] or 0,
                "successful_requests": overall_stats[1] or 0,
                "failed_requests": overall_stats[2] or 0,
                "cached_requests": cached_reqs,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "total_prompt_tokens": overall_stats[4] or 0,
                "total_completion_tokens": overall_stats[5] or 0,
                "total_tokens": overall_stats[6] or 0,
                "total_cost_usd": round(overall_stats[7] or 0.0, 6),
                "avg_duration_ms": round(overall_stats[8] or 0.0, 2),
                "avg_cost_usd": round(overall_stats[9] or 0.0, 6),
                "by_call_type": by_call_type,
                "by_model": by_model,
            }
    except Exception as e:
        logger.error(f"Error getting aggregated metrics: {e}", exc_info=True)
        return {
            "error": str(e),
            "total_requests": 0,
            "total_cost_usd": 0.0,
        }
