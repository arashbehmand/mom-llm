"""
Metrics API Endpoint

Provides REST API access to usage metrics and statistics.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .. import metrics_db
from ..auth import verify_bearer_token
from ..config import load_config

logger = logging.getLogger(__name__)

config = load_config()
metrics_router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])


@metrics_router.get("/usage")
async def get_usage_metrics(
    request: Request,
    start_time: Optional[float] = Query(
        None, description="Start timestamp (Unix time) for filtering metrics"
    ),
    end_time: Optional[float] = Query(
        None, description="End timestamp (Unix time) for filtering metrics"
    ),
    model_name: Optional[str] = Query(None, description="Filter by MoM model name"),
):
    """
    Get aggregated usage metrics.

    Returns comprehensive statistics about LLM usage including:
    - Total requests, tokens, and costs
    - Success/failure rates
    - Cache hit rates
    - Breakdown by call type (fanout vs concluding)
    - Breakdown by model

    Query parameters:
    - start_time: Filter metrics after this timestamp (Unix time)
    - end_time: Filter metrics before this timestamp (Unix time)
    - model_name: Filter by specific MoM model name

    Requires Bearer token authentication.
    """
    verify_bearer_token(request)

    try:
        aggregated_stats = metrics_db.get_aggregated_metrics(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
        )

        return {
            "status": "ok",
            "filters": {
                "start_time": start_time,
                "end_time": end_time,
                "model_name": model_name,
            },
            "metrics": aggregated_stats,
        }
    except Exception as e:
        logger.error(f"Error retrieving usage metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Error retrieving metrics: {str(e)}",
                "type": "internal_server_error",
            },
        )


# pylint: disable=too-many-arguments,too-many-positional-arguments
@metrics_router.get("/usage/raw")
async def get_raw_usage_metrics(
    request: Request,
    start_time: Optional[float] = Query(
        None, description="Start timestamp (Unix time) for filtering metrics"
    ),
    end_time: Optional[float] = Query(
        None, description="End timestamp (Unix time) for filtering metrics"
    ),
    model_name: Optional[str] = Query(None, description="Filter by MoM model name"),
    call_type: Optional[str] = Query(
        None, description="Filter by call type (fanout or concluding)"
    ),
    limit: int = Query(1000, description="Maximum number of records to return (default: 1000)"),
):
    """
    Get raw usage metrics records.

    Returns individual metric records with full details about each LLM call.
    Useful for detailed analysis and debugging.

    Query parameters:
    - start_time: Filter metrics after this timestamp (Unix time)
    - end_time: Filter metrics before this timestamp (Unix time)
    - model_name: Filter by specific MoM model name
    - call_type: Filter by call type ("fanout" or "concluding")
    - limit: Maximum number of records to return (default: 1000)

    Requires Bearer token authentication.
    """
    verify_bearer_token(request)

    try:
        raw_metrics = metrics_db.query_metrics(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            call_type=call_type,
            limit=limit,
        )

        return {
            "status": "ok",
            "filters": {
                "start_time": start_time,
                "end_time": end_time,
                "model_name": model_name,
                "call_type": call_type,
                "limit": limit,
            },
            "count": len(raw_metrics),
            "records": raw_metrics,
        }
    except Exception as e:
        logger.error(f"Error retrieving raw metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Error retrieving raw metrics: {str(e)}",
                "type": "internal_server_error",
            },
        )
