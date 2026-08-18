"""``GET /v1/metrics/usage`` — aggregated usage and cost (optionally grouped)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep


router = APIRouter(dependencies=[Depends(require_api_key)])

GroupBy = Literal["member", "turn_type", "day", "ensemble", "status"]


@router.get("/metrics/usage")
async def usage(
    container: ContainerDep,
    start: float | None = None,
    end: float | None = None,
    model: str | None = None,
    by: Annotated[GroupBy | None, Query(description="Group rows by this dimension.")] = None,
) -> dict[str, Any]:
    if container.metrics_reader is None:
        return {"calls": 0}
    if by is not None:
        groups = await container.metrics_reader.aggregate_by(
            by, start=start, end=end, ensemble=model
        )
        return {"by": by, "groups": groups}
    return await container.metrics_reader.aggregate(start=start, end=end, ensemble=model)
