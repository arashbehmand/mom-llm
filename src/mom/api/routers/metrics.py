"""``GET /v1/metrics/usage`` — aggregated usage and cost."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/metrics/usage")
async def usage(
    container: ContainerDep,
    start: float | None = None,
    end: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    if container.metrics_reader is None:
        return {"calls": 0}
    return await container.metrics_reader.aggregate(start=start, end=end, ensemble=model)
