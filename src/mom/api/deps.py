"""The composition container and the FastAPI dependencies that read it from app.state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from mom.config.resolve import ResolvedCatalog
from mom.domain.metrics import MetricsReader, MetricsSink
from mom.domain.ports import Clock, IdFactory, LLMClient
from mom.runtime.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    """Everything the app needs, constructed once at startup (or injected in tests)."""

    settings: Settings
    catalog: ResolvedCatalog
    client: LLMClient
    clock: Clock
    ids: IdFactory
    metrics: MetricsSink | None = None
    metrics_reader: MetricsReader | None = None


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
