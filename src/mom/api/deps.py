"""FastAPI access to the composition ``Container`` stored on ``app.state``.

The ``Container`` itself lives in ``mom.runtime.container`` (the runtime layer owns composition,
so nothing in ``runtime`` has to import ``api``); this module adds the FastAPI dependency and
re-exports ``Container`` for the routers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mom.runtime.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


__all__ = ["Container", "ContainerDep", "get_container"]
