"""FastAPI dependencies that read the composition :class:`Container` from ``app.state``.

The :class:`Container` dataclass itself lives in :mod:`mom.runtime.container` (so the composition
root can build it without a ``runtime -> api`` import); it is re-exported here for the routers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mom.runtime.container import Container


__all__ = ["Container", "ContainerDep", "get_container"]


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
