"""Application factory.

``create_app`` is the single place routers meet wiring. During Phase 1 it exposes only a
liveness endpoint; chat/responses/messages/models/metrics/progress routers are mounted here as
later phases land. The factory takes explicit dependencies so tests construct it with fakes and
no global state is required.
"""

from __future__ import annotations

from fastapi import FastAPI

from mom import __version__
from mom.runtime.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the MoM FastAPI application."""
    settings = settings or Settings()
    app = FastAPI(
        title="MoM — Mixture of Models",
        version=__version__,
    )
    app.state.settings = settings

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe — verifies the process is up, nothing more."""
        return {"status": "ok", "version": __version__}

    return app
