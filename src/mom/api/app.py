"""Application factory and lifespan.

``create_app`` is the single place routers meet wiring. Pass a prebuilt ``Container`` (tests, with
fakes) or let the lifespan construct one from ``Settings`` (load config, LiteLLM client, clock).
Building the app has no import-time side effects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mom import __version__
from mom.api.deps import Container
from mom.api.errors import install_error_handlers
from mom.api.routers.anthropic import router as anthropic_router
from mom.api.routers.chat import router as chat_router
from mom.api.routers.metrics import router as metrics_router
from mom.api.routers.models import router as models_router
from mom.api.routers.responses import router as responses_router
from mom.runtime.settings import Settings


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    """Build the MoM FastAPI application."""
    settings = settings or (container.settings if container else Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is not None:
            app.state.container = container
            yield
            return
        from mom.runtime.wiring import build_container

        built, cleanup = await build_container(settings)
        app.state.container = built
        try:
            yield
        finally:
            await cleanup()

    app = FastAPI(title="MoM — Mixture of Models", version=__version__, lifespan=lifespan)
    if container is not None:  # tests: make the container available without the lifespan
        app.state.container = container

    install_error_handlers(app)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(anthropic_router, prefix="/v1")
    app.include_router(responses_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    return app
