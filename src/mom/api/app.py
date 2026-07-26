"""Application factory and lifespan.

``create_app`` is the single place routers meet wiring. Pass a prebuilt ``Container`` (tests, with
fakes) or let the lifespan construct one from ``Settings`` (load config, LiteLLM client, clock).
Building the app has no import-time side effects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from mom import __version__
from mom.api.deps import Container
from mom.api.errors import install_error_handlers
from mom.api.middleware import BudgetAlarmMiddleware
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

    # Install CORS from the catalog config. A prebuilt container (tests) exposes it directly;
    # otherwise the container is built later in the lifespan, so best-effort load the config file
    # here. Any failure leaves CORS off — a bad config must never break the (pure) app build.
    cors = None
    if container is not None:
        cors = container.catalog.config.server.cors
    elif settings.config_file is not None:
        try:
            from mom.config.loader import load_config

            cors = load_config(settings.config_file).config.server.cors
        except Exception:
            cors = None
    if cors is not None and cors.origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors.origins),
            allow_credentials=cors.allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Soft daily-budget alarm (adds a header when over budget; never blocks). Self-guards when no
    # budget is configured, so it is safe to install unconditionally.
    app.add_middleware(BudgetAlarmMiddleware)

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
