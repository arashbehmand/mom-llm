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
from mom.api.routers.chat import models_router
from mom.api.routers.chat import router as chat_router
from mom.runtime.settings import Settings


def _build_container(settings: Settings) -> Container:
    from mom.adapters.litellm_client import LiteLLMClient
    from mom.config.loader import load_config
    from mom.runtime.clock import SystemClock, UuidIds

    if settings.config_file is None:
        raise RuntimeError("MOM_CONFIG must point to a config file to serve")
    catalog = load_config(settings.config_file)
    return Container(
        settings=settings,
        catalog=catalog,
        client=LiteLLMClient(),
        clock=SystemClock(),
        ids=UuidIds(),
    )


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    """Build the MoM FastAPI application."""
    settings = settings or (container.settings if container else Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or _build_container(settings)
        yield

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
    return app
