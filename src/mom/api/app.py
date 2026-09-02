"""Application factory and lifespan.

``create_app`` is the single place routers meet wiring. Pass a prebuilt ``Container`` (tests, with
fakes), or a resolved ``catalog`` (``serve_app``), or neither and let the lifespan resolve one.
Building the app has no import-time side effects and reads no files: everything it needs about
the config is handed to it.

``serve_app`` is the factory ``mom serve`` points uvicorn at. It resolves the config search path
and materializes secrets *once*, in the process that will serve — which is the child process
under ``--reload``, not the parent that parsed the flags — and hands the result to ``create_app``.
Keeping that out of ``create_app`` is what lets the app be built in a test without discovering,
reading, or applying anything from the developer's machine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from mom import __version__
from mom.api.deps import Container
from mom.api.errors import install_error_handlers
from mom.api.mcp import mount_mcp, serve_mcp
from mom.api.middleware import BudgetAlarmMiddleware
from mom.api.routers.anthropic import router as anthropic_router
from mom.api.routers.chat import router as chat_router
from mom.api.routers.metrics import router as metrics_router
from mom.api.routers.models import router as models_router
from mom.api.routers.progress import router as progress_router
from mom.api.routers.responses import router as responses_router
from mom.config.resolve import ResolvedCatalog
from mom.runtime.discovery import ConfigSources
from mom.runtime.logging import configure_logging, get_logger
from mom.runtime.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
    catalog: ResolvedCatalog | None = None,
    sources: ConfigSources | None = None,
    warnings: Sequence[str] = (),
) -> FastAPI:
    """Build the MoM FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The MCP server is built HERE rather than in create_app's body because its session
        # manager may only be run once per instance: an app whose lifespan is entered a second
        # time (an embedder restarting it, two consecutive TestClient blocks) needs a fresh one,
        # not the one the previous run consumed. Serving /mcp needs the manager's task group even
        # in stateless mode, and a sub-app's own lifespan is never run by the parent.
        async with serve_mcp(app):
            if container is not None:
                app.state.container = container
                yield
                return
            from mom.runtime.wiring import build_container

            # Here, not in create_app's body: the lifespan runs in every real serving process
            # (uvicorn factory mode, --reload children, direct ASGI) while tests with a prebuilt
            # container return above and never mutate global logging state. Before
            # build_container, so its startup catalog warnings come out formatted.
            run_settings, resolved, resolved_sources, found = _resolve(settings, catalog, sources)
            configure_logging(level=run_settings.log_level, fmt=run_settings.log_format)
            for warning in (*warnings, *found):
                get_logger("mom.config").warning("config discovery", detail=warning)
            built, cleanup = await build_container(run_settings, resolved, sources=resolved_sources)
            app.state.container = built
            try:
                yield
            finally:
                await cleanup()

    app = FastAPI(title="MoM — Mixture of Models", version=__version__, lifespan=lifespan)
    if container is not None:  # tests: make the container available without the lifespan
        app.state.container = container

    # Install CORS from the catalog. Middleware has to be added before the app is first called
    # (Starlette builds the stack then and `add_middleware` raises afterwards), so this cannot
    # wait for the lifespan — which is why the catalog is a parameter. It used to be re-loaded
    # from `settings.config_file` right here, a second read that dropped `MOM_CONFIG_OVERLAY`
    # and so could disagree with the catalog the lifespan went on to serve.
    cors = None
    if container is not None:
        cors = container.catalog.config.server.cors
    elif catalog is not None:
        cors = catalog.config.server.cors
    if cors is not None and cors.origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors.origins),
            allow_credentials=cors.allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            # A browser MCP client reads its session id off the response; without this the
            # header is invisible to it even though the request itself succeeded.
            expose_headers=["Mcp-Session-Id"],
        )

    # Soft daily-budget alarm (adds a header when over budget; never blocks). Self-guards when no
    # budget is configured, so it is safe to install unconditionally.
    app.add_middleware(BudgetAlarmMiddleware)

    install_error_handlers(app)

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, object]:
        body: dict[str, object] = {"status": "ok", "version": __version__}
        # `getattr` all the way down (not a MetricsSink.dropped protocol member — most
        # implementations, e.g. a test fake, don't have one) so a missing container/metrics/
        # attribute just omits the field rather than 500ing the health check.
        running_container = getattr(request.app.state, "container", None)
        metrics = getattr(running_container, "metrics", None)
        dropped = getattr(metrics, "dropped", None)
        if dropped is not None:
            body["metrics_dropped"] = dropped
        return body

    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(anthropic_router, prefix="/v1")
    app.include_router(responses_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    app.include_router(progress_router, prefix="/v1")
    # Not under /v1: MCP is its own protocol at its own well-known path, not another wire
    # dialect of the model endpoint. Gated per request by `server.mcp.enabled` (see McpGate).
    mount_mcp(app)
    return app


def _resolve(
    settings: Settings | None,
    catalog: ResolvedCatalog | None,
    sources: ConfigSources | None,
) -> tuple[Settings, ResolvedCatalog, ConfigSources | None, tuple[str, ...]]:
    """What `serve_app` already resolved, or a resolution done now.

    The fallback exists for `uvicorn mom.api.app:create_app --factory` and for library embedders,
    both of which reach the lifespan without a catalog. It resolves here rather than at
    construction, so building the app stays free of I/O.

    Two things it must get right, and both used to be wrong:

    * **A caller's `Settings` is an instruction, not decoration.** `create_app(Settings(
      config_file=X))` has to serve X. Bootstrapping bare would re-derive the pin from the
      environment and quietly serve the discovered stack instead, while still reporting X as
      `container.settings.config_file`.
    * **The bootstrapped `Settings` are adopted, not discarded.** They carry what the discovered
      `.env` files defined, so keeping an env-only `Settings` would mean `MOM_API_TOKEN` in
      `~/.mom/.env` authenticated under `mom serve` and nowhere else.

    A supplied `catalog` is the third case, and it means the caller has taken over config
    resolution entirely. Discovery stays off: an embedder handing in a catalog it built itself
    does not expect mom to go reading `$HOME` for a data directory, an API token, or a Redis URL.
    Its process environment still configures it, because that is how any library is configured.
    """
    if catalog is not None:
        if settings is not None:
            return settings, catalog, sources, ()
        return Settings(), catalog, sources, ()

    from mom.runtime.bootstrap import bootstrap

    booted = bootstrap(
        config=settings.config_file if settings else None,
        overlay=settings.config_overlay if settings else None,
        data_dir=settings.data_dir if settings else None,
        auth_from_opencode=bool(settings and settings.auth_from_opencode),
    )
    return booted.settings, booted.catalog(), booted.sources, booted.warnings


def serve_app() -> FastAPI:
    """The factory `mom serve` runs: resolve config and secrets once, then build the app.

    Deliberately does not configure logging or emit anything. Building an app must not reach
    global state — structlog is process-wide, so a factory that reconfigured it would change the
    behaviour of everything else in the process. The warnings it collected ride along and are
    logged in the lifespan, once the sink is set up.
    """
    from mom.runtime.bootstrap import bootstrap

    booted = bootstrap()
    return create_app(
        booted.settings,
        catalog=booted.catalog(),
        sources=booted.sources,
        warnings=booted.warnings,
    )
