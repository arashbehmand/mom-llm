"""Serving the same tools over stdio, for a local MCP client (behind ``mom mcp``).

Builds its own container from the same config and data dir the gateway uses, so it needs no
running server: consults it runs are recorded in the same metrics.db and warm the same cache.

Unauthenticated by design — there are no headers on a pipe, and a process the operator started
is already inside the trust boundary, exactly like the rest of the CLI. ``server.mcp.enabled``
gates the network surface only; running this command *is* the opt-in.
"""

from __future__ import annotations

from pathlib import Path
import sys

from mom.api.mcp.server import build_mcp_server
from mom.runtime.settings import Settings


async def run_stdio(*, config: Path | None = None, data_dir: Path | None = None) -> None:
    """Serve the MCP tools on stdin/stdout until the client disconnects."""
    from mom.runtime.logging import configure_logging
    from mom.runtime.wiring import build_container

    settings = _settings(config, data_dir)
    # stdout carries JSON-RPC frames here: a log line written there is not noise, it is a
    # protocol violation that disconnects the client.
    configure_logging(level=settings.log_level, fmt=settings.log_format, stream=sys.stderr)
    container, cleanup = await build_container(settings)
    try:
        server = build_mcp_server(lambda: container)
        await server.run_stdio_async()
    finally:
        await cleanup()


def _settings(config: Path | None, data_dir: Path | None) -> Settings:
    """Env-derived settings with the CLI's overrides applied.

    ``model_copy`` rather than keyword construction: these fields carry validation aliases
    (``MOM_CONFIG``/``MOM_CONFIG_PATH``), so constructing by field name would not bind them.
    """
    settings = Settings()
    overrides: dict[str, object] = {}
    if config is not None:
        overrides["config_file"] = str(config)
    if data_dir is not None:
        overrides["data_dir"] = str(data_dir)
    return settings.model_copy(update=overrides) if overrides else settings
