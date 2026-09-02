"""Serving the same tools over stdio, for a local MCP client (behind ``mom mcp``).

Builds its own container from the same config and data dir the gateway uses, so it needs no
running server: consults it runs are recorded in the same metrics.db and warm the same cache.

This is the entry point config discovery exists for. An MCP client launches it from whatever
directory it happens to be in, with an environment stripped down to almost nothing, so a client
entry used to have to carry ``--config /abs/path`` and its own ``env`` block. Resolution now goes
through the same search path every other command uses, and the entry is just
``{"command": "mom", "args": ["mcp"]}``.

Unauthenticated by design — there are no headers on a pipe, and a process the operator started
is already inside the trust boundary, exactly like the rest of the CLI. ``server.mcp.enabled``
gates the network surface only; running this command *is* the opt-in.
"""

from __future__ import annotations

from pathlib import Path
import sys

from mom.api.mcp.server import build_mcp_server


async def run_stdio(
    *,
    config: Path | None = None,
    data_dir: Path | None = None,
    overlay: Path | None = None,
    auth_from_opencode: bool = False,
) -> None:
    """Serve the MCP tools on stdin/stdout until the client disconnects."""
    from mom.runtime.bootstrap import bootstrap
    from mom.runtime.logging import configure_logging, get_logger
    from mom.runtime.wiring import build_container

    booted = bootstrap(
        config=config,
        overlay=overlay,
        data_dir=data_dir,
        auth_from_opencode=auth_from_opencode,
    )
    # stdout carries JSON-RPC frames here: a log line written there is not noise, it is a
    # protocol violation that disconnects the client. Discovery therefore reports its warnings
    # as data and they are emitted here, once the sink is pointed at stderr — never from inside
    # the resolver, which has no idea which process it is running in.
    configure_logging(
        level=booted.settings.log_level, fmt=booted.settings.log_format, stream=sys.stderr
    )
    for warning in booted.warnings:
        get_logger("mom.config").warning("config discovery", detail=warning)

    container, cleanup = await build_container(
        booted.settings, booted.catalog(), sources=booted.sources
    )
    try:
        server = build_mcp_server(lambda: container)
        await server.run_stdio_async()
    finally:
        await cleanup()
