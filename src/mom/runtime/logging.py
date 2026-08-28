"""Structured logging setup.

``configure_logging`` is idempotent and never mutates global state at import time — it is
called explicitly from the app factory / CLI. Text renderer by default; JSON in production.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog


def configure_logging(
    *, level: str = "INFO", fmt: str = "text", stream: TextIO | None = None
) -> None:
    """Configure stdlib + structlog once. Safe to call more than once.

    ``stream`` defaults to stdout (structlog's own default). ``mom mcp`` passes stderr: on the
    stdio transport stdout carries JSON-RPC frames, and one log line written there corrupts the
    protocol rather than merely being noisy.
    """
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
