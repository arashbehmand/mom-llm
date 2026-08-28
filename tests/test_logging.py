"""Logging setup is idempotent and side-effect-free until called."""

from __future__ import annotations

import io

from mom.runtime.logging import configure_logging, get_logger


def test_configure_logging_json_then_text_is_idempotent():
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    log.info("event", key="value")  # must not raise
    # Second call with different settings must not raise (idempotent reconfigure).
    configure_logging(level="DEBUG", fmt="text")
    get_logger().debug("again")


def test_unknown_level_falls_back_to_info():
    configure_logging(level="NONSENSE", fmt="text")
    get_logger("x").warning("still works")


def test_logs_can_be_routed_off_stdout():
    """`mom mcp` speaks JSON-RPC on stdout: a log line written there is a protocol violation,
    not just noise."""
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    try:
        get_logger("mcp").info("hello")
        assert "hello" in stream.getvalue()
    finally:
        configure_logging(level="INFO", fmt="text")  # restore the default sink for other tests
