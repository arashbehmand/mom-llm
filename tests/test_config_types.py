"""Config primitives: durations, byte sizes, effort parsing, nearest-tier, loader edges."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from mom.config.loader import load_config, load_raw
from mom.config.resolve import ConfigError
from mom.config.types import (
    EffortLevel,
    nearest_tier,
    normalize_effort_cell,
    parse_bytes,
    parse_duration,
    parse_effort_level,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("500ms", 0.5), ("2s", 2), ("20m", 1200), ("1h", 3600), ("14d", 1_209_600)],
)
def test_parse_duration(text: str, seconds: float):
    assert parse_duration(text) == timedelta(seconds=seconds)


def test_parse_duration_passthrough_and_errors():
    assert parse_duration(timedelta(seconds=5)) == timedelta(seconds=5)
    assert parse_duration(90) == timedelta(seconds=90)
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("soon")
    with pytest.raises(TypeError):
        parse_duration(["nope"])


@pytest.mark.parametrize(
    ("text", "value"),
    [("512KB", 512 * 1024), ("1GB", 1024**3), ("2MB", 2 * 1024**2), ("10B", 10)],
)
def test_parse_bytes(text: str, value: int):
    assert parse_bytes(text) == value


def test_parse_bytes_passthrough_and_errors():
    assert parse_bytes(1234) == 1234
    with pytest.raises(ValueError, match="invalid size"):
        parse_bytes("huge")


def test_effort_parsing():
    assert parse_effort_level("h") is EffortLevel.HIGH
    assert parse_effort_level("MAX") is EffortLevel.MAX
    assert normalize_effort_cell("m") == "medium"
    assert normalize_effort_cell("pass") == "pass"
    assert normalize_effort_cell(EffortLevel.XHIGH) == "xhigh"
    with pytest.raises(ValueError, match="invalid effort level"):
        parse_effort_level("banana")
    with pytest.raises(ValueError, match="invalid effort cell"):
        normalize_effort_cell("banana")


def test_nearest_tier_rounds_up_on_ties():
    tiers = [EffortLevel.LOW, EffortLevel.HIGH]
    # MEDIUM is equidistant from LOW and HIGH -> rounds up to HIGH.
    assert nearest_tier(EffortLevel.MEDIUM, tiers) is EffortLevel.HIGH
    # exact match returns itself
    assert nearest_tier(EffortLevel.LOW, tiers) is EffortLevel.LOW
    # above range clamps to the max defined
    assert nearest_tier(EffortLevel.MAX, tiers) is EffortLevel.HIGH
    # below range clamps to the min defined
    assert nearest_tier(EffortLevel.NONE, tiers) is EffortLevel.LOW


def test_duration_serializes_back_to_string():
    from mom.config.schema import CallDefaults

    dumped = CallDefaults().model_dump(mode="json")
    assert dumped["timeout"] == "20m"
    assert dumped["retry_backoff"] == "2s"


# ---- loader edges ----------------------------------------------------------------------------
def test_loader_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_loader_invalid_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 2\nllms: [oops\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_loader_non_mapping_top_level(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(bad)


def test_loader_layering_overlay(tmp_path: Path):
    base = tmp_path / "models.yaml"
    base.write_text(
        "version: 2\n"
        "llms: { a: { model: openai/x } }\n"
        "ensembles: { e: { members: [{ llm: a }], synthesizer: { llm: a } } }\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "local.yaml"
    overlay.write_text("server: { auth: none }\n", encoding="utf-8")
    merged = load_raw(base, overlay=overlay)
    assert merged["server"]["auth"] == "none"
    catalog = load_config(base, overlay=overlay)
    assert catalog.config.server.auth == "none"
