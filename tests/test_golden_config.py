"""Characterization golden: v1 config expansion.

This test freezes what ``mom_service.config.load_config`` produces for a snapshot of the
owner's real ``config.yaml`` (base + variant expansion, ``base+alias`` materialization,
reference validation, service/langfuse blocks). The resulting normalized JSON projection is
the *specification* the v2 config loader and the one-time migration script must reproduce.

It deliberately loads a FROZEN snapshot (``tests/golden/fixtures/live_config_snapshot.yaml``),
not the live ``config.yaml``, so editing the live config never breaks this test — unlike v1's
``test_default_config_uses_current_verified_model_matrix``, which coupled to the live model set.

Regenerate after an intentional change to expansion semantics::

    REGEN_GOLDENS=1 .venv/bin/python -m pytest tests/test_golden_config.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mom_service.config import load_config

_HERE = Path(__file__).parent
SNAPSHOT = _HERE / "golden" / "fixtures" / "live_config_snapshot.yaml"
GOLDEN = _HERE / "golden" / "config_expansion.json"


def project_config(config_path: Path) -> dict[str, Any]:
    """Load a config file and return a deterministic, normalized projection of its expansion.

    Top-level named lists are sorted by ``name`` for stable ordering, but each ensemble's
    ``llms_to_query`` order is preserved (fan-out order is semantically meaningful).
    """
    config = load_config(str(config_path))
    dump = config.model_dump(mode="json")

    def by_name(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return sorted(items or [], key=lambda item: item["name"])

    return {
        "llm_definitions": by_name(dump.get("llm_definitions")),
        "prompt_definitions": by_name(dump.get("prompt_definitions")),
        "models": by_name(dump.get("models")),
        "service": dump.get("service"),
        "langfuse": dump.get("langfuse"),
    }


def _dumps(projection: dict[str, Any]) -> str:
    return json.dumps(projection, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_config_expansion_matches_golden() -> None:
    projection = project_config(SNAPSHOT)

    if os.getenv("REGEN_GOLDENS") == "1":
        GOLDEN.write_text(_dumps(projection), encoding="utf-8")

    assert GOLDEN.exists(), f"golden missing; regenerate with REGEN_GOLDENS=1 ({GOLDEN})"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert projection == expected


def test_snapshot_has_expected_shape() -> None:
    """Guard the snapshot itself so a truncated/edited fixture is caught explicitly."""
    projection = project_config(SNAPSHOT)
    # The frozen live config: 14 hand-authored ensembles + 1 auto `mom-debug`
    # (enable_mom_debug_model: true), 4 synthesis prompts, many expanded LLMs.
    model_names = {model["name"] for model in projection["models"]}
    assert len(projection["models"]) == 15
    assert "mom-debug" in model_names
    assert len(projection["prompt_definitions"]) == 4
    assert len(projection["llm_definitions"]) >= 60
    llm_names = {llm["name"] for llm in projection["llm_definitions"]}
    # Flagship ensemble is present, and variant expansion actually produced `base:suffix` names.
    assert "mom" in model_names
    assert any(":" in name for name in llm_names), "expected expanded variant names like 'oai56s:h'"
