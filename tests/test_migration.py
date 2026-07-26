"""Acceptance gate: the v1 -> v2 migration reproduces the real config's meaning.

Loads the frozen live-config snapshot through the v1 loader, migrates it, loads the result
through the v2 loader, and proves every v1 LLM and ensemble survives with equivalent semantics
(modulo the documented, deliberate deviations: names sanitized ':'/'+' -> '-', and gemini's
api_key_env moving GOOGLE -> GEMINI with GOOGLE as fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom_service.config import load_config as load_v1
from tools.migrate_v1_config import migrate, sanitize_name

SNAPSHOT = Path(__file__).parent / "golden" / "fixtures" / "live_config_snapshot.yaml"


@pytest.fixture(scope="module")
def migrated():
    v1 = load_v1(str(SNAPSHOT))
    v2_dict = migrate(v1)
    catalog = resolve_catalog(Config.model_validate(v2_dict))
    return v1, catalog


def test_migrated_config_validates_and_resolves(migrated):
    v1, catalog = migrated
    # Every expanded v1 LLM and every v1 model (incl. auto mom-debug) survives.
    assert len(catalog.llms) == len(v1.llm_definitions)
    assert len(catalog.ensembles) == len(v1.models)


def test_every_v1_llm_is_equivalent(migrated):
    v1, catalog = migrated
    for llm in v1.llm_definitions:
        name = sanitize_name(llm.name)
        assert name in catalog.llms, f"missing migrated llm {name}"
        resolved = catalog.llms[name]
        assert resolved.model == llm.model
        assert dict(resolved.params) == (llm.params or {})
        expected_api = "responses" if llm.api_mode == "responses" else "chat"
        assert resolved.api == expected_api


def test_every_v1_ensemble_is_equivalent(migrated):
    v1, catalog = migrated
    for model in v1.models:
        name = sanitize_name(model.name)
        assert name in catalog.ensembles, f"missing migrated ensemble {name}"
        ens = catalog.ensembles[name]
        member_llms = [m.llm for m in ens.members]
        assert member_llms == [sanitize_name(q) for q in model.llms_to_query]
        assert ens.synthesizer.llm == sanitize_name(model.concluding_llm)
        assert ens.synthesizer.prompt == model.concluding_prompt
        expected_show_work = "inline" if model.include_thinking_context else "off"
        assert ens.show_work == expected_show_work


def test_gemini_key_env_is_fixed_to_gemini(migrated):
    _v1, catalog = migrated
    gemini_llms = [llm for llm in catalog.llms.values() if llm.model.startswith("gemini/")]
    assert gemini_llms, "snapshot should contain gemini models"
    for llm in gemini_llms:
        # Deliberate deviation: v2 prefers GEMINI_API_KEY with GOOGLE_API_KEY as fallback.
        assert llm.api_key_env == "GEMINI_API_KEY"
        assert "GOOGLE_API_KEY" in llm.key_env_candidates


def test_names_are_v2_legal(migrated):
    _v1, catalog = migrated
    for name in list(catalog.llms) + list(catalog.ensembles):
        assert ":" not in name and "+" not in name
