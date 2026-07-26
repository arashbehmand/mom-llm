"""Characterization golden: v1 response-cache key.

``_generate_cache_key`` is a crown-jewel function: its SHA256 output is the primary key of the
on-disk response cache. If v2 changes it even slightly, every cached entry silently misses and
the whole cache is re-spent against paid providers. This golden pins the exact digests for a
corpus of representative inputs AND asserts the behavioral invariants (ignore-list params,
sensitive-key redaction, S3 presigned-URL stripping, key-order canonicalization, alias
sensitivity) so the v2 port can be proven bit-compatible.

Regenerate ONLY on a deliberate, understood key-format change (it invalidates every cache)::

    REGEN_GOLDENS=1 .venv/bin/python -m pytest tests/test_golden_cache_keys.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mom_service.config import LLMDefinition
from mom_service.llm_calls import _generate_cache_key

GOLDEN = Path(__file__).parent / "golden" / "cache_keys.json"

_LLM = LLMDefinition(name="oai56s:h", model="openai/gpt-5.6-sol", api_key_env="OPENAI_API_KEY")
_LLM_ALIAS = LLMDefinition(
    name="oai56s:h+second", model="openai/gpt-5.6-sol", api_key_env="OPENAI_API_KEY"
)

_TEXT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing simply."},
]
_VISION_MESSAGES: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://bucket.s3.amazonaws.com/k.jpg?X-Amz-Signature=AAA&X-Amz-Date=20260101"
                },
            },
        ],
    },
]
_VISION_MESSAGES_OTHER_SIG: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://bucket.s3.amazonaws.com/k.jpg?X-Amz-Signature=ZZZ&X-Amz-Date=20260202"
                },
            },
        ],
    },
]


def _key(llm: LLMDefinition, messages: list[dict[str, Any]], params: dict[str, Any] | None) -> str:
    return _generate_cache_key(llm, messages, params)


# case name -> computed digest (the corpus that gets frozen)
def corpus() -> dict[str, str]:
    return {
        "text_no_params": _key(_LLM, _TEXT_MESSAGES, None),
        "text_with_temp": _key(_LLM, _TEXT_MESSAGES, {"temperature": 0.7}),
        "text_ignorelist_params": _key(
            _LLM,
            _TEXT_MESSAGES,
            {"temperature": 0.7, "api_key": "sk-secret", "num_retries": 5, "timeout": 30},
        ),
        "text_reordered_params": _key(
            _LLM, _TEXT_MESSAGES, {"top_p": 0.9, "temperature": 0.7}
        ),
        "text_reordered_params_swapped": _key(
            _LLM, _TEXT_MESSAGES, {"temperature": 0.7, "top_p": 0.9}
        ),
        "text_alias_name": _key(_LLM_ALIAS, _TEXT_MESSAGES, None),
        "vision_presigned": _key(_LLM, _VISION_MESSAGES, None),
        "vision_presigned_other_sig": _key(_LLM, _VISION_MESSAGES_OTHER_SIG, None),
    }


def test_cache_keys_match_golden() -> None:
    keys = corpus()
    if os.getenv("REGEN_GOLDENS") == "1":
        GOLDEN.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert GOLDEN.exists(), f"golden missing; regenerate with REGEN_GOLDENS=1 ({GOLDEN})"
    assert keys == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_cache_key_invariants() -> None:
    keys = corpus()
    # Ignore-list params (api_key / num_retries / timeout) must not affect the key.
    assert keys["text_ignorelist_params"] == keys["text_with_temp"]
    # Param key order must be canonical.
    assert keys["text_reordered_params"] == keys["text_reordered_params_swapped"]
    # A distinct llm identity (alias) must produce a distinct key.
    assert keys["text_alias_name"] != keys["text_no_params"]
    # Volatile S3 presigned-URL query params must be stripped -> stable across signatures.
    assert keys["vision_presigned"] == keys["vision_presigned_other_sig"]
    # All digests are full-length SHA256 hex.
    assert all(len(digest) == 64 for digest in keys.values())
