"""The v2 cache-key port is bit-compatible with the v1 golden.

Feeds the exact corpus inputs used to generate ``tests/golden/cache_keys.json`` and asserts the
v2 ``cache_key`` produces identical digests — proving the port did not silently change the
keyspace (which would re-spend the entire on-disk response cache).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mom.domain.cachekey import cache_key


GOLDEN = json.loads(
    (Path(__file__).parent.parent / "tests" / "golden" / "cache_keys.json").read_text()
)

_LLM = "oai56s:h"
_ALIAS = "oai56s:h+second"
_MODEL = "openai/gpt-5.6-sol"

_TEXT: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing simply."},
]


def _vision(sig: str, date: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"https://bucket.s3.amazonaws.com/k.jpg?X-Amz-Signature={sig}&X-Amz-Date={date}"
                    },
                },
            ],
        },
    ]


def _key(name: str, messages: list[dict[str, Any]], params: dict[str, Any] | None) -> str:
    return cache_key(llm_name=name, model=_MODEL, messages=messages, params=params)


def test_v2_cache_keys_match_v1_golden():
    computed = {
        "text_no_params": _key(_LLM, _TEXT, None),
        "text_with_temp": _key(_LLM, _TEXT, {"temperature": 0.7}),
        "text_ignorelist_params": _key(
            _LLM,
            _TEXT,
            {"temperature": 0.7, "api_key": "sk-secret", "num_retries": 5, "timeout": 30},
        ),
        "text_reordered_params": _key(_LLM, _TEXT, {"top_p": 0.9, "temperature": 0.7}),
        "text_reordered_params_swapped": _key(_LLM, _TEXT, {"temperature": 0.7, "top_p": 0.9}),
        "text_alias_name": _key(_ALIAS, _TEXT, None),
        "vision_presigned": _key(_LLM, _vision("AAA", "20260101"), None),
        "vision_presigned_other_sig": _key(_LLM, _vision("ZZZ", "20260202"), None),
    }
    assert computed == GOLDEN


def test_v2_cache_key_invariants():
    # Presigned-URL params stripped -> stable across signatures.
    assert _key(_LLM, _vision("AAA", "1"), None) == _key(_LLM, _vision("ZZZ", "2"), None)
    # Sensitive keys never appear in the canonical form.
    key_with_secret = _key(_LLM, _TEXT, {"extra_headers": {"Authorization": "Bearer x"}})
    key_plain = _key(_LLM, _TEXT, {"extra_headers": {}})
    assert key_with_secret == key_plain
