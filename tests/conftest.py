"""Shared test fixtures.

Hermetic environment: a developer's local ``.env`` (or litellm's import-time ``load_dotenv()``,
which reads the CWD ``.env`` into ``os.environ`` the first time any test imports litellm) must not
leak secrets/config into tests. CI has no ``.env``; this keeps local runs identical to CI.
"""

from __future__ import annotations

import os

import pytest


# Ambient names a `.env` might inject (legacy/unprefixed) that could perturb tests.
_AMBIENT_VARS = (
    "API_TOKEN",
    "REDIS_URL",
    "MOM_CONFIG_PATH",
    "LITELLM_VERBOSE",
    "ALLOWED_CORS_ORIGINS",
    "MUSE_SPARK_PROXY_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
    "PERPLEXITY_API_KEY",
    "XAI_API_KEY",
    "QWEN_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient secrets/config before each test (tests that need a var set it themselves)."""
    for var in _AMBIENT_VARS:
        monkeypatch.delenv(var, raising=False)
    for key in list(os.environ):
        if key.startswith("MOM_"):
            monkeypatch.delenv(key, raising=False)
