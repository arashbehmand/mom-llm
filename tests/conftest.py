"""Shared test fixtures.

Hermetic environment: a developer's local ``.env`` (or litellm's import-time ``load_dotenv()``,
which reads the CWD ``.env`` into ``os.environ`` the first time any test imports litellm) must not
leak secrets/config into tests. CI has no ``.env``; this keeps local runs identical to CI.

Since config resolution became a search path, "ambient" also means the *filesystem*. A developer
with a real ``~/.mom/config.yaml`` would otherwise get a different answer from the same test than
CI does — the exact class of divergence this module exists to prevent, just moved from the
environment to disk. So the fixture also relocates ``$HOME``, the XDG roots, and the working
directory into an empty ``tmp_path``.

Relocating ``$HOME`` is worth it on its own: ``platformdirs.user_data_dir`` expands it, so a test
that reaches the data-dir fallback would otherwise write into the developer's real
``~/Library/Application Support/mom-llm``.
"""

from __future__ import annotations

import os
from pathlib import Path

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
def _hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear ambient secrets/config before each test (tests that need a var set it themselves)."""
    for var in _AMBIENT_VARS:
        monkeypatch.delenv(var, raising=False)
    for key in list(os.environ):
        if key.startswith("MOM_"):
            monkeypatch.delenv(key, raising=False)

    # An empty home and an empty working directory, so config discovery finds exactly what a
    # test puts there. Safe to chdir: every on-disk reference in the suite is anchored on
    # `Path(__file__).parent.parent` or `tmp_path`, and `sys.path` is fixed at collection time.
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
