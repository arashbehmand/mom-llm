"""Settings: MOM_ prefix + legacy-name back-compat."""

from __future__ import annotations

import pytest

from mom.runtime.settings import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.api_token is None
    assert settings.log_format == "text"


def test_legacy_api_token_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_TOKEN", "legacy-secret")
    settings = Settings(_env_file=None)
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "legacy-secret"


def test_mom_prefix_wins_over_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_TOKEN", "legacy-secret")
    monkeypatch.setenv("MOM_API_TOKEN", "new-secret")
    settings = Settings(_env_file=None)
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "new-secret"


def test_legacy_redis_and_config_aliases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("MOM_CONFIG_PATH", "/etc/mom/config.yaml")
    settings = Settings(_env_file=None)
    assert settings.redis_url == "redis://localhost:6379"
    assert str(settings.config_file) == "/etc/mom/config.yaml"
