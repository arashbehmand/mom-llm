"""
Regression tests: ensure previously reported runtime issues don't recur.

1) LLM definitions with `params: null` must not cause a TypeError when expanded.
2) metrics DB is initialized on import and contains the `usage_metrics` table.
"""

import os
import sqlite3
from builtins import anext
from unittest.mock import AsyncMock, patch

import pytest

from mom_service import metrics_db
from mom_service.config import LLMDefinition, ModelConfig, MoMConfig, ServiceConfig


@pytest.mark.asyncio
async def test_llm_params_none_does_not_raise(mock_litellm_response):
    """Ensure _call_lite_llm / call_llm tolerates LLMDefinition.params == None."""
    llm_def = LLMDefinition(
        name="reg-none", model="gpt-test", api_key_env="OPENAI_API_KEY", params=None
    )

    # Minimal MoMConfig for the call
    cfg = MoMConfig(
        llm_definitions=[llm_def],
        models=[ModelConfig(name="m", llms_to_query=["reg-none"], concluding_llm="reg-none")],
        service=ServiceConfig(timeout_seconds=30, cache_enabled=False, max_llm_retries=1),
    )

    messages = [{"role": "user", "content": "Hello"}]

    # Patch litellm.acompletion to return a valid mocked response
    with patch("mom_service.llm_calls.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = mock_litellm_response

        # Call the internal async generator directly; it should not raise TypeError
        gen = __import__(
            "mom_service.llm_calls", fromlist=["_"]
        )._call_lite_llm(  # pylint: disable=protected-access
            llm_def,
            messages,
            timeout=30,
            config=cfg,
            options={"request_id": "reg-1", "mom_model_name": "m"},
        )

        result = await anext(gen)
        assert result is not None


def test_metrics_db_initialized_and_has_table():
    """Verify metrics DB file exists and contains the usage_metrics table after import."""
    db_path = metrics_db.METRICS_DB_PATH
    assert os.path.exists(db_path), f"Metrics DB not found at {db_path}"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_metrics'")
    row = cur.fetchone()
    conn.close()

    assert (
        row is not None and row[0] == "usage_metrics"
    ), "usage_metrics table not present in metrics DB"
