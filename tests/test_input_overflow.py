"""Input-overflow guard: a member over its llm's max_input_tokens is skipped or the request 400s.

Resolved entirely at plan time (a cheap ~chars/4 estimate), so ``reject`` is a clean pre-flight
400 and ``skip`` simply shrinks the panel before any fan-out spend.
"""

from __future__ import annotations

from textwrap import dedent

import pytest
import yaml

from mom.config.resolve import ResolvedCatalog, resolve_catalog
from mom.config.schema import Config
from mom.domain.errors import InvalidRequestError
from mom.domain.request import ChatRequestIR, MessageIR
from mom.engine.plan import resolve_plan


def _catalog(text: str) -> ResolvedCatalog:
    return resolve_catalog(Config.model_validate(yaml.safe_load(dedent(text))))


def _ir(content: str) -> ChatRequestIR:
    return ChatRequestIR(model="e", messages=(MessageIR(role="user", content=content),))


SKIP_CONFIG = """
version: 2
llms:
  small: { model: openai/small, max_input_tokens: 5 }
  big: { model: openai/big }
ensembles:
  e:
    on_input_overflow: skip
    members: [{ llm: small }, { llm: big }]
    synthesizer: { llm: big }
"""


def test_overflow_skip_drops_only_the_over_budget_member():
    catalog = _catalog(SKIP_CONFIG)
    plan = resolve_plan(catalog, _ir("x" * 400))  # ~100 tokens, over 'small', under 'big' (no cap)
    assert {m.identity for m in plan.members} == {"big"}


def test_within_budget_keeps_every_member():
    catalog = _catalog(SKIP_CONFIG)
    plan = resolve_plan(catalog, _ir("hi"))  # ~1 token, under both caps
    assert {m.identity for m in plan.members} == {"small", "big"}


REJECT_CONFIG = """
version: 2
llms:
  small: { model: openai/small, max_input_tokens: 5 }
ensembles:
  e:
    on_input_overflow: reject
    members: [{ llm: small }]
    synthesizer: { llm: small }
"""


def test_overflow_reject_raises_400_pre_flight():
    catalog = _catalog(REJECT_CONFIG)
    with pytest.raises(InvalidRequestError) as exc_info:
        resolve_plan(catalog, _ir("x" * 400))
    assert exc_info.value.http_status == 400
    assert "max_input_tokens" in exc_info.value.safe_message


def test_reject_mode_allows_within_budget_input():
    catalog = _catalog(REJECT_CONFIG)
    plan = resolve_plan(catalog, _ir("hi"))
    assert {m.identity for m in plan.members} == {"small"}
