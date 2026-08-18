"""Provider prompt-cache planning (pure).

Two levers, kept provider-specific: Anthropic reads cache breakpoints from ``cache_control``
markers on message content; OpenAI/xAI do automatic prefix caching and only need a stable
``prompt_cache_key`` for routing affinity. Both are cost levers, not correctness — a target that
does not support a marker/key simply ignores it, so emitting them is always safe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


_ANTHROPIC_FAMILY = frozenset({"anthropic", "claude", "vertex_ai", "bedrock"})


def is_anthropic_family(model: str) -> bool:
    """True for provider prefixes whose upstream honors Anthropic ``cache_control`` markers."""
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
    return provider in _ANTHROPIC_FAMILY


def openai_prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    """A stateless, conversation-affine cache key: a hash of the stable leading messages.

    Excludes the final (volatile) message so the same conversation prefix routes to the same
    cache across turns, without the gateway holding any server-side state.
    """
    stable = messages[:-1] if len(messages) > 1 else messages
    blob = json.dumps(stable, sort_keys=True, default=str)
    return "mom-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    """Promote message content to a list of content blocks so a marker has somewhere to live."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]
    return [{"type": "text", "text": str(content)}]


def inject_anthropic_cache(
    messages: list[dict[str, Any]], *, ttl: str = "5m"
) -> list[dict[str, Any]]:
    """Copy ``messages`` with ``cache_control`` breakpoints at the stable prefix (max 4).

    Marks the last system message (stable system prompt) and the end of the stable history (the
    message before the final, volatile one). String content is promoted to a text block. The
    input is never mutated.
    """
    if not messages:
        return messages
    marker: dict[str, Any] = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        marker["ttl"] = ttl

    targets: set[int] = set()
    last_system = max((i for i, m in enumerate(messages) if m.get("role") == "system"), default=-1)
    if last_system >= 0:
        targets.add(last_system)
    if len(messages) >= 2:
        targets.add(len(messages) - 2)

    result = [dict(m) for m in messages]
    for i in sorted(targets)[:4]:  # Anthropic allows at most 4 cache breakpoints
        blocks = _as_blocks(result[i].get("content", ""))
        blocks[-1] = {**blocks[-1], "cache_control": marker}
        result[i]["content"] = blocks
    return result
