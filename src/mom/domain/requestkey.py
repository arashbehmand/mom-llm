"""The *request identity* used to coalesce identical concurrent turns onto one fan-out run.

Structurally mirrors ``cachekey.py`` (same ``normalize()``, same canonical-JSON-then-SHA256
approach) but is a distinct, separately-versioned function serving a different purpose: the
response cache is per-member-call and pinned bit-for-bit to v1 (``cache_key`` must never change);
this is per-*request*, new in v2, and free to evolve — nothing here feeds ``cache_key`` and
nothing there feeds this.

Two concurrent requests share an identity exactly when they would produce the same fan-out +
synthesis plan: same ensemble, same directive-stripped messages, same tools/search/sampling/
effort/resolved directives. Deliberately excluded because they're labels, not plan inputs:
``request_id``/``X-Request-Id`` (a client retry mints a fresh one *because* the first attempt
looked stuck — that must still coalesce onto the original run, not prove itself a "different"
request), timestamps, ``stream``, ``include_usage``, and ``metadata``.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

from mom.domain.cachekey import normalize
from mom.domain.directives import extract_system_block
from mom.domain.request import ChatRequestIR, SpecificTool
from mom.domain.synthesis import messages_to_dicts


_VERSION = "mom-request-identity/v1"


def _tool_choice_repr(choice: str | SpecificTool) -> Any:
    return choice if isinstance(choice, str) else asdict(choice)


def request_identity(ir: ChatRequestIR) -> str:
    """Derive the coalescing key (SHA256 hex) for a chat request.

    Directives (``<<SYSTEM>>`` ``exclude:``/``only:``/``show_work:``/``synth:``) are parsed here
    — the same catalog-agnostic parse ``resolve_plan`` performs — and folded in as their resolved
    values, not the raw block text, so inconsequential formatting inside it (comma- vs.
    whitespace-separated lists, extra blank lines) can't fragment an otherwise-identical request
    into a distinct identity.
    """
    stripped_messages, directives = extract_system_block(ir.messages)
    key_data = {
        "v": _VERSION,
        "model": ir.model,
        "messages": normalize(messages_to_dicts(stripped_messages), redact_sensitive=False),
        "effort": ir.effort,
        "tools": normalize([asdict(t) for t in ir.tools], redact_sensitive=False),
        "mcp_tools": normalize(list(ir.mcp_tools), redact_sensitive=False),
        "tool_choice": normalize(_tool_choice_repr(ir.tool_choice), redact_sensitive=False),
        "parallel_tool_calls": ir.parallel_tool_calls,
        "web_search": ir.web_search,
        "response_format": normalize(ir.response_format, redact_sensitive=False),
        "sampling": normalize(asdict(ir.sampling), redact_sensitive=False),
        "user": ir.user,
        "directives": normalize(
            asdict(directives) if directives is not None else None, redact_sensitive=False
        ),
    }
    canonical = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
