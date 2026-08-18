"""Tool-calling helpers: turn classification and wire conversion (pure).

Ownership rule: the synthesizer emits the client-visible tool calls; fan-out members are
advisory. Continuation turns (the conversation tail is tool results) relay straight to the
synthesizer instead of paying for a fresh fan-out. `vote`/`first` strategies let a member
proposal be promoted directly (see ``select_member_tool_call``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any, Literal

from mom.domain.request import MessageIR, SpecificTool, ToolChoice, ToolSpec
from mom.domain.results import ModelOutcome


TurnKind = Literal["fresh", "relay"]

# Providers whose synthesizer can be handed opaque Responses `type: mcp` tool blocks to relay.
_REMOTE_MCP_PROVIDERS = frozenset({"openai", "azure"})


def classify_turn(messages: tuple[MessageIR, ...]) -> TurnKind:
    """A turn is a relay continuation iff a tool message appears after the last assistant turn."""
    last_assistant = -1
    for index, message in enumerate(messages):
        if message.role == "assistant":
            last_assistant = index
    for message in messages[last_assistant + 1 :]:
        if message.role == "tool":
            return "relay"
    return "fresh"


def toolspec_to_wire(tool: ToolSpec) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.name}
    if tool.description is not None:
        function["description"] = tool.description
    if tool.parameters is not None:
        function["parameters"] = tool.parameters
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def tools_to_wire(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [toolspec_to_wire(tool) for tool in tools]


def tool_choice_to_wire(choice: ToolChoice) -> Any:
    if isinstance(choice, SpecificTool):
        return {"type": "function", "function": {"name": choice.name}}
    return choice


def member_tool_summary(tools: tuple[ToolSpec, ...], *, max_tools: int = 40) -> str:
    """A short, schema-free description of the available tools for advisory members."""
    lines = []
    for tool in tools[:max_tools]:
        desc = (tool.description or "").split(".")[0].strip()
        lines.append(f"- {tool.name}: {desc}" if desc else f"- {tool.name}")
    extra = len(tools) - max_tools
    if extra > 0:
        lines.append(f"(+{extra} more)")
    body = "\n".join(lines)
    return (
        "The assistant that will produce the final answer has these tools available (you "
        f"cannot invoke them):\n{body}"
    )


def _normalize_arguments(raw: object) -> str:
    """Canonicalize a tool call's ``arguments`` (JSON string) so equal calls compare equal.

    Falls back to the trimmed raw text when the arguments are not valid JSON (a partial or
    provider-specific encoding), so comparison stays total and never raises.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def tool_call_signature(call: Mapping[str, Any]) -> tuple[str, str]:
    """A call's identity for voting/dedup: ``(name, normalized-arguments)``."""
    fn = call.get("function", {})
    name = str(fn.get("name") or "") if isinstance(fn, Mapping) else ""
    arguments = fn.get("arguments") if isinstance(fn, Mapping) else None
    return name, _normalize_arguments(arguments)


def summarize_member_tool_calls(outcomes: Sequence[ModelOutcome]) -> str | None:
    """A concise, advisory summary of member-proposed tool calls for the synthesizer's context.

    Returns ``None`` when no member proposed a tool call (so no empty block is appended).
    """
    lines: list[str] = []
    for outcome in outcomes:
        for call in outcome.tool_calls:
            name, arguments = tool_call_signature(call)
            if name:
                lines.append(f"- {outcome.identity} proposed {name}({arguments})")
    if not lines:
        return None
    return (
        "For reference, some panel members proposed these tool calls (advisory — you decide "
        "whether to call a tool):\n" + "\n".join(lines)
    )


def _vote_winner(outcomes: Sequence[ModelOutcome], threshold: int) -> dict[str, Any] | None:
    """The tool call proposed by the most *distinct* members, if it clears ``threshold``."""
    votes: dict[tuple[str, str], int] = {}
    representative: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for outcome in outcomes:
        counted: set[tuple[str, str]] = set()
        for call in outcome.tool_calls:
            sig = tool_call_signature(call)
            if not sig[0] or sig in counted:  # one vote per member per signature; skip unnamed
                continue
            counted.add(sig)
            if sig not in votes:
                votes[sig] = 0
                representative[sig] = dict(call)
                order.append(sig)
            votes[sig] += 1
    best: tuple[str, str] | None = None
    for sig in order:  # first-appearance order breaks ties deterministically
        if votes[sig] >= threshold and (best is None or votes[sig] > votes[best]):
            best = sig
    return representative[best] if best is not None else None


def select_member_tool_call(
    outcomes: Sequence[ModelOutcome], *, strategy: str, threshold: int
) -> tuple[dict[str, Any], ...] | None:
    """Resolve a ``vote``/``first`` strategy against member proposals (``None`` = fall back).

    - ``first``: the proposing (config-ordered) first member's full tool-call set.
    - ``vote``: the single call proposed by at least ``threshold`` distinct members.
    ``arbitrate`` (and any no-decision case) returns ``None`` so the caller synthesizes normally.
    """
    if strategy == "first":
        for outcome in outcomes:
            if outcome.tool_calls:
                return tuple(outcome.tool_calls)
        return None
    if strategy == "vote":
        winner = _vote_winner(outcomes, threshold)
        return (winner,) if winner is not None else None
    return None


def restore_provider_tool_ids(
    messages: list[dict[str, Any]], lookup: Callable[[str], str | None]
) -> list[dict[str, Any]]:
    """Swap MoM-minted client tool-call ids back to their provider-native ids for a relay turn.

    ``lookup`` maps a client id to the provider id to restore (or ``None`` to leave it). Applies
    to both an assistant message's ``tool_calls[].id`` and a tool message's ``tool_call_id``.
    Pure: the input list/dicts are never mutated.
    """

    def _restore(client_id: object) -> object:
        if isinstance(client_id, str):
            provider_id = lookup(client_id)
            if provider_id is not None:
                return provider_id
        return client_id

    restored: list[dict[str, Any]] = []
    for message in messages:
        updated = dict(message)
        calls = updated.get("tool_calls")
        if isinstance(calls, list):
            updated["tool_calls"] = [
                {**call, "id": _restore(call.get("id"))} if isinstance(call, dict) else call
                for call in calls
            ]
        if "tool_call_id" in updated:
            updated["tool_call_id"] = _restore(updated.get("tool_call_id"))
        restored.append(updated)
    return restored


def provider_supports_remote_mcp(model: str, api: str) -> bool:
    """Whether a synthesizer can be forwarded opaque Responses ``type: mcp`` tool blocks.

    Remote MCP is a Responses-API feature; only providers known to accept it qualify. Everything
    else keeps the clean 400 (the tools are executed client-side instead).
    """
    if api != "responses":
        return False
    provider = model.split("/", 1)[0].lower() if "/" in model else "openai"
    return provider in _REMOTE_MCP_PROVIDERS
