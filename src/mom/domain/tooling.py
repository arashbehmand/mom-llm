"""Tool-calling helpers: turn classification and wire conversion (pure).

Ownership rule: the synthesizer emits the client-visible tool calls; fan-out members are
advisory. Continuation turns (the conversation tail is tool results) relay straight to the
synthesizer instead of paying for a fresh fan-out.
"""

from __future__ import annotations

from typing import Any, Literal

from mom.domain.request import MessageIR, SpecificTool, ToolChoice, ToolSpec


TurnKind = Literal["fresh", "relay"]


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
