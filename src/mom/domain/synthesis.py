"""Message assembly: IR -> provider dicts, and the concluding-synthesis prompt.

The synthesis assembly keeps v1's tail-append order (client history, then the volatile candidate
block, then the synthesis prompt) so the stable history stays a cacheable prefix. Candidate
ordering is deterministic (config/member order), unlike v1's completion-order — see DEVIATIONS.md.
"""

from __future__ import annotations

from typing import Any

from mom.domain.request import ImagePart, MessageIR, TextPart
from mom.domain.results import ModelOutcome
from mom.domain.tooling import summarize_member_tool_calls


def message_to_dict(message: MessageIR) -> dict[str, Any]:
    """Render one IR message to an OpenAI-shaped dict."""
    content: Any
    if isinstance(message.content, str):
        content = message.content
    else:
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                image: dict[str, Any] = {"url": part.url}
                if part.detail:
                    image["detail"] = part.detail
                parts.append({"type": "image_url", "image_url": image})
        content = parts
    out: dict[str, Any] = {"role": message.role, "content": content}
    if message.name:
        out["name"] = message.name
    if message.tool_call_id:
        out["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in message.tool_calls
        ]
    return out


def messages_to_dicts(messages: tuple[MessageIR, ...]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def append_instruction(
    messages: list[dict[str, Any]], instruction: str | None
) -> list[dict[str, Any]]:
    """Append the ``<<SYSTEM>>``/``<<CONCLUDING-INSTRUCTION>>`` instruction as the final message,
    if any (a no-op otherwise).

    Shared by the normal synthesis path (the tail of :func:`build_synthesis_messages`) and the
    passthrough/relay ``skip_fanout`` path — which, before this, silently dropped the instruction
    entirely: it was stripped from the client message during plan resolution but never
    re-attached anywhere on that path.
    """
    if not instruction:
        return messages
    return [*messages, {"role": "user", "content": instruction}]


def build_synthesis_messages(
    client_messages: list[dict[str, Any]],
    outcomes: list[ModelOutcome],
    *,
    prompt: str | None,
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the concluding model's messages: history + candidate block + synthesis prompt."""
    successful = [o for o in outcomes if o.ok]
    total = len(successful)
    blocks = [
        f"===== RESPONSE {i} of {total} =====\n{outcome.content}"
        for i, outcome in enumerate(successful, start=1)
    ]
    candidate_message = (
        f"Below are {total} independent responses from different models to the conversation "
        "above. Synthesize them into a single, superior answer.\n\n" + "\n\n".join(blocks)
    )
    messages = [*client_messages, {"role": "user", "content": candidate_message}]
    # Surface any member-proposed tool calls as advisory context (the candidate envelope). Volatile,
    # so it stays after the cacheable history prefix and before the fixed synthesis prompt.
    tool_note = summarize_member_tool_calls(outcomes)
    if tool_note:
        messages.append({"role": "user", "content": tool_note})
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return append_instruction(messages, instruction)


def all_failed_message(outcomes: list[ModelOutcome]) -> list[dict[str, Any]]:
    """Fallback synthesis input when no member succeeded."""
    errors = "; ".join(f"{o.identity}: {o.error or o.status}" for o in outcomes) or "unknown"
    return [
        {
            "role": "user",
            "content": (
                "All ensemble members failed to produce a response. Reply with a brief apology "
                f"and, if useful, note the failure reasons: {errors}"
            ),
        }
    ]
