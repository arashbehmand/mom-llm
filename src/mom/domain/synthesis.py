"""Message assembly: IR -> provider dicts, and the concluding-synthesis prompt.

The synthesis assembly keeps v1's tail-append order (client history, then the volatile candidate
block, then the synthesis prompt) so the stable history stays a cacheable prefix. Candidate
ordering is deterministic (config/member order), unlike v1's completion-order — see DEVIATIONS.md.
"""

from __future__ import annotations

import re
from typing import Any

from mom.domain.request import ImagePart, MessageIR, TextPart
from mom.domain.results import ModelOutcome


_CONCLUDING_INSTRUCTION_RE = re.compile(
    r"<<CONCLUDING-INSTRUCTION>>(.*?)<</CONCLUDING-INSTRUCTION>>", re.DOTALL
)


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


def extract_concluding_instruction(
    messages: tuple[MessageIR, ...],
) -> tuple[tuple[MessageIR, ...], str | None]:
    """Pull a ``<<CONCLUDING-INSTRUCTION>>...<</...>>`` block out of the last user message.

    Returns the messages with the block stripped and the extracted instruction (or None).
    """
    if not messages:
        return messages, None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "user" or not isinstance(message.content, str):
            continue
        match = _CONCLUDING_INSTRUCTION_RE.search(message.content)
        if not match:
            return messages, None
        instruction = match.group(1).strip()
        stripped = _CONCLUDING_INSTRUCTION_RE.sub("", message.content).strip()
        new_messages = (
            *messages[:index],
            MessageIR(role=message.role, content=stripped, name=message.name),
            *messages[index + 1 :],
        )
        return new_messages, instruction or None
    return messages, None


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
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if instruction:
        messages.append({"role": "user", "content": instruction})
    return messages


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
