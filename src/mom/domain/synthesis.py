"""Message assembly: IR -> provider dicts, and the concluding-synthesis prompt.

Assembly order is client history, the synthesis prompt, then the candidate block — instructions
before the evidence they are about. v1 put the prompt last; that ended the turn with a long block
of meta-instructions, and a subscription proxy in front of a CLI-shaped API answered such a turn
with a greeting instead of an answer, as though no question had been asked (2026-09-19, reproduced
on both of two failing runs and fixed by this order alone). It also lengthens the cacheable
prefix, since the prompt is fixed and only the candidates are volatile. Candidate ordering is
deterministic (config/member order), unlike v1's completion-order — see DEVIATIONS.md.
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


def merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join runs of consecutive messages that share a role into one message.

    Off by default (``defaults.call.merge_same_role``), because a conversation is normally sent
    as it was written. It exists for upstreams that cannot handle several turns in a row from the
    same role: a subscription proxy in front of a CLI-shaped API was observed keeping only the
    LAST of them, which silently deleted the question and every candidate answer and left the
    synthesizer replying to the system prompt alone — a greeting where the answer should be, with
    nothing in the logs to say why. Merged, the same content survives as one turn.

    Tool plumbing is never merged: a tool result and an assistant turn carrying ``tool_calls``
    each answer for one call id, and joining them would break that pairing. Messages with
    different ``name`` values stay apart for the same reason — the name identifies the speaker.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if previous is not None and _joinable(previous, message):
            merged[-1] = {
                **previous,
                "content": _join_content(previous["content"], message["content"]),
            }
            continue
        merged.append(dict(message))
    return merged


def _joinable(previous: dict[str, Any], message: dict[str, Any]) -> bool:
    if previous.get("role") != message.get("role"):
        return False
    if previous.get("name") != message.get("name"):
        return False
    return not any(
        m.get("tool_calls") or m.get("tool_call_id") is not None for m in (previous, message)
    )


def _join_content(first: Any, second: Any) -> Any:
    """Two message bodies as one. Plain text joins with a blank line; anything multipart keeps its
    parts, so an image — or a ``cache_control`` breakpoint already placed on a block — survives."""
    if isinstance(first, str) and isinstance(second, str):
        return f"{first}\n\n{second}" if first and second else first or second
    return [*_as_parts(first), *_as_parts(second)]


def _as_parts(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content or [])


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
        # Closed with an END marker: every shipped synthesis prompt tells the model the candidates
        # are "enclosed between `===== RESPONSE i of N =====` and `===== END RESPONSE i =====`",
        # and until now that second marker was never written — leaving each candidate to run into
        # the next with nothing but a blank line between them.
        f"===== RESPONSE {i} of {total} =====\n{outcome.content}\n===== END RESPONSE {i} ====="
        for i, outcome in enumerate(successful, start=1)
    ]
    candidate_message = (
        f"Below are {total} independent responses from different models to the conversation "
        "above. Synthesize them into a single, superior answer.\n\n" + "\n\n".join(blocks)
    )
    messages = [*client_messages]
    if prompt:
        # Before the candidates, not after: see the module docstring. The prompt is also fixed
        # while the candidate block is not, so this is the longer cacheable prefix too.
        messages.append({"role": "user", "content": prompt})
    messages.append({"role": "user", "content": candidate_message})
    # Member-proposed tool calls, as advisory context about the candidates just read.
    tool_note = summarize_member_tool_calls(outcomes)
    if tool_note:
        messages.append({"role": "user", "content": tool_note})
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
