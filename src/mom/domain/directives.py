"""The ``<<SYSTEM>>`` in-message directive block (and its ``<<CONCLUDING-INSTRUCTION>>`` alias).

A human types this into the chat box; it never reaches a fan-out member or the synthesizer's
history. Grammar (header-block, HTTP-headers/git-trailers style — not YAML, which a chat box's
auto-indentation would mangle)::

    <<SYSTEM>>
    only: oai56s, cl48op
    exclude: k3
    include: g31p
    show_work: off
    dedupe: on
    Answer as a terse bullet list, no preamble.
    <</SYSTEM>>

Header lines are consumed from the top *while the key is a known directive*; the first line that
doesn't look like ``key: value`` at all ends the header zone, and everything from there on
(inclusive) is the instruction, verbatim. A bare body with no header lines is exactly today's
behavior: the whole thing is the instruction.

**Blank lines inside the header zone are skipped, not terminators.** They used to end it — an
escape hatch for an instruction that starts with something ``key:``-shaped — and that quietly cost
every block typed into a markdown chat box, which puts a blank line after the opening tag and
between every line the human presses Enter on. The whole header then read as prose: directives
silently inert, no warning, the fan-out fired anyway (observed live 2026-09-06 on a block whose
``include:`` never applied). The escape hatch that remains is the explicit terminator key
``instruction:``; prose starting with an unrecognized ``key:`` needs no hatch at all, since an
unknown key ends the header and survives verbatim (below).

A line that DOES look like ``key: value`` but names an unknown key ends the header zone like any
other prose line — and records a warning on ``warnings``, which
``engine/plan.py`` carries onto the plan and every surface renders in the think block. A typo'd
directive silently doing nothing (and firing the fan-out anyway) is the failure mode this is built
to avoid; taking the whole turn down over one is too blunt an instrument for it.

The legacy ``<<CONCLUDING-INSTRUCTION>>...<</CONCLUDING-INSTRUCTION>>`` marker keeps working
indefinitely, as the exact same block with header parsing disabled — a documented tag must never
retroactively reinterpret existing prose (e.g. a saved prompt that happens to start with
``Note:``) as a directive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from mom.domain.request import ImagePart, MessageIR, TextPart


# Case-insensitive tag, DOTALL body, non-greedy, closer tolerates one or two leading `<`.
_SYSTEM_RE = re.compile(r"<<\s*SYSTEM\s*>>(.*?)<{1,2}/\s*SYSTEM\s*>>", re.DOTALL | re.IGNORECASE)
# The original marker, unchanged: case-sensitive, exact double-`<` closer — no new leniency here,
# so nothing that worked (or didn't) before this feature shipped changes for it.
_CONCLUDING_RE = re.compile(
    r"<<CONCLUDING-INSTRUCTION>>(.*?)<</CONCLUDING-INSTRUCTION>>", re.DOTALL
)

_KEY_LINE_RE = re.compile(r"^([A-Za-z_]+):\s*(.*)$")
_FILTER_KEYS = frozenset({"exclude", "only", "include", "show_work", "synth", "dedupe"})
_KNOWN_KEYS = ", ".join(sorted({*_FILTER_KEYS, "instruction"}))


@dataclass(frozen=True, slots=True)
class SystemDirectives:
    """Parsed ``<<SYSTEM>>`` (or legacy ``<<CONCLUDING-INSTRUCTION>>``) block.

    Names in ``exclude``/``only``/``include`` are lowercased, requested member *identities*
    (``member_as or llm`` — the name shown in the think block and the progress dashboard) —
    matched against the catalog by the caller (this module is catalog-agnostic; see
    ``engine/plan.py``).
    """

    instruction: str | None = None
    exclude: tuple[str, ...] = ()
    only: tuple[str, ...] = ()
    # Names to ADD to this turn's panel: a member the tier or an `only:`/`exclude:` dropped, or any
    # llm in the catalog that isn't on the panel at all. Resolved in ``engine/plan.py``.
    include: tuple[str, ...] = ()
    show_work: str | None = None
    synth: str | None = None
    # Raw, un-validated like ``show_work`` — ``engine/plan.py`` owns the vocabulary check so every
    # directive reports a bad value the same way.
    dedupe: str | None = None
    # What the block asked for and didn't get, in the reader's words — rendered in the think block
    # alongside whatever notices plan resolution adds. Never a reason to fail the turn.
    warnings: tuple[str, ...] = ()


def _split_values(raw: str) -> tuple[str, ...]:
    """Comma- and/or whitespace-separated values, lowercased, order-preserving deduplicated."""
    tokens = (token.strip().lower() for token in re.split(r"[,\s]+", raw.strip()))
    return tuple(dict.fromkeys(token for token in tokens if token))


def _parse_body(body: str, *, legacy: bool) -> SystemDirectives:
    if legacy:
        return SystemDirectives(instruction=body.strip() or None)

    # No leading-newline surgery and no counting of blank lines: they are skipped wherever they
    # appear in the header zone (see the module docstring). What a human's editor does to
    # whitespace is not a directive.
    lines = body.splitlines()
    collected: dict[str, list[str]] = {}
    warnings: list[str] = []
    idx = 0
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue  # blank lines are the editor's, not the human's
        match = _KEY_LINE_RE.match(stripped)
        if match is None:
            break  # doesn't look like a directive at all -> header zone ends here
        key, value = match.group(1).lower(), match.group(2).strip()
        if key == "instruction":  # escape hatch #2: an explicit terminator
            if value:
                lines[idx] = value  # fold this line's value in as the instruction's first line
            else:
                idx += 1  # bare "instruction:" line itself carries no text — drop it
            break
        if key not in _FILTER_KEYS:
            # Not a directive we know: the header zone ends here and this line starts the
            # instruction, exactly as an ordinary prose line would. Nothing the human typed is
            # dropped — but nothing is silently obeyed either, hence the warning.
            warnings.append(
                f"unknown <<SYSTEM>> directive {key!r} — the header ends there, so that line "
                f"and everything after it is instruction text (known keys: {_KNOWN_KEYS})"
            )
            break
        collected.setdefault(key, []).append(value)
    else:
        idx = len(lines)  # every line was a recognized directive -> no instruction body at all

    instruction = "\n".join(lines[idx:]).strip() or None

    def _merged(key: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v for raw in collected.get(key, []) for v in _split_values(raw)))

    exclude = _merged("exclude")
    only = _merged("only")
    show_work_lines = collected.get("show_work", [])
    synth_lines = collected.get("synth", [])
    dedupe_lines = collected.get("dedupe", [])
    return SystemDirectives(
        instruction=instruction,
        exclude=exclude,
        only=only,
        include=_merged("include"),
        show_work=(show_work_lines[-1].lower() or None) if show_work_lines else None,
        synth=(synth_lines[-1].lower() or None) if synth_lines else None,
        dedupe=(dedupe_lines[-1].lower() or None) if dedupe_lines else None,
        warnings=tuple(warnings),
    )


def _try_extract(text: str) -> tuple[str, SystemDirectives] | None:
    """If ``text`` contains a ``<<SYSTEM>>`` or legacy block, return ``(stripped_text,
    directives)`` — ``<<SYSTEM>>`` is tried first (so it wins if a message somehow contains both).

    Every occurrence of the matched marker is stripped (not just the one parsed for directives) —
    matches the original marker's behavior: a second block's content is silently discarded, not
    honored, but its raw text never leaks into what a member/the synthesizer sees either.
    """
    match = _SYSTEM_RE.search(text)
    if match is not None:
        return _SYSTEM_RE.sub("", text).strip(), _parse_body(match.group(1), legacy=False)
    match = _CONCLUDING_RE.search(text)
    if match is not None:
        stripped = _CONCLUDING_RE.sub("", text).strip()
        return stripped, _parse_body(match.group(1), legacy=True)
    return None


def extract_system_block(
    messages: tuple[MessageIR, ...],
) -> tuple[tuple[MessageIR, ...], SystemDirectives | None]:
    """Pull a directive block out of the LAST user message, scanning messages backward.

    Only that one message is inspected — if it has no block, the scan stops there and returns
    None; older history is never searched (matches the original marker's behavior). For a
    multipart message, each ``TextPart`` is checked in message order and the first one containing
    a *complete* block is stripped; every ``ImagePart`` and any other ``TextPart`` is left
    untouched. This is the fix for the marker being dead on every multipart request (e.g. every
    request an image- or file-aware client sends) — the original implementation skipped any
    non-plain-string user message outright.
    """
    if not messages:
        return messages, None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            found = _try_extract(message.content)
            if found is None:
                return messages, None
            stripped_text, directives = found
            new_message = replace(message, content=stripped_text)
        else:
            found = None
            part_index: int | None = None
            for pi, part in enumerate(message.content):
                if isinstance(part, TextPart):
                    result = _try_extract(part.text)
                    if result is not None:
                        found = result
                        part_index = pi
                        break
            if found is None or part_index is None:
                return messages, None
            stripped_text, directives = found
            new_parts: list[TextPart | ImagePart] = list(message.content)
            new_parts[part_index] = TextPart(text=stripped_text)
            new_message = replace(message, content=tuple(new_parts))
        return (*messages[:index], new_message, *messages[index + 1 :]), directives
    return messages, None
