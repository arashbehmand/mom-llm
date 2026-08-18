"""The <<SYSTEM>> directive block, and its <<CONCLUDING-INSTRUCTION>> legacy alias."""

from __future__ import annotations

import pytest

from mom.domain.directives import SystemDirectives, extract_system_block
from mom.domain.errors import InvalidRequestError
from mom.domain.request import ImagePart, MessageIR, TextPart


def _msgs(*messages: MessageIR) -> tuple[MessageIR, ...]:
    return tuple(messages)


def _ci(body: str) -> str:
    """Wrap ``body`` in the (long) legacy marker, to keep test lines under the length limit."""
    return f"<<CONCLUDING-INSTRUCTION>>{body}<</CONCLUDING-INSTRUCTION>>"


# -------------------------------------------------------------------------------------------
# Legacy <<CONCLUDING-INSTRUCTION>> — characterization of the original marker's behavior. The
# regex/strictness is byte-for-byte unchanged from before this feature; only multipart handling
# is deliberately different (see the multipart section below).
# -------------------------------------------------------------------------------------------


def test_legacy_marker_extracted_and_stripped_from_last_user_message():
    messages = _msgs(MessageIR(role="user", content=f"Question? {_ci('Be terse.')}"))
    new_messages, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Be terse."
    assert new_messages[0].content == "Question?"
    assert directives.exclude == ()
    assert directives.only == ()


def test_legacy_marker_absent_from_last_message_stops_the_scan():
    # A marker in an EARLIER user turn is never searched — only the last user message counts.
    messages = _msgs(
        MessageIR(role="user", content=_ci("old")),
        MessageIR(role="assistant", content="ok"),
        MessageIR(role="user", content="no marker here"),
    )
    new_messages, directives = extract_system_block(messages)
    assert directives is None
    assert new_messages == messages  # untouched, including the earlier marker


def test_legacy_marker_in_assistant_message_is_ignored():
    messages = _msgs(
        MessageIR(role="assistant", content=_ci("x")),
        MessageIR(role="user", content="plain question"),
    )
    _, directives = extract_system_block(messages)
    assert directives is None


def test_legacy_marker_is_case_sensitive():
    messages = _msgs(MessageIR(role="user", content=_ci("x").lower()))
    _, directives = extract_system_block(messages)
    assert directives is None  # lowercase doesn't match — exact behavior preserved


def test_legacy_marker_requires_the_exact_double_lt_closer():
    single_lt = "<<CONCLUDING-INSTRUCTION>>x</CONCLUDING-INSTRUCTION>>"
    messages = _msgs(MessageIR(role="user", content=single_lt))
    _, directives = extract_system_block(messages)
    assert directives is None  # single-`<` closer not accepted for the legacy marker


def test_legacy_marker_two_blocks_strips_both_honors_first():
    messages = _msgs(MessageIR(role="user", content=f"{_ci('first')} mid {_ci('second')}"))
    new_messages, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "first"
    assert "CONCLUDING-INSTRUCTION" not in new_messages[0].content
    assert new_messages[0].content == "mid"


def test_legacy_marker_empty_body_yields_none_instruction():
    messages = _msgs(MessageIR(role="user", content=_ci("")))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction is None


def test_legacy_marker_never_parses_directives_even_if_shaped_like_them():
    """A documented tag must never retroactively reinterpret existing prose as a directive —
    header parsing is disabled entirely for the legacy marker."""
    messages = _msgs(MessageIR(role="user", content=_ci("exclude: k3")))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ()
    assert directives.instruction == "exclude: k3"


def test_message_name_is_preserved_after_stripping():
    messages = _msgs(
        MessageIR(
            role="user",
            name="alice",
            content="<<CONCLUDING-INSTRUCTION>>x<</CONCLUDING-INSTRUCTION>>",
        )
    )
    new_messages, _ = extract_system_block(messages)
    assert new_messages[0].name == "alice"


# -------------------------------------------------------------------------------------------
# Multipart messages — the bug fix. The original implementation skipped ANY non-plain-string
# user message outright (so the marker was dead on every multipart request, e.g. every request
# an image-aware client sends). Now each TextPart is scanned; ImageParts are left untouched.
# -------------------------------------------------------------------------------------------


def test_multipart_message_marker_in_a_text_part_is_found_and_stripped():
    messages = _msgs(
        MessageIR(
            role="user",
            content=(
                TextPart(text="Look at this. <<SYSTEM>>Be terse.<</SYSTEM>>"),
                ImagePart(url="https://x/i.jpg"),
            ),
        )
    )
    new_messages, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Be terse."
    new_content = new_messages[0].content
    assert isinstance(new_content, tuple)
    assert new_content[0] == TextPart(text="Look at this.")
    assert new_content[1] == ImagePart(url="https://x/i.jpg")  # untouched


def test_multipart_message_with_no_marker_stops_the_scan_there():
    """Once we reach a multipart last-user-message, it's inspected like any other — if it has no
    block, the scan stops (does not fall through to an earlier turn)."""
    messages = _msgs(
        MessageIR(role="user", content="<<SYSTEM>>old<</SYSTEM>>"),
        MessageIR(role="assistant", content="ok"),
        MessageIR(
            role="user", content=(TextPart(text="no marker"), ImagePart(url="https://x/i.jpg"))
        ),
    )
    _, directives = extract_system_block(messages)
    assert directives is None


def test_multipart_message_only_the_first_matching_text_part_is_touched():
    messages = _msgs(
        MessageIR(
            role="user",
            content=(
                TextPart(text="first part, no marker"),
                TextPart(text="<<SYSTEM>>Be terse.<</SYSTEM>>"),
            ),
        )
    )
    new_messages, directives = extract_system_block(messages)
    assert directives is not None
    content = new_messages[0].content
    assert content[0] == TextPart(text="first part, no marker")  # untouched
    assert content[1] == TextPart(text="")


# -------------------------------------------------------------------------------------------
# <<SYSTEM>> — tag flexibility (case-insensitive, tolerant closer).
# -------------------------------------------------------------------------------------------


def test_system_marker_is_case_insensitive():
    messages = _msgs(MessageIR(role="user", content="<<system>>Be terse.<</system>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Be terse."


def test_system_marker_accepts_single_lt_closer():
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>Be terse.</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Be terse."


def test_bare_system_body_with_no_headers_is_just_an_instruction():
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>Reply in Farsi.<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives == SystemDirectives(instruction="Reply in Farsi.")


# -------------------------------------------------------------------------------------------
# <<SYSTEM>> — directive header parsing.
# -------------------------------------------------------------------------------------------


def test_parses_skip_and_instruction_together():
    body = "exclude: k3, glm52\nAnswer as a terse numbered plan, no preamble."
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ("k3", "glm52")
    assert directives.instruction == "Answer as a terse numbered plan, no preamble."


def test_directive_only_block_has_no_instruction():
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>exclude: qwn37max<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ("qwn37max",)
    assert directives.instruction is None


def test_whitespace_separated_and_mixed_case_values_normalize():
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>exclude: K3 GLM52,\n<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ("k3", "glm52")


def test_multiple_lines_of_the_same_key_accumulate_and_dedupe():
    body = "exclude: k3\nexclude: glm52, k3\n"
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ("k3", "glm52")


def test_only_and_show_work_and_synth_all_parse():
    body = "only: oai56s, cl48op\nshow_work: off\nsynth: g35fl\n"
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives == SystemDirectives(
        instruction=None, exclude=(), only=("oai56s", "cl48op"), show_work="off", synth="g35fl"
    )


def test_opening_tag_on_its_own_line_still_parses_the_directive_header():
    # This is how a human actually types the block (press Enter after `<<SYSTEM>>`, then write
    # directives) and exactly how the README documents it — must not be confused with the real
    # blank-line escape hatch (see the test above).
    content = (
        "<<SYSTEM>>\nexclude: k3\nsynth: cl48op\n<</SYSTEM>>\n\nwhat is the capital of france?"
    )
    messages = _msgs(MessageIR(role="user", content=content))
    stripped, directives = extract_system_block(messages)
    assert directives == SystemDirectives(
        instruction=None, exclude=("k3",), only=(), show_work=None, synth="cl48op"
    )
    assert stripped[0].text.strip() == "what is the capital of france?"


def test_readme_documented_multi_directive_example_parses_as_documented():
    content = (
        "Compare these two approaches.\n"
        "<<SYSTEM>>\n"
        "exclude: k3, glm52\n"
        "only: oai56s, cl48op\n"
        "show_work: off\n"
        "synth: cl48op\n"
        "Weigh whichever response cites real sources most heavily.\n"
        "<</SYSTEM>>"
    )
    messages = _msgs(MessageIR(role="user", content=content))
    _, directives = extract_system_block(messages)
    assert directives == SystemDirectives(
        instruction="Weigh whichever response cites real sources most heavily.",
        exclude=("k3", "glm52"),
        only=("oai56s", "cl48op"),
        show_work="off",
        synth="cl48op",
    )


def test_unknown_directive_key_raises_400():
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>skpi: k3\ntext<</SYSTEM>>"))
    with pytest.raises(InvalidRequestError, match="skpi"):
        extract_system_block(messages)


def test_body_starting_with_non_directive_line_becomes_instruction_verbatim():
    # "Note:" LOOKS key-shaped but appears before any directive has been recognized, so it is
    # subject to the same unknown-key rule as anywhere else in the header zone.
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>Note: be careful<</SYSTEM>>"))
    with pytest.raises(InvalidRequestError):
        extract_system_block(messages)


def test_leading_blank_line_is_the_escape_hatch_for_key_shaped_prose():
    # "Leave a blank line before it" (per the README) means a genuinely empty line — two newlines,
    # not one. The one newline that unavoidably follows `<<SYSTEM>>` when it's written on its own
    # line (see the header-parsing block below) must NOT by itself be mistaken for this.
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>\n\nNote: be careful<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Note: be careful"


def test_a_single_newline_after_the_opening_tag_is_not_the_blank_line_escape_hatch():
    # Regression for a real bug: writing `<<SYSTEM>>` on its own line (the only way a human types
    # it, and exactly how the README's own multi-directive example is formatted) put exactly one
    # newline at the start of the captured body. `splitlines()` on a string starting with `\n`
    # yields a leading empty string, which the parser mistook for the deliberate blank-line escape
    # hatch above — silently discarding every directive and turning the whole header into inert
    # instruction text. `exclude:`/`synth:`/etc. must still be parsed here, not skipped.
    messages = _msgs(MessageIR(role="user", content="<<SYSTEM>>\nNote: be careful<</SYSTEM>>"))
    with pytest.raises(InvalidRequestError, match="note"):
        extract_system_block(messages)


def test_instruction_terminator_key_is_the_other_escape_hatch():
    messages = _msgs(
        MessageIR(role="user", content="<<SYSTEM>>instruction: Note: be careful<</SYSTEM>>")
    )
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction == "Note: be careful"


def test_instruction_terminator_with_no_inline_value_uses_the_next_lines():
    body = "exclude: k3\ninstruction:\nNote: be careful\nMore text."
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.exclude == ("k3",)
    assert directives.instruction == "Note: be careful\nMore text."


def test_a_directive_shaped_line_after_legitimate_directives_still_400s():
    # Once we're mid-header-scan, a key-shaped-but-unknown line is still an error, not silently
    # folded into the instruction — the parser can't tell "meant to end the header" from "typo".
    body = "exclude: k3\nWarning: something\nmore text"
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    with pytest.raises(InvalidRequestError, match="warning"):
        extract_system_block(messages)


def test_only_directives_no_instruction_when_every_line_is_a_directive():
    body = "exclude: k3\nonly: a, b\n"
    messages = _msgs(MessageIR(role="user", content=f"<<SYSTEM>>{body}<</SYSTEM>>"))
    _, directives = extract_system_block(messages)
    assert directives is not None
    assert directives.instruction is None
    assert directives.exclude == ("k3",)
    assert directives.only == ("a", "b")
