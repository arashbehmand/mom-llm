"""``request_identity``: the coalescing key derived from a chat request IR.

Two requests must share an identity exactly when they'd produce the same fan-out + synthesis
plan — same ensemble, same directive-stripped messages, same tools/search/sampling/effort/
resolved directives — and must NOT share one when any of those differ. Fields that are pure labels
(``request_id`` lives outside the IR entirely; ``stream``/``include_usage``/``metadata`` are on the
IR but documented as excluded) must have zero effect either way.
"""

from __future__ import annotations

from mom.domain.request import ChatRequestIR, MessageIR, Sampling, SpecificTool, ToolSpec
from mom.domain.requestkey import request_identity


def _messages(text: str = "hi") -> tuple[MessageIR, ...]:
    return (MessageIR(role="user", content=text),)


def _ir(
    *,
    model: str = "e",
    messages: tuple[MessageIR, ...] | None = None,
    tools: tuple[ToolSpec, ...] = (),
    mcp_tools: tuple[dict[str, object], ...] = (),
    tool_choice: str | SpecificTool = "auto",
    parallel_tool_calls: bool | None = None,
    effort: str | None = None,
    web_search: bool = False,
    response_format: dict[str, object] | None = None,
    sampling: Sampling | None = None,
    stream: bool = False,
    include_usage: bool = False,
    user: str | None = None,
    metadata: dict[str, str] | None = None,
) -> ChatRequestIR:
    return ChatRequestIR(
        model=model,
        messages=messages if messages is not None else _messages(),
        tools=tools,
        mcp_tools=mcp_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        effort=effort,
        web_search=web_search,
        response_format=response_format,
        sampling=sampling if sampling is not None else Sampling(),
        stream=stream,
        include_usage=include_usage,
        user=user,
        metadata=metadata or {},
    )


def test_identical_requests_share_an_identity():
    assert request_identity(_ir()) == request_identity(_ir())


def test_request_id_has_no_representation_in_the_ir_at_all():
    # request_identity only ever sees a ChatRequestIR — request_id/X-Request-Id live on the
    # router side and never reach this function, so two calls built from otherwise-identical IRs
    # are indistinguishable no matter what id a caller minted for either one.
    assert request_identity(_ir()) == request_identity(_ir())


def test_stream_include_usage_and_metadata_do_not_affect_identity():
    a = _ir(stream=True, include_usage=True, metadata={"trace": "x"})
    b = _ir(stream=False, include_usage=False, metadata={"trace": "y"})
    assert request_identity(a) == request_identity(b)


def test_different_model_differs():
    assert request_identity(_ir(model="e")) != request_identity(_ir(model="f"))


def test_different_message_content_differs():
    a = _ir(messages=_messages("hi"))
    b = _ir(messages=_messages("bye"))
    assert request_identity(a) != request_identity(b)


def test_different_effort_differs():
    assert request_identity(_ir(effort="low")) != request_identity(_ir(effort="high"))


def test_different_sampling_differs():
    a = _ir(sampling=Sampling(temperature=0.1))
    b = _ir(sampling=Sampling(temperature=0.9))
    assert request_identity(a) != request_identity(b)


def test_seed_is_part_of_sampling_and_differs():
    a = _ir(sampling=Sampling(seed=1))
    b = _ir(sampling=Sampling(seed=2))
    assert request_identity(a) != request_identity(b)


def test_different_tools_differ():
    a = _ir(tools=(ToolSpec(name="foo"),))
    b = _ir(tools=(ToolSpec(name="bar"),))
    assert request_identity(a) != request_identity(b)


def test_different_tool_choice_differs():
    a = _ir(tool_choice="auto")
    b = _ir(tool_choice=SpecificTool(name="foo"))
    assert request_identity(a) != request_identity(b)


def test_different_parallel_tool_calls_differs():
    a = _ir(parallel_tool_calls=True)
    b = _ir(parallel_tool_calls=False)
    assert request_identity(a) != request_identity(b)


def test_web_search_flag_differs():
    assert request_identity(_ir(web_search=True)) != request_identity(_ir(web_search=False))


def test_different_mcp_tools_differ():
    a = _ir(mcp_tools=({"type": "mcp", "server_label": "x"},))
    b = _ir(mcp_tools=({"type": "mcp", "server_label": "y"},))
    assert request_identity(a) != request_identity(b)


def test_different_response_format_differs():
    a = _ir(response_format={"type": "json_object"})
    b = _ir(response_format=None)
    assert request_identity(a) != request_identity(b)


def test_different_user_differs():
    assert request_identity(_ir(user="alice")) != request_identity(_ir(user="bob"))


def test_directive_formatting_differences_collapse_to_the_same_identity():
    # comma- vs. whitespace-separated `only:` lists parse to the identical resolved directive —
    # the whole point of folding in RESOLVED directives rather than raw block text. (No leading
    # blank line after the opening tag — that's the documented escape hatch that disables header
    # parsing entirely; see test_directives.py.)
    a = _ir(messages=_messages("<<SYSTEM>>only: a, b\nhi<</SYSTEM>>"))
    b = _ir(messages=_messages("<<SYSTEM>>only: a b\nhi<</SYSTEM>>"))
    assert request_identity(a) == request_identity(b)


def test_different_resolved_directives_differ():
    a = _ir(messages=_messages("<<SYSTEM>>only: a\nhi<</SYSTEM>>"))
    b = _ir(messages=_messages("<<SYSTEM>>only: b\nhi<</SYSTEM>>"))
    assert request_identity(a) != request_identity(b)


def test_directive_block_is_stripped_before_hashing_like_the_plan_resolver_sees_it():
    # A directive-only difference in show_work must move the identity even though the VISIBLE
    # instruction text (what a member/synth would see) is identical — the directive is part of
    # the plan, not just decoration on the message.
    a = _ir(messages=_messages("<<SYSTEM>>show_work: off\nsame instruction<</SYSTEM>>"))
    b = _ir(messages=_messages("<<SYSTEM>>show_work: inline\nsame instruction<</SYSTEM>>"))
    assert request_identity(a) != request_identity(b)
