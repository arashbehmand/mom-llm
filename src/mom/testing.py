"""Shipped test doubles for driving the engine without a real provider.

Public on purpose: downstream users testing a MoM deployment get a working ``FakeLLM`` and
deterministic ``ManualClock`` / ``SequentialIds`` without reimplementing the ports.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

from mom.domain.errors import UpstreamError
from mom.domain.ports import CallSpec, Completion, CompletionChunk
from mom.domain.results import Usage


class ManualClock:
    """A clock whose time only moves when you tell it to."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class SequentialIds:
    """Deterministic id factory: ``<prefix>-1``, ``<prefix>-2``, ..."""

    def __init__(self) -> None:
        self._counter = 0

    def new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"


class FakeLLM:
    """A scripted :class:`~mom.domain.ports.LLMClient`.

    ``replies`` maps an llm identity to its non-streaming content (fan-out); ``fail`` is a set of
    identities whose ``complete`` raises. ``synth_chunks`` are streamed by ``stream`` (the
    synthesizer), followed by a terminal usage/finish chunk.
    """

    def __init__(
        self,
        *,
        replies: Mapping[str, str] | None = None,
        synth_chunks: tuple[str, ...] = ("synth", "esized ", "answer"),
        fail: frozenset[str] = frozenset(),
        finish_reason: str = "stop",
        tool_calls: tuple[dict[str, str], ...] = (),
    ) -> None:
        self._replies = dict(replies or {})
        self._synth_chunks = synth_chunks
        self._fail = fail
        self._finish_reason = finish_reason
        self._tool_calls = tool_calls
        self.completions: list[CallSpec] = []
        self.streams: list[CallSpec] = []

    async def complete(self, spec: CallSpec) -> Completion:
        self.completions.append(spec)
        if spec.llm_name in self._fail:
            raise UpstreamError(f"{spec.llm_name} failed")
        content = self._replies.get(spec.llm_name, f"reply from {spec.llm_name}")
        return Completion(
            content=content,
            reasoning=None,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, spec: CallSpec) -> AsyncIterator[CompletionChunk]:
        self.streams.append(spec)
        if self._tool_calls:
            for index, call in enumerate(self._tool_calls):
                yield CompletionChunk(
                    tool_call={"index": index, "id": call["id"], "name": call["name"]}
                )
                yield CompletionChunk(tool_call={"index": index, "arguments": call["arguments"]})
            yield CompletionChunk(
                finish_reason="tool_calls",
                usage=Usage(prompt_tokens=50, completion_tokens=5),
            )
            return
        for piece in self._synth_chunks:
            yield CompletionChunk(content=piece)
        yield CompletionChunk(
            finish_reason=self._finish_reason,
            usage=Usage(prompt_tokens=50, completion_tokens=20),
        )
