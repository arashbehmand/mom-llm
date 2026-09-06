"""The MCP tools' published result shapes.

These models ARE the contract: the SDK derives each tool's ``outputSchema`` from its return
annotation, so a field added here appears in every client's tool listing. Pydantic models rather
than plain dicts for exactly that reason — a documented schema an agent can read before calling.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PricingInfo(BaseModel):
    """Per-1M-token USD rates. Every field optional: providers price different axes."""

    input_per_1m: float | None = None
    output_per_1m: float | None = None
    reasoning_per_1m: float | None = None
    cache_read_per_1m: float | None = None
    cache_write_per_1m: float | None = None


class LlmInfo(BaseModel):
    """One catalog llm — a base or an expanded variant, indistinguishable by design."""

    name: str = Field(description="Catalog name; use this in an inline consult panel.")
    model: str = Field(description="Provider model id the call is made against.")
    api: str = Field(description="Wire the llm is called over: chat | responses.")
    vision: bool | None = None
    tools: bool | None = None
    reasoning: bool | None = None
    web_search: bool = Field(default=False, description="A `search:` block is configured.")
    context_length: int | None = None
    max_output_tokens: int | None = None
    max_input_tokens: int | None = None
    timeout_seconds: float | None = None
    pricing: PricingInfo | None = None
    pricing_source: Literal["config", "litellm", "unknown"] = Field(
        default="unknown",
        description=(
            "Where `pricing` came from: `config` (a declared pricing: block, what mom bills "
            "against), `litellm` (the pinned catalog, indicative only), `unknown` (neither knows "
            "this model — it may record as $0)."
        ),
    )


class MemberInfo(BaseModel):
    identity: str = Field(description="Distinct name within the ensemble (`as:` or the llm name).")
    llm: str
    model: str
    effort_by_tier: dict[str, str] = Field(default_factory=dict)


class EnsembleInfo(BaseModel):
    """One configured ensemble, as advertised to clients."""

    name: str
    description: str | None = None
    strategy: str
    members: list[MemberInfo] = Field(default_factory=list)
    synthesizer: MemberInfo
    effort_tiers: list[str] = Field(default_factory=list)
    default_tier: str | None = None
    show_work: str = "off"
    tool_strategy: str = "arbitrate"
    supports_tools: bool = True
    supports_vision: bool = True
    supports_reasoning: bool = False
    supports_web_search: bool = False
    context_length: int | None = None
    max_output_tokens: int | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_write_tokens: int = 0


class ErrorInfo(BaseModel):
    code: str
    message: str = Field(description="Client-safe text; operator detail stays in the gateway log.")
    http_status: int = 502


class MemberReport(BaseModel):
    """One member's contribution to a consult — including the ones that did not contribute."""

    identity: str
    llm: str = Field(
        description=(
            "The member's identity within the ensemble, which is how the engine and the metrics "
            "ledger label it. For a member listed under an `as:` alias that is the alias, not the "
            "catalog llm name — `model` is the provider model that actually ran."
        )
    )
    model: str
    status: str = Field(
        description=(
            "ok | empty | error | timeout | skipped | aborted | abandoned. `abandoned` means the "
            "fan-out deadline passed before this member answered."
        )
    )
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    cached: bool = False
    finish_reason: str | None = None
    error: str | None = None
    answer: str | None = Field(default=None, description="Only when include_member_answers.")
    reasoning: str | None = Field(default=None, description="Only when include_member_answers.")


class ConsultResult(BaseModel):
    """The outcome of one panel run — the same envelope whether it answered, called a tool, or
    failed. ``status`` is the discriminator; a failed run still reports what it spent."""

    status: Literal["ok", "tool_calls", "failed"]
    answer: str = Field(default="", description="Synthesized text; empty unless status is ok.")
    reasoning: str = ""
    finish_reason: str = ""
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="OpenAI wire shape. Non-empty only when status is tool_calls.",
    )
    error: ErrorInfo | None = None
    ensemble: str
    request_id: str
    coalesced: bool = Field(
        default=False, description="This turn attached to an identical run already in flight."
    )
    progress_url: str | None = None
    total_cost_usd: float = 0.0
    usage: UsageInfo = Field(default_factory=UsageInfo)
    members: list[MemberReport] = Field(default_factory=list)
    notices: list[str] = Field(
        default_factory=list,
        description=(
            "What a `<<SYSTEM>>` block in the prompt asked for and didn't get — an unknown "
            "member name, an unusable directive value. The run went ahead without it."
        ),
    )


class RunMemberReport(BaseModel):
    identity: str
    model: str | None = None
    status: str | None = Field(default=None, description="Null while the member is still running.")
    duration_ms: float | None = None
    cost_usd: float | None = None


class InFlightRun(BaseModel):
    request_id: str
    ensemble: str
    state: Literal["running", "synthesizing", "completed", "failed"]
    started_at: float
    updated_at: float
    members_total: int | None = None
    members: list[RunMemberReport] = Field(default_factory=list)
    cost_usd: float = 0.0
    finish_reason: str | None = None
    detail: str | None = None


class RecentRun(BaseModel):
    request_id: str
    ensemble: str = ""
    started_ts: float = 0.0
    last_ts: float = 0.0
    calls: int = 0
    cost_usd: float = 0.0
    failures: int = 0
    cache_hits: int = 0


class RunCall(BaseModel):
    ts: float = 0.0
    ensemble: str = ""
    llm: str = ""
    model: str | None = None
    role: str = ""
    turn_type: str = ""
    status: str = ""
    cache_hit: bool = False
    cost_usd: float | None = None
    duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    attempts: int = 1


class RunsReport(BaseModel):
    """In-flight and finished runs, from the two sources that each know half the story."""

    in_flight: list[InFlightRun] = Field(
        default_factory=list,
        description=(
            "Live runs this process is driving. Process-local: a multi-worker gateway reports "
            "the worker that answered, and `mom mcp` reports only consults it ran itself."
        ),
    )
    just_finished: list[InFlightRun] = Field(
        default_factory=list,
        description=(
            "Runs this process finished recently, with their final state and cost. Covers the "
            "window before the metrics recorder flushes, where `recent` cannot see them yet; "
            "the same run appears in both once it has."
        ),
    )
    recent: list[RecentRun] = Field(
        default_factory=list,
        description=(
            "Finished runs from the metrics ledger, newest first. Durable and shared across "
            "processes, but a call only lands here after it completes. Omitted when a "
            "`request_id` is given — `calls` covers that run in more detail."
        ),
    )
    calls: list[RunCall] | None = Field(
        default=None, description="Per-call detail; present only when request_id was given."
    )
    in_flight_visibility: Literal["process", "none"] = "process"


class UsageGroup(BaseModel):
    key: str
    calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    cache_hits: int = 0


class UsageReport(BaseModel):
    """Spend over a window — the same aggregation `mom metrics usage` prints."""

    window_days: float | None = Field(default=None, description="Null means all time.")
    ensemble: str | None = None
    calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    errors: int = 0
    timeouts: int = 0
    cache_hits: int = 0
    billable_calls: int = 0
    estimated_cache_savings_usd: float = 0.0
    by_ensemble: list[UsageGroup] = Field(default_factory=list)
    by_llm: list[UsageGroup] = Field(default_factory=list)
    note: str | None = Field(
        default=None,
        description="Set when metrics are unavailable, or to qualify what the numbers mean.",
    )


class CacheStats(BaseModel):
    """Response-cache occupancy. Hits are cumulative since the entry was written."""

    enabled: bool
    entries: int = 0
    bytes: int = 0
    hits: int = 0
