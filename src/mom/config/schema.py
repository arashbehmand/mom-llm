"""The v2 configuration schema (Pydantic v2).

Design goals (see the approved plan): maps keyed by name (no ``base:suffix`` string sprawl),
one inheritance primitive (``extends``), a compact per-member effort matrix aligned to each
ensemble's ``effort_tiers``, ``extra="forbid"`` everywhere (typos are errors), and secrets by
env-var *name* only. Cross-reference and ``extends`` resolution happen in ``resolve.py``; this
module is pure structural validation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from mom.config.types import (
    ByteSize,
    Duration,
    EffortLevel,
    normalize_effort_cell,
    parse_effort_level,
)


EffortCell = Annotated[str, BeforeValidator(normalize_effort_cell)]
Tier = Annotated[EffortLevel, BeforeValidator(parse_effort_level)]
EnvVarName = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ----------------------------------------------------------------------------------------------
# server / defaults / cache / storage / observability / budgets
# ----------------------------------------------------------------------------------------------
class CorsConfig(_Model):
    origins: list[str] = Field(default_factory=list)
    allow_credentials: bool = False

    @model_validator(mode="after")
    def _no_wildcard_with_credentials(self) -> CorsConfig:
        if self.allow_credentials and "*" in self.origins:
            raise ValueError("cors.allow_credentials cannot be true with a wildcard '*' origin")
        return self


class ServerConfig(_Model):
    auth: Literal["bearer", "none"] = "bearer"
    public_url: str | None = None
    cors: CorsConfig = Field(default_factory=CorsConfig)
    # Emit an SSE keepalive comment (`: ...`) on the response stream whenever it goes idle this
    # long, so a slow fan-out doesn't trip a client's idle read-timeout. null = off.
    stream_heartbeat: Duration | None = None


class CallDefaults(_Model):
    timeout: Duration = timedelta(minutes=20)
    retries: int = Field(default=3, ge=0)
    retry_backoff: Duration = timedelta(seconds=2)


class FanoutDefaults(_Model):
    max_concurrency: int | None = Field(default=None, ge=1)
    min_results: int = Field(default=1, ge=0)
    deadline: Duration | None = None
    # When the client disconnects mid-fan-out, let the in-flight member calls finish (and cache) in
    # the background instead of cancelling them — so a retry of the same turn hits cache and goes
    # straight to synthesis. Default false keeps the zero-orphaned-spend behavior.
    detach_on_disconnect: bool = False


class AnthropicCache(_Model):
    enabled: bool = True
    ttl: Duration = timedelta(minutes=5)


class OpenAICache(_Model):
    prompt_cache_key: Literal["auto", "off"] = "auto"


class ProviderCacheConfig(_Model):
    anthropic: AnthropicCache = Field(default_factory=AnthropicCache)
    openai: OpenAICache = Field(default_factory=OpenAICache)


class DefaultsConfig(_Model):
    call: CallDefaults = Field(default_factory=CallDefaults)
    fanout: FanoutDefaults = Field(default_factory=FanoutDefaults)
    provider_cache: ProviderCacheConfig = Field(default_factory=ProviderCacheConfig)


class CacheConfig(_Model):
    enabled: bool = True
    ttl: Duration = timedelta(days=14)
    max_size: ByteSize = 1024**3  # 1 GiB
    coalesce: bool = True


class StorageConfig(_Model):
    data_dir: str | None = None


class LangfuseObs(_Model):
    enabled: bool = False
    public_key_env: EnvVarName = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: EnvVarName = "LANGFUSE_SECRET_KEY"  # noqa: S105 (env var NAME, not a secret)
    host_env: EnvVarName = "LANGFUSE_HOST"


class OtelObs(_Model):
    """OpenTelemetry tracing via OTLP (GenAI semantic conventions). Deps are optional."""

    enabled: bool = False
    endpoint: str = "http://localhost:4318"  # OTLP collector (http/protobuf default port)
    protocol: Literal["http", "grpc"] = "http"
    service_name: str = "mom-llm"


class ObservabilityConfig(_Model):
    langfuse: LangfuseObs = Field(default_factory=LangfuseObs)
    otel: OtelObs = Field(default_factory=OtelObs)


class BudgetsConfig(_Model):
    daily_usd: float | None = Field(default=None, ge=0)
    per_ensemble: dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# llms
# ----------------------------------------------------------------------------------------------
class PricingConfig(_Model):
    input_per_1m: float | None = Field(default=None, ge=0)
    output_per_1m: float | None = Field(default=None, ge=0)
    reasoning_per_1m: float | None = Field(default=None, ge=0)
    cache_read_per_1m: float | None = Field(default=None, ge=0)
    cache_write_per_1m: float | None = Field(default=None, ge=0)


class CapabilityOverride(_Model):
    context_length: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    vision: bool | None = None
    tools: bool | None = None
    reasoning: bool | None = None


_RESERVED_PARAM_KEYS = frozenset(
    {"model", "messages", "stream", "api_key", "num_retries", "timeout"}
)


class LlmVariantConfig(_Model):
    """One entry in an llm's ``variants:`` map — expands to a sibling llm named ``<parent>-<key>``.

    Sugar over ``extends``: inherits the parent's ``model``/``api``/``api_key_env``/
    ``proxy_url_env`` unless overridden here, and deep-merges ``params``. Deliberately does NOT
    inherit capability-ish fields (``search``/``pricing``/``capabilities``/...) — a family of
    effort variants shouldn't silently gain the parent's web-search capability, say. Set one of
    those explicitly on a variant in the rare case it's actually wanted.
    """

    model: str | None = None
    api: Literal["chat", "responses"] | None = None
    api_key_env: EnvVarName | None = None
    proxy_url_env: EnvVarName | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class LlmConfig(_Model):
    model: str | None = None  # required directly or via `extends`
    extends: str | None = None
    api: Literal["chat", "responses"] = "chat"
    api_key_env: EnvVarName | None = None  # inferred from provider when omitted
    proxy_url_env: EnvVarName | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # Provider-specific params merged in when the client requests web search. Presence (even an
    # empty {} for inherently-online models) marks the LLM as search-capable.
    search: dict[str, Any] | None = None
    pricing: PricingConfig | None = None
    capabilities: CapabilityOverride | None = None
    max_input_tokens: int | None = Field(default=None, ge=1)
    timeout: Duration | None = None
    cache_ttl: Duration | None = None
    # A compact way to author an effort/variant family without repeating `model:` per sibling —
    # see LlmVariantConfig. Expanded into ordinary sibling llm entries at resolve time.
    variants: dict[str, LlmVariantConfig] | None = None

    @model_validator(mode="after")
    def _reject_reserved_params(self) -> LlmConfig:
        clashes = _RESERVED_PARAM_KEYS & set(self.params)
        if clashes:
            raise ValueError(
                f"params may not contain reserved keys {sorted(clashes)} "
                "(they are set by the gateway)"
            )
        return self


# ----------------------------------------------------------------------------------------------
# ensembles
# ----------------------------------------------------------------------------------------------
# A per-member effort spec: a single cell (applies to every tier), a positional list (one cell
# per tier, aligned to `effort_tiers`), or an explicit {tier: cell} map. Resolution validates
# alignment against the ensemble's tiers.
EffortSpec = EffortCell | list[EffortCell] | dict[str, EffortCell]


class MemberConfig(_Model):
    llm: str
    member_as: str | None = Field(default=None, alias="as")
    effort: EffortSpec | None = None

    @property
    def identity(self) -> str:
        """Distinct identity within an ensemble (supports the same llm listed twice)."""
        return self.member_as or self.llm


class AllMembersConfig(_Model):
    """``members: all`` (or ``{all: true, exclude: [...]}``) — every llm in the catalog (bases and
    expanded variants alike) becomes a member, so a debug/eval panel never needs manual upkeep as
    llms are added or removed. ``exclude`` opts specific llms out by name (e.g. slow/costly
    special-purpose variants that don't belong in a routine debug fan-out). Expanded in
    resolve.py, where the full catalog is known.
    """

    all: Literal[True] = True
    exclude: list[str] = Field(default_factory=list)


class SynthesizerConfig(_Model):
    llm: str
    prompt: str | None = None
    # Used instead of `prompt` only when the request has web_search=true — request-triggered,
    # like `search:` on an llm. Falls back to `prompt` when unset or web_search is false.
    search_prompt: str | None = None
    effort: EffortSpec | None = None


class EnsembleToolsConfig(_Model):
    continuation: Literal["relay", "fanout"] = "relay"
    member_tool_context: Literal["summary", "none"] = "summary"
    # How a tool call is chosen. `arbitrate` (default): the synthesizer decides, seeing member
    # proposals as advisory context. `vote`: return the call proposed by >= `vote_threshold`
    # members directly (skip synthesis). `first`: return the first member's proposed call.
    strategy: Literal["arbitrate", "vote", "first"] = "arbitrate"
    vote_threshold: int = Field(default=2, ge=1)
    # Streaming tool-call delta shape. `compat` (default) re-emits id/type/name on every delta
    # (safe for AI-SDK-style clients); `strict` sends them only on the first delta.
    stream_profile: Literal["compat", "strict"] = "compat"


class EnsembleConfig(_Model):
    description: str | None = None
    strategy: Literal["synthesize", "passthrough"] = "synthesize"
    effort_tiers: list[Tier] | None = None
    default_tier: Tier | None = None
    # See AllMembersConfig for the "all"/{all: true, exclude: [...]} kitchen-sink shorthand.
    members: AllMembersConfig | list[MemberConfig] = Field(default_factory=list)
    synthesizer: SynthesizerConfig
    show_work: Literal["off", "inline", "native"] = "off"
    tools: EnsembleToolsConfig = Field(default_factory=EnsembleToolsConfig)
    advertise: dict[str, Any] = Field(default_factory=dict)
    on_input_overflow: Literal["skip", "reject"] = "skip"

    @field_validator("show_work", mode="before")
    @classmethod
    def _coerce_show_work(cls, value: object) -> object:
        # YAML parses `off`/`on` as booleans; map them back to the intended tokens.
        if value is False:
            return "off"
        if value is True:
            return "inline"
        return value

    @field_validator("members", mode="before")
    @classmethod
    def _coerce_bare_member_names(cls, value: object) -> object:
        # `members: all` is shorthand for `members: {all: true}` (no exclusions).
        if value == "all":
            return {"all": True}
        # `members: [name, ...]` is shorthand for `members: [{llm: name}, ...]` — for a panel
        # with no per-member effort override (e.g. a debug/kitchen-sink ensemble), a flow-style
        # list of names is far more compact than one `- llm: name` mapping per line.
        if not isinstance(value, list):
            return value
        return [{"llm": item} if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def _validate_tiers_and_members(self) -> EnsembleConfig:
        if self.effort_tiers is not None:
            if len(set(self.effort_tiers)) != len(self.effort_tiers):
                raise ValueError("effort_tiers must not contain duplicates")
            if self.default_tier is None:
                raise ValueError("default_tier is required when effort_tiers is set")
            if self.default_tier not in self.effort_tiers:
                raise ValueError(
                    f"default_tier {self.default_tier.label!r} is not one of effort_tiers "
                    f"{[tier.label for tier in self.effort_tiers]}"
                )
        elif self.default_tier is not None:
            raise ValueError("default_tier requires effort_tiers to be set")

        if self.strategy == "synthesize" and not self.members:
            raise ValueError("a 'synthesize' ensemble needs at least one member")
        if self.strategy == "passthrough" and (
            isinstance(self.members, AllMembersConfig) or len(self.members) > 1
        ):
            raise ValueError("a 'passthrough' ensemble takes at most one member")

        if isinstance(self.members, list):
            seen: set[str] = set()
            for member in self.members:
                if member.identity in seen:
                    raise ValueError(f"duplicate member identity {member.identity!r} in ensemble")
                seen.add(member.identity)
        return self


# ----------------------------------------------------------------------------------------------
# top-level
# ----------------------------------------------------------------------------------------------
class Config(_Model):
    version: Literal[2]
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    llms: dict[str, LlmConfig]
    prompts: dict[str, str] = Field(default_factory=dict)
    ensembles: dict[str, EnsembleConfig]

    @model_validator(mode="after")
    def _no_reserved_chars_in_names(self) -> Config:
        for scope, names in (("llms", self.llms), ("ensembles", self.ensembles)):
            for name in names:
                if ":" in name or "+" in name:
                    raise ValueError(
                        f"{scope} name {name!r} may not contain ':' or '+' (reserved characters)"
                    )
        return self
