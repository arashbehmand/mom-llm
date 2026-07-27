"""Resolve a validated :class:`~mom.config.schema.Config` into an immutable catalog.

Resolution does the three things structural validation cannot:

1. ``extends`` chains — deep-merging ``params`` (a ``null`` value deletes an inherited key) and
   inheriting explicitly-set scalar fields, with cycle and missing-target detection.
2. Provider → ``api_key_env`` inference where omitted.
3. Cross-references (members / synthesizer / prompt targets exist) and the per-member effort
   matrix aligned to each ensemble's ``effort_tiers``.

The result is a frozen ``ResolvedCatalog`` the rest of the app consumes; the input ``Config`` is
never mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mom.config.schema import (
    Config,
    EnsembleConfig,
    LlmConfig,
)
from mom.config.types import (
    EFFORT_OFF,
    EFFORT_SKIP,
    EffortLevel,
    parse_effort_level,
)


class ConfigError(ValueError):
    """A configuration is structurally valid but semantically inconsistent."""


# Provider prefix -> ordered candidate env var names (first set wins at call time).
_PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "vertex_ai": ("GOOGLE_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
}


def infer_key_env_candidates(model: str) -> tuple[str, ...]:
    """Return the ordered env-var candidates for a ``provider/model`` string."""
    provider = model.split("/", 1)[0] if "/" in model else ""
    return _PROVIDER_KEY_ENV.get(provider, ())


@dataclass(frozen=True, slots=True)
class ResolvedLlm:
    name: str
    model: str
    api: str
    api_key_env: str | None
    key_env_candidates: tuple[str, ...]
    proxy_url_env: str | None
    params: Mapping[str, Any]
    search: Mapping[str, Any] | None
    pricing: Any | None
    capabilities: Any | None
    max_input_tokens: int | None
    timeout: Any | None
    cache_ttl: Any | None


@dataclass(frozen=True, slots=True)
class ResolvedMember:
    identity: str
    llm: str
    # tier -> effort token (level label or "pass"/"off"; "skip" tiers are omitted). Empty for
    # non-tiered ensembles (the member simply runs its llm's configured params).
    effort_by_tier: Mapping[EffortLevel, str]


@dataclass(frozen=True, slots=True)
class ResolvedSynthesizer:
    llm: str
    prompt: str | None
    effort_by_tier: Mapping[EffortLevel, str]


@dataclass(frozen=True, slots=True)
class ResolvedEnsemble:
    name: str
    description: str | None
    strategy: str
    effort_tiers: tuple[EffortLevel, ...] | None
    default_tier: EffortLevel | None
    members: tuple[ResolvedMember, ...]
    synthesizer: ResolvedSynthesizer
    show_work: str
    tools_continuation: str
    member_tool_context: str
    tool_strategy: str
    vote_threshold: int
    stream_profile: str
    advertise: Mapping[str, Any]
    on_input_overflow: str

    def members_at(self, tier: EffortLevel | None) -> tuple[ResolvedMember, ...]:
        """Members that participate at ``tier`` (drops any marked ``skip`` there)."""
        if tier is None or self.effort_tiers is None:
            return self.members
        return tuple(m for m in self.members if m.effort_by_tier.get(tier) != EFFORT_SKIP)


@dataclass(frozen=True, slots=True)
class ResolvedCatalog:
    config: Config
    llms: Mapping[str, ResolvedLlm]
    ensembles: Mapping[str, ResolvedEnsemble]


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in over.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_llm(
    name: str,
    raw: Mapping[str, LlmConfig],
    memo: dict[str, ResolvedLlm],
    stack: tuple[str, ...],
) -> ResolvedLlm:
    if name in memo:
        return memo[name]
    if name in stack:
        chain = " -> ".join([*stack, name])
        raise ConfigError(f"cyclic 'extends' chain: {chain}")
    cfg = raw.get(name)
    if cfg is None:
        raise ConfigError(f"llm {name!r} referenced by 'extends' does not exist")

    if cfg.extends is None:
        model = cfg.model
        api: str = cfg.api
        api_key_env = cfg.api_key_env
        proxy_url_env = cfg.proxy_url_env
        params: dict[str, Any] = dict(cfg.params)
        search: Mapping[str, Any] | None = cfg.search
        pricing = cfg.pricing
        capabilities = cfg.capabilities
        max_input_tokens = cfg.max_input_tokens
        timeout = cfg.timeout
        cache_ttl = cfg.cache_ttl
    else:
        parent = _resolve_llm(cfg.extends, raw, memo, (*stack, name))
        provided = cfg.model_fields_set
        model = cfg.model if "model" in provided else parent.model
        api = cfg.api if "api" in provided else parent.api
        api_key_env = cfg.api_key_env if "api_key_env" in provided else parent.api_key_env
        proxy_url_env = cfg.proxy_url_env if "proxy_url_env" in provided else parent.proxy_url_env
        params = _deep_merge(parent.params, cfg.params)
        search = cfg.search if "search" in provided else parent.search
        pricing = cfg.pricing if "pricing" in provided else parent.pricing
        capabilities = cfg.capabilities if "capabilities" in provided else parent.capabilities
        max_input_tokens = (
            cfg.max_input_tokens if "max_input_tokens" in provided else parent.max_input_tokens
        )
        timeout = cfg.timeout if "timeout" in provided else parent.timeout
        cache_ttl = cfg.cache_ttl if "cache_ttl" in provided else parent.cache_ttl

    if not model:
        raise ConfigError(f"llm {name!r} has no 'model' (directly or via 'extends')")

    candidates = infer_key_env_candidates(model)
    if api_key_env is None and candidates:
        api_key_env = candidates[0]
    # If an explicit env var outside the inferred set was given, that is the only candidate.
    if api_key_env is not None and api_key_env not in candidates:
        key_candidates: tuple[str, ...] = (api_key_env,)
    else:
        key_candidates = candidates

    resolved = ResolvedLlm(
        name=name,
        model=model,
        api=api,
        api_key_env=api_key_env,
        key_env_candidates=key_candidates,
        proxy_url_env=proxy_url_env,
        params=MappingProxyType(params),
        search=MappingProxyType(search) if search is not None else None,
        pricing=pricing,
        capabilities=capabilities,
        max_input_tokens=max_input_tokens,
        timeout=timeout,
        cache_ttl=cache_ttl,
    )
    memo[name] = resolved
    return resolved


def _resolve_effort_matrix(
    spec: Any,
    tiers: tuple[EffortLevel, ...] | None,
    where: str,
) -> dict[EffortLevel, str]:
    """Resolve a member/synthesizer effort spec into a per-tier token map."""
    if tiers is None:
        # Non-tiered ensemble: the member simply runs its llm's own params at every call.
        return {}

    if spec is None:
        return dict.fromkeys(tiers, EFFORT_OFF)

    if isinstance(spec, str):
        return dict.fromkeys(tiers, spec)

    if isinstance(spec, list):
        if len(spec) != len(tiers):
            raise ConfigError(
                f"{where}: effort list has {len(spec)} entries but the ensemble has "
                f"{len(tiers)} tiers {[t.label for t in tiers]}"
            )
        return dict(zip(tiers, spec, strict=True))

    if isinstance(spec, dict):
        by_tier: dict[EffortLevel, str] = {}
        tier_set = set(tiers)
        for key, value in spec.items():
            level = parse_effort_level(key)
            if level not in tier_set:
                raise ConfigError(
                    f"{where}: effort map references tier {level.label!r} not in "
                    f"{[t.label for t in tiers]}"
                )
            by_tier[level] = value
        for tier in tiers:
            by_tier.setdefault(tier, EFFORT_OFF)
        return by_tier

    raise ConfigError(f"{where}: unsupported effort spec {spec!r}")


def _resolve_ensemble(
    name: str,
    ens: EnsembleConfig,
    llms: Mapping[str, ResolvedLlm],
    prompts: Mapping[str, str],
) -> ResolvedEnsemble:
    tiers = tuple(ens.effort_tiers) if ens.effort_tiers is not None else None

    def require_llm(llm_name: str, ref: str) -> None:
        if llm_name not in llms:
            raise ConfigError(f"ensemble {name!r} {ref} references unknown llm {llm_name!r}")

    members: list[ResolvedMember] = []
    for member in ens.members:
        require_llm(member.llm, f"member {member.identity!r}")
        members.append(
            ResolvedMember(
                identity=member.identity,
                llm=member.llm,
                effort_by_tier=MappingProxyType(
                    _resolve_effort_matrix(
                        member.effort, tiers, f"ensemble {name!r} member {member.identity!r}"
                    )
                ),
            )
        )

    require_llm(ens.synthesizer.llm, "synthesizer")
    if ens.synthesizer.prompt is not None and ens.synthesizer.prompt not in prompts:
        raise ConfigError(
            f"ensemble {name!r} synthesizer references unknown prompt {ens.synthesizer.prompt!r}"
        )
    synthesizer = ResolvedSynthesizer(
        llm=ens.synthesizer.llm,
        prompt=ens.synthesizer.prompt,
        effort_by_tier=MappingProxyType(
            _resolve_effort_matrix(ens.synthesizer.effort, tiers, f"ensemble {name!r} synthesizer")
        ),
    )

    return ResolvedEnsemble(
        name=name,
        description=ens.description,
        strategy=ens.strategy,
        effort_tiers=tiers,
        default_tier=ens.default_tier,
        members=tuple(members),
        synthesizer=synthesizer,
        show_work=ens.show_work,
        tools_continuation=ens.tools.continuation,
        member_tool_context=ens.tools.member_tool_context,
        tool_strategy=ens.tools.strategy,
        vote_threshold=ens.tools.vote_threshold,
        stream_profile=ens.tools.stream_profile,
        advertise=MappingProxyType(dict(ens.advertise)),
        on_input_overflow=ens.on_input_overflow,
    )


def _expand_variants(llms: Mapping[str, LlmConfig]) -> dict[str, LlmConfig]:
    """Expand each llm's ``variants:`` map into synthetic ``<name>-<suffix>`` sibling entries.

    Pure sugar over ``extends``: a variant becomes ``{extends: name, params: variant.params, ...}``
    resolved through the normal path below, so it inherits everything `extends` inherits — EXCEPT
    the variant's own fields are limited to model/api/api_key_env/proxy_url_env/params (see
    ``LlmVariantConfig``), so capability fields like ``search`` never propagate to a variant.
    """
    expanded = dict(llms)
    for name, cfg in llms.items():
        for suffix, variant in (cfg.variants or {}).items():
            child = f"{name}-{suffix}"
            if child in llms:
                raise ConfigError(
                    f"llm {child!r} (variant {suffix!r} of {name!r}) collides with an "
                    "explicitly defined llm of the same name"
                )
            overrides = {
                k: v
                for k, v in (
                    ("model", variant.model),
                    ("api", variant.api),
                    ("api_key_env", variant.api_key_env),
                    ("proxy_url_env", variant.proxy_url_env),
                )
                if v is not None
            }
            # Explicit None (not omission) is what makes these register as "set" below, so the
            # extends resolution overrides to None instead of inheriting the parent's value.
            expanded[child] = LlmConfig(
                extends=name,
                params=variant.params,
                search=None,
                pricing=None,
                capabilities=None,
                max_input_tokens=None,
                timeout=None,
                cache_ttl=None,
                **overrides,
            )
    return expanded


def resolve_catalog(config: Config) -> ResolvedCatalog:
    """Resolve a validated ``Config`` into an immutable catalog (raises ``ConfigError``)."""
    memo: dict[str, ResolvedLlm] = {}
    llms = _expand_variants(config.llms)
    for llm_name in llms:
        _resolve_llm(llm_name, llms, memo, ())

    ensembles: dict[str, ResolvedEnsemble] = {}
    for ens_name, ens in config.ensembles.items():
        ensembles[ens_name] = _resolve_ensemble(ens_name, ens, memo, config.prompts)

    return ResolvedCatalog(
        config=config,
        llms=MappingProxyType(memo),
        ensembles=MappingProxyType(ensembles),
    )
