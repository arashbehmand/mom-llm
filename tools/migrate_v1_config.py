"""One-time migration: v1 ``config.yaml`` -> v2 config.

This is throwaway tooling, not part of the shipped package. It reads a v1 config through the
*existing* v1 loader (so all base/variant/alias expansion and reference validation happens with
battle-tested code), then emits the equivalent v2 config — migrating the *meaning*, not the text.

Names containing ``:`` or ``+`` (v1's variant/alias conventions) are sanitized to ``-`` since v2
reserves those characters. The already-expanded v1 LLMs become flat v2 ``llms`` (no ``extends``);
effort profiles are opt-in and deliberately not synthesized here. Run once, review, discard.

    python -m tools.migrate_v1_config config.yaml -o models.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

from mom_service.config import LLMDefinition, ModelConfig, MoMConfig
import yaml

from mom.config.resolve import infer_key_env_candidates


def sanitize_name(name: str) -> str:
    """Map a v1 name to a v2-legal name (``:`` and ``+`` -> ``-``)."""
    return name.replace(":", "-").replace("+", "-")


def _format_seconds(seconds: int) -> str:
    if seconds and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _migrate_llm(llm: LLMDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {"model": llm.model}
    if llm.api_mode == "responses":
        out["api"] = "responses"
    # Emit api_key_env only when it is NOT inferable from the provider (keeps the config clean;
    # e.g. gemini's v1 GOOGLE_API_KEY is a v2 candidate, so it is dropped and v2 infers GEMINI).
    candidates = infer_key_env_candidates(llm.model)
    if llm.api_key_env and llm.api_key_env not in candidates:
        out["api_key_env"] = llm.api_key_env
    if llm.proxy_url_env:
        out["proxy_url_env"] = llm.proxy_url_env
    if llm.params:
        out["params"] = llm.params
    if llm.pricing is not None:
        pricing: dict[str, float] = {}
        if llm.pricing.prompt_cost_per_token is not None:
            pricing["input_per_1m"] = round(llm.pricing.prompt_cost_per_token * 1_000_000, 6)
        if llm.pricing.completion_cost_per_token is not None:
            pricing["output_per_1m"] = round(llm.pricing.completion_cost_per_token * 1_000_000, 6)
        if llm.pricing.reasoning_cost_per_token is not None:
            pricing["reasoning_per_1m"] = round(llm.pricing.reasoning_cost_per_token * 1_000_000, 6)
        if pricing:
            out["pricing"] = pricing
    return out


def _migrate_ensemble(model: ModelConfig) -> dict[str, Any]:
    members = [{"llm": sanitize_name(name)} for name in model.llms_to_query]
    synthesizer: dict[str, Any] = {"llm": sanitize_name(model.concluding_llm)}
    if model.concluding_prompt:
        synthesizer["prompt"] = model.concluding_prompt
    out: dict[str, Any] = {"members": members, "synthesizer": synthesizer}
    if model.include_thinking_context:
        out["show_work"] = "inline"
    return out


def migrate(v1: MoMConfig) -> dict[str, Any]:
    """Convert a loaded v1 config into a v2 config dict."""
    out: dict[str, Any] = {"version": 2}

    out["defaults"] = {
        "call": {
            "timeout": _format_seconds(v1.service.timeout_seconds),
            "retries": v1.service.max_llm_retries,
            "retry_backoff": _format_seconds(v1.service.llm_retry_delay_seconds),
        }
    }
    out["cache"] = {"enabled": v1.service.cache_enabled}

    if v1.langfuse is not None:
        langfuse: dict[str, Any] = {"enabled": True}
        for field, default in (
            ("public_key_env", "LANGFUSE_PUBLIC_KEY"),
            ("secret_key_env", "LANGFUSE_SECRET_KEY"),
            ("host_env", "LANGFUSE_HOST"),
        ):
            value = getattr(v1.langfuse, field)
            if value != default:
                langfuse[field] = value
        out["observability"] = {"langfuse": langfuse}

    out["llms"] = {sanitize_name(llm.name): _migrate_llm(llm) for llm in v1.llm_definitions}
    if v1.prompt_definitions:
        out["prompts"] = {p.name: p.content for p in v1.prompt_definitions}
    out["ensembles"] = {sanitize_name(m.name): _migrate_ensemble(m) for m in v1.models}
    return out


def _warnings(v1: MoMConfig) -> list[str]:
    notes = []
    if v1.service.exposed_apis and v1.service.exposed_apis != ["openai"]:
        notes.append("`service.exposed_apis` was never enforced in v1 and is dropped in v2.")
    notes.append(
        "v2 now ACTS on client reasoning_effort/thinking/reasoning params (v1 dropped them). "
        "Consider adding `effort_tiers` to ensembles you want tier-selectable."
    )
    notes.append(
        "Anthropic `cache_control` prompt caching is on by default (see `provider_cache`)."
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a v1 mom-llm config.yaml to v2.")
    parser.add_argument("input", help="Path to the v1 config.yaml")
    parser.add_argument("-o", "--output", help="Write v2 YAML here (default: stdout)")
    args = parser.parse_args()

    from mom_service.config import load_config as load_v1

    v1 = load_v1(args.input)
    v2 = migrate(v1)
    text = yaml.safe_dump(v2, sort_keys=False, allow_unicode=True, width=100)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    for note in _warnings(v1):
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
