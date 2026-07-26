# Migrating a v1 config to v2

v2 is a new configuration shape, not a new client protocol. **Your clients don't change** — they
keep speaking the ordinary OpenAI and Anthropic APIs against MoM (see
[PROVIDERS.md](./PROVIDERS.md)); there was never a proprietary MoM client and there still isn't.
What changes is the YAML: v1's parallel *lists* keyed by string names (`base_name:suffix`) become
v2 *maps* keyed by name, with a single inheritance primitive (`extends`) and a compact per-member
effort matrix.

Migration is a one-time, by-hand edit of your config file. This document explains the concept and
the field-by-field mapping; see [CONFIGURATION.md](./CONFIGURATION.md) for the full v2 reference.

## What changed, at a glance

| v1 | v2 |
| --- | --- |
| (no version marker) | `version: 2` (required) |
| `llm_definitions:` — a **list** with `base_name` + `variants`/`suffix` | `llms:` — a **map** keyed by name, with `extends` for variants |
| `prompt_definitions:` — a list of `{name, content}` | `prompts:` — a map of `{name: content}` |
| `models:` — a list of ensembles | `ensembles:` — a map keyed by name |
| `llms_to_query:` (list of names) | `members:` (list, each with per-tier `effort`) |
| `concluding_llm:` + `concluding_prompt:` | `synthesizer: { llm, prompt }` |
| baked-in `reasoning_effort` variants (`oai-o3:m`) | separate `llms`, **or** an ensemble `effort_tiers` matrix |
| per-**token** `pricing` | per-**1M** `pricing`, and now **optional** (cost is automatic) |
| `api_mode: responses` | `api: responses` |
| `service.timeout_seconds: 30` | `defaults.call.timeout: 30s` (duration string) |
| top-level `langfuse:` | `observability.langfuse:` |
| names like `oai-o3:m` | `:` and `+` are **reserved** — rename (e.g. `oai-o3-m`) |

## `llm_definitions` (list) → `llms` (map)

In v1, one entry could spawn several models via `variants`, each named `<base_name>:<suffix>`. In
v2 every model is its own map entry, and shared settings come from `extends` (which deep-merges
`params` — a `null` value deletes an inherited key). This removes the `base:suffix` string sprawl
and makes each callable model explicit.

The old variant mechanism was often used to bake a fixed `reasoning_effort` into a suffix (an
`oai-o3:m` for "medium"). In v2 you have two choices:

- **Keep them as distinct llms** — define `oai-o3-low`, `oai-o3-high`, etc. (each with its own
  `params`), typically via `extends` from a common base.
- **Consolidate into an ensemble effort matrix** — define the model once and let the ensemble's
  `effort_tiers` pick the depth per request. This is the idiomatic v2 approach and removes the
  need for `mom-high`-style alias models entirely. See
  [the effort matrix](./CONFIGURATION.md#the-effort-matrix).

## `models` (list) → `ensembles` (map)

`llms_to_query` becomes `members`; `concluding_llm` + `concluding_prompt` become a single
`synthesizer: { llm, prompt }`. Each member may now declare its own effort per tier rather than
inheriting a fixed variant. v1's `include_thinking_context` (exposing member reasoning) maps to
the v2 `show_work: off | inline | native` knob on the ensemble.

## Pricing: per-token → per-1M, and now optional

v1 required per-**token** rates (`prompt_cost_per_token`, `completion_cost_per_token`,
`reasoning_cost_per_token`). v2 uses per-**1M-token** rates (`input_per_1m`, `output_per_1m`,
`reasoning_per_1m`, plus `cache_read_per_1m` / `cache_write_per_1m`) — multiply your old numbers
by 1,000,000.

More importantly, **pricing is now optional**. MoM measures real cost automatically (LiteLLM's
cost map for direct providers, OpenRouter's returned usage cost for OpenRouter models). Keep a
`pricing:` block only for a model neither source can price; otherwise delete it.

## Before / after

**v1**

```yaml
llm_definitions:
  - base_name: "oai-g4.1"
    model: "openai/gpt-4.1"
    params: { temperature: 0.7 }
  - base_name: "oai-o3"
    model: "openai/o3-mini"
    variants:
      - suffix: "m"          # -> "oai-o3:m", a fixed medium-effort variant
  - base_name: "g25f"
    model: "gemini/gemini-2.5-flash-preview-04-17"

prompt_definitions:
  - name: "synth_default"
    content: |
      Synthesize the expert responses above into one superior answer.

models:
  - name: "mom"
    llms_to_query: ["oai-g4.1", "oai-o3:m", "g25f"]
    concluding_llm: "g25f"
    concluding_prompt: "synth_default"
    include_thinking_context: true

service:
  timeout_seconds: 30

langfuse:
  public_key_env: "LANGFUSE_PUBLIC_KEY"
```

**v2**

```yaml
version: 2

defaults:
  call: { timeout: 30s }

observability:
  langfuse: { enabled: true }   # key env-var names default to the LANGFUSE_* conventions

llms:
  oai-g4.1: { model: openai/gpt-4.1, params: { temperature: 0.7 } }
  oai-o3:   { model: openai/o3-mini }        # effort now comes from the tier, not a suffix
  g25f:     { model: gemini/gemini-2.5-flash-preview-04-17 }

prompts:
  synth_default: |
    Synthesize the expert responses above into one superior answer.

ensembles:
  mom:
    effort_tiers: [low, medium, high]
    default_tier: medium
    members:
      - { llm: oai-g4.1, effort: pass }        # relay the client's requested effort
      - { llm: oai-o3,   effort: [l, m, h] }   # follows the tier (replaces the ":m" variant)
      - { llm: g25f,     effort: off }         # no reasoning param
    synthesizer: { llm: g25f, prompt: synth_default }
    show_work: native                          # replaces include_thinking_context
```

Note that `oai-o3:m` disappeared: the model is defined once as `oai-o3` and the ensemble's
`effort_tiers` supply the depth. The old `:` in the variant name would be rejected by v2 anyway
(`:` and `+` are reserved in `llms`/`ensembles` names).

## Environment variables

Env vars are largely unchanged, and v2 still accepts the legacy names as aliases (`API_TOKEN`,
`MOM_CONFIG_PATH`, `REDIS_URL`, `LITELLM_VERBOSE`), preferring the `MOM_`-prefixed spelling when
both are set. Provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) and per-model proxy
URLs work exactly as before. See
[Environment & settings](./CONFIGURATION.md#environment--settings).

## Checklist

1. Add `version: 2` at the top.
2. Turn `llm_definitions` into an `llms:` map; replace `variants`/`suffix` with explicit entries
   or `extends`, and rename any name containing `:` or `+`.
3. Turn `prompt_definitions` into a `prompts:` map.
4. Turn `models` into an `ensembles:` map: `llms_to_query` → `members`, `concluding_llm` +
   `concluding_prompt` → `synthesizer`, and decide effort per member (tiers vs. fixed).
5. Convert any `pricing` to per-1M rates — or drop it and rely on automatic cost tracking.
6. Move `service.timeout_seconds` to `defaults.call.timeout` (as a duration string) and
   top-level `langfuse` under `observability`.
7. Change `api_mode: responses` to `api: responses` on any model that used it.
8. Validate: `mom config validate <file>` (and `mom config show <file>` to eyeball the resolved
   effort matrix). Clients need no changes.
```
