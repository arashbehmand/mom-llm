# Configuration Reference (v2)

MoM is configured from two independent sources:

1. **A YAML config file** — the *model catalog*: the individual models (`llms`), the
   synthesis prompts (`prompts`), and the ensembles clients actually call (`ensembles`),
   plus server/cache/observability settings. This file never contains secrets — only the
   **names** of the environment variables that hold them.
2. **The process environment** — machine-local facts and secrets: the listen host/port, the
   API token, provider API keys, proxy URLs. Read from `MOM_`-prefixed variables (with a few
   legacy aliases) and from a `.env` file if present.

The YAML schema is Pydantic v2 with `extra = "forbid"` **everywhere**, so a misspelled key is
a hard error rather than a silently-ignored line. Point the server at your config with the
`MOM_CONFIG` environment variable, and validate it before starting:

```bash
export MOM_CONFIG=/etc/mom/config.yaml
mom config validate $MOM_CONFIG        # loads, validates, resolves; non-zero on any problem
mom config show     $MOM_CONFIG        # prints the fully-resolved catalog (flattened efforts)
mom serve --host 0.0.0.0 --port 8000   # run the gateway
```

A minimal but complete file — `version`, `llms`, and `ensembles` are the only required keys;
everything else has a sensible default:

```yaml
version: 2
llms:
  claude: { model: anthropic/claude-opus-4-8 }
  gpt:    { model: openai/gpt-5.6-sol }
ensembles:
  panel:
    members:
      - { llm: claude }
      - { llm: gpt }
    synthesizer: { llm: claude }
```

---

## Top level

```yaml
version: 2              # required — must be the integer 2
server:   { ... }       # HTTP surface: auth, CORS, public URL
defaults: { ... }       # per-call timeouts/retries, fan-out limits, provider caching
cache:    { ... }       # the gateway's own response cache
storage:  { ... }       # where the metrics DB / cache live on disk
observability: { ... }  # Langfuse tracing
budgets:  { ... }       # spend ceilings
llms:     { ... }       # required — one entry per callable upstream model
prompts:  { ... }       # named synthesis prompts referenced by ensembles
ensembles: { ... }      # required — the "models" clients call
```

`llms` and `ensembles` names may not contain `:` or `+` (reserved characters).

---

## `server`

```yaml
server:
  auth: bearer          # bearer (default) | none
  public_url: null      # external base URL, used to build progress/callback links
  cors:
    origins: []         # allowed origins; [] disables CORS
    allow_credentials: false
```

- **`auth`** — `bearer` requires every request to carry the token (see
  [Environment & settings](#environment--settings)) as `Authorization: Bearer <token>` or an
  `x-api-key` header; comparison is timing-safe. `none` is an explicit opt-out for local dev —
  no token is checked. When `auth: bearer` but no token is configured, requests fail with a
  configuration error.
- **`public_url`** — the externally-reachable base URL of the gateway, used when it needs to
  emit an absolute link back to itself. Optional.
- **`cors`** — an empty `origins` list means CORS middleware is not installed at all. Listing
  `"*"` together with `allow_credentials: true` is rejected (a browser would refuse it anyway).

---

## `defaults`

Baseline behavior for every call. Individual `llms` can override `timeout`; ensembles inherit
the rest.

```yaml
defaults:
  call:
    timeout: 20m        # per upstream call (duration string)
    retries: 3          # retry attempts on a failed call (>= 0)
    retry_backoff: 2s   # base backoff between retries
  fanout:
    max_concurrency: null   # cap on simultaneous member calls
    min_results: 1          # successful members required before synthesis (quorum)
    deadline: null          # optional overall fan-out wall-clock budget
  provider_cache:
    anthropic: { enabled: true, ttl: 5m }   # Anthropic prompt-cache breakpoints
    openai:    { prompt_cache_key: auto }    # OpenAI/xAI prefix-cache affinity
```

**`defaults.call`** — `timeout`, `retries`, and `retry_backoff` are the per-call transport
settings applied to both members and the synthesizer.

**`defaults.fanout`** —
- `max_concurrency` bounds how many members hit upstream at once. `null` does **not** mean
  unbounded: the gateway falls back to a built-in cap of **16** concurrent calls so a large
  panel can never open an unbounded number of connections. Set an integer to lower it.
- `min_results` is the quorum of successful members required before synthesis proceeds.
- `deadline` is an optional overall wall-clock budget for the fan-out; stragglers still running
  when it elapses are abandoned. `null` = no deadline (each call is still bounded by `timeout`).

**`defaults.provider_cache`** — controls *provider-side* prompt caching (distinct from MoM's own
[`cache`](#cache)):
- `anthropic.enabled` injects `cache_control` breakpoints on Anthropic-family synthesizers;
  `ttl` over 5 minutes selects Anthropic's `1h` cache tier, otherwise `5m`.
- `openai.prompt_cache_key: auto` sends a stable `prompt_cache_key` to OpenAI/Azure/xAI models
  for prefix-cache affinity; `off` disables it.

---

## `cache`

MoM's own response cache — identical member calls are served from disk instead of re-billed.

```yaml
cache:
  enabled: true
  ttl: 14d              # how long an entry is reused
  max_size: 1GB         # on-disk ceiling (base-1024 size string)
  coalesce: true        # collapse identical concurrent fan-out calls into one upstream call
```

`coalesce` deduplicates *in-flight* identical calls (common when two ensembles share a member),
so a burst of identical requests results in a single upstream call. A cache hit is billed at
**$0**.

---

## `storage`

```yaml
storage:
  data_dir: null        # null = platform default (or the MOM_DATA_DIR env var)
```

`data_dir` is where the metrics database and cache live. `null` uses the OS-appropriate app
data directory, unless `MOM_DATA_DIR` is set in the environment (which takes precedence).

---

## `observability`

```yaml
observability:
  langfuse:
    enabled: false
    public_key_env: LANGFUSE_PUBLIC_KEY   # env var NAMES, not the keys themselves
    secret_key_env: LANGFUSE_SECRET_KEY
    host_env: LANGFUSE_HOST
```

When `enabled: true`, each member and synthesizer call is recorded as a Langfuse generation,
grouped per request. Credentials are read from the environment variables **named** here (the
defaults match Langfuse's conventional names). Tracing is fire-and-forget: it never raises into
the request path, and it silently no-ops if the credentials are missing.

---

## `budgets`

```yaml
budgets:
  daily_usd: null                 # optional overall daily ceiling
  per_ensemble:                   # optional per-ensemble daily ceilings
    bmom: 50.0
    mom-code: 10.0
```

Spend ceilings, in USD. Both are optional. Costs are measured automatically (see below).

---

## `llms`

One entry per callable upstream model. The **key** is the short name you reference from
ensembles; the model string is a LiteLLM `provider/model` identifier.

```yaml
llms:
  gpt:     { model: openai/gpt-5.6-sol }
  claude:  { model: anthropic/claude-opus-4-8 }
  qwen:    { model: openrouter/qwen/qwen-3.6-max }
  gpt-o:                                  # a full example with every field
    model: openai/gpt-5.6-sol
    api: responses                        # chat (default) | responses
    api_key_env: OPENAI_API_KEY           # inferred from the provider when omitted
    proxy_url_env: US_PROXY_URL           # route this model's calls through a proxy
    params:                               # arbitrary provider params, passed through
      temperature: 0.3
      reasoning: { effort: high }
    search: { web_search_options: { search_context_size: high } }
    pricing: { input_per_1m: 1.25, output_per_1m: 10.0 }   # OPTIONAL override, see below
    capabilities: { context_length: 400000, vision: true }
    max_input_tokens: 380000
    timeout: 10m
    cache_ttl: 1h
```

Field by field:

- **`model`** *(required, directly or via `extends`)* — the LiteLLM model id, e.g.
  `anthropic/claude-opus-4-8`, `openai/gpt-5.6-sol`, `gemini/gemini-3.1-pro`,
  `openrouter/qwen/qwen-3.6-max`.
- **`api`** — `chat` (default, `/chat/completions` upstream) or `responses` (the provider's
  Responses API). Choose `responses` for models exposed only through it.
- **`api_key_env`** — the **name** of the env var holding the API key. When omitted it is
  **inferred from the provider prefix** of `model`:

  | provider prefix | inferred env var(s) |
  | --- | --- |
  | `openai/` | `OPENAI_API_KEY` |
  | `anthropic/` | `ANTHROPIC_API_KEY` |
  | `gemini/` | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` |
  | `vertex_ai/` | `GOOGLE_API_KEY` |
  | `xai/` | `XAI_API_KEY` |
  | `mistral/` | `MISTRAL_API_KEY` |
  | `openrouter/` | `OPENROUTER_API_KEY` |
  | `deepseek/` | `DEEPSEEK_API_KEY` |
  | `groq/` | `GROQ_API_KEY` |
  | `cohere/` | `COHERE_API_KEY` |

  Providers with more than one candidate are tried in order (first one set wins). Set
  `api_key_env` explicitly to point an unusual model at a specific key.
- **`proxy_url_env`** — the name of an env var holding an `http(s)` proxy URL. When set, this
  model's calls go **only** through that proxy — if the env var is unset or malformed the call
  fails rather than silently connecting directly (a hard guarantee, useful for region-locked
  models). See `MUSE_SPARK_PROXY_URL` in the example config.
- **`params`** — an open dict merged into the upstream request (temperature, `top_p`,
  provider-native `reasoning` objects, etc.). Reserved keys the gateway sets itself —
  `model`, `messages`, `stream`, `api_key`, `num_retries`, `timeout` — are **rejected**.
- **`search`** — provider params merged in **only when the client requests web search**. The
  mere presence of this block (even an empty `{}` for an inherently-online model) marks the LLM
  as search-capable; without it, the model answers without search even on a web-search request.
- **`pricing`** — an **optional** per-1M-token price override (see the next section). Omit it
  unless MoM can't price the model automatically.
- **`capabilities`** — override the model's advertised capability card:
  `context_length`, `max_output_tokens`, `vision`, `tools`, `reasoning`. For example
  `vision: false` excludes this model from image requests.
- **`max_input_tokens`** — a guard on the input size routed to this model.
- **`timeout`** — per-model call timeout, overriding `defaults.call.timeout`.
- **`cache_ttl`** — per-model override of the response-cache TTL.

### `extends` — inheritance

`extends` is the single inheritance primitive. A child inherits every field it does not set
itself; `params` are **deep-merged** (a `null` value **deletes** an inherited key):

```yaml
llms:
  base-claude:
    model: anthropic/claude-opus-4-8
    params: { temperature: 0.2, top_p: 0.9 }
    pricing: { input_per_1m: 15.0, output_per_1m: 75.0 }
  claude-creative:
    extends: base-claude
    params: { temperature: 0.9, top_p: null }   # override temperature, drop top_p
    # model + pricing inherited unchanged
```

Chains are resolved with cycle and missing-target detection. Use `extends` to define a family
of variants (different sampling, different reasoning) from one base without repetition.

---

## Cost is tracked automatically

**You do not need to configure pricing.** MoM measures the real dollar cost of every call
automatically:

- For **direct providers** (OpenAI, Anthropic, Gemini, …) it uses LiteLLM's bundled
  cost-per-token map, keyed on the model id and the returned token usage (including cached and
  reasoning tokens).
- For **OpenRouter** models it asks OpenRouter to return the actual usage-based cost of the
  call and uses that number verbatim — so even models LiteLLM's map doesn't know are priced
  correctly.
- A **cache hit costs $0**.

The optional `pricing:` block on an LLM is **only an override**, for the rare model that
*neither* source can price. When present it takes precedence over the automatic figure; when
absent the automatic cost is used. Rates are per **1M tokens**:

```yaml
llms:
  house-model:
    model: openrouter/acme/house-model-1
    pricing:
      input_per_1m: 0.50
      output_per_1m: 1.50
      reasoning_per_1m: 1.50      # optional, defaults to unpriced
      cache_read_per_1m: 0.05     # optional
      cache_write_per_1m: 0.60    # optional
```

Cached prompt tokens are billed at `cache_read_per_1m` and the rest of the prompt at
`input_per_1m`; Anthropic-style cache-creation tokens are billed additively at
`cache_write_per_1m`.

---

## `prompts`

Named synthesis instructions, referenced by ensembles' synthesizers. Keeping them here avoids
inlining long strings in every ensemble.

```yaml
prompts:
  synth_default: |
    You are the concluding model of a multi-model ensemble. Synthesize the candidate responses
    into a single, superior answer. Resolve disagreements, drop hallucinations, and keep the
    strongest reasoning from each.
```

---

## `ensembles`

An ensemble is what a client calls (the client sends its **name** as the `model`). Members fan
out in parallel; the synthesizer merges their answers into the client-visible response.

```yaml
ensembles:
  bmom:
    description: Balanced panel across four providers.
    strategy: synthesize            # synthesize (default) | passthrough
    effort_tiers: [low, medium, high]
    default_tier: medium
    members:
      - { llm: gpt,    effort: [l, m, h] }
      - { llm: claude, effort: [m, h, h] }
      - { llm: qwen,   effort: [m, h, h] }
      - { llm: gemini, effort: pass }
      - { llm: flash,  effort: [skip, m, h] }
    synthesizer: { llm: claude, effort: [m, h, h], prompt: synth_default }
    show_work: native
    tools: { continuation: relay, member_tool_context: summary }
    advertise: { context_length: 200000 }
    on_input_overflow: skip
```

- **`description`** — free text, surfaced in the `/v1/models` capability card.
- **`strategy`** —
  - `synthesize` (default): fan out to all members, then the synthesizer merges them. Requires
    at least one member.
  - `passthrough`: **no fan-out** — the synthesizer model answers the client's messages
    directly (the synthesis prompt is not applied). Ideal for a single-model, low-latency
    endpoint (e.g. a coding agent). Takes at most one member.
- **`effort_tiers`** / **`default_tier`** — the discrete reasoning tiers this ensemble exposes.
  A client's `reasoning_effort` is snapped to the **nearest** defined tier (ties round *up*);
  when the client sends none, `default_tier` is used. `default_tier` is required whenever
  `effort_tiers` is set and must be one of them. Omit both for an ensemble with no tiers (each
  member just runs its own configured params). See [The effort matrix](#the-effort-matrix).
- **`members`** — the panel. Each entry names an `llm`, may set `as` to give a distinct
  identity (so the same `llm` can appear twice with different effort), and may set `effort`
  (the per-tier matrix cell). Identities must be unique within the ensemble.
- **`synthesizer`** *(required)* — the concluding model: `llm`, an optional `prompt` (a key
  from `prompts`), and an optional `effort`. The synthesizer owns the client-visible output,
  its tool calls, and structured-output/`response_format`. Client sampling controls
  (`temperature`, `max_tokens`, `stop`, `seed`, …) are applied to the synthesizer.
- **`show_work`** — how member reasoning is exposed to the client:
  - `off` (default): hidden.
  - `inline`: member perspectives are rendered as a `<think>…</think>` block prepended to the
    answer content.
  - `native`: reasoning is emitted through the provider-native reasoning channel
    (`reasoning_content` / thinking blocks), separate from the answer.

  > Note: YAML parses a bare `off`/`on` as a boolean; the schema maps `off` → `off` and a bare
  > `on`/`true` → `inline`. Quote the value if you want to be explicit.
- **`tools`** —
  - `continuation: relay` (default) | `fanout`. On a **tool-continuation** turn (the client is
    feeding back a tool result), `relay` skips the fan-out and lets the synthesizer drive the
    tool loop directly — fast and coherent for agent loops. `fanout` re-runs the full panel on
    every turn.
  - `member_tool_context: summary` (default) | `none`. On a fresh turn that carries tools,
    `summary` gives advisory members a schema-free description of the available tools (they
    cannot invoke tools themselves); `none` omits it.
- **`advertise`** — override fields of the computed capability card by name:
  `vision`, `tools`, `reasoning` (booleans) and `context_length`, `max_output_tokens`
  (integers). By default the card is computed from the panel (vision = any member; tools =
  synthesizer; reasoning = has tiers or shows work; web_search = any member/synthesizer has a
  `search` block; `context_length` = the smallest member's window; `max_output_tokens` = the
  synthesizer's). Use `advertise` to correct any of these for a client that reads them.
- **`on_input_overflow`** — `skip` (default) drops a member whose `max_input_tokens` would be
  exceeded; `reject` fails the request instead.

---

## The effort matrix

The effort matrix is how a single ensemble serves several reasoning depths without a separate
`mom-high`-style alias model per depth. Each member (and the synthesizer) declares **its own**
effort per tier, so "give me a high-effort answer" can mean *high* for the flagship models and
*medium* for a cheaper one.

**Each cell** is one of:

| value | meaning |
| --- | --- |
| a **level** | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` — with short aliases `min`, `l`, `med`/`m`, `h`, `xh`. Sends that reasoning effort. |
| `pass` | relay **the client's** requested effort to this member unchanged. |
| `off` | send **no** reasoning param (model default / a non-reasoning model). |
| `skip` | **exclude** this member entirely at that tier. |

**Three ways to write a member's `effort`**, given `effort_tiers: [low, medium, high]`:

```yaml
members:
  - { llm: gpt,    effort: [l, m, h] }          # positional list, aligned to the tiers in order
  - { llm: claude, effort: { low: m, high: h } } # explicit {tier: value} map (unlisted tiers -> off)
  - { llm: gemini, effort: pass }                # a scalar -> the same value at every tier
  - { llm: flash,  effort: [skip, m, h] }        # excluded at the 'low' tier, medium/high otherwise
```

- A **positional list** must have exactly one entry per tier, in the same order as
  `effort_tiers`.
- A **map** may name any subset of tiers; tiers you don't mention default to `off`.
- A **scalar** applies to every tier.
- **Omitting `effort`** entirely means `off` at every tier.

At request time, the client's `reasoning_effort` selects a tier (nearest, ties up), then each
member's cell for that tier is resolved and — because providers accept different vocabularies —
**clamped** to what the target provider allows (e.g. `xhigh`/`max` collapse to the provider's
`high`; `minimal` becomes `low` for non-OpenAI providers). An effort a model can't accept is
dropped rather than causing a 400. Non-tiered ensembles ignore effort cells entirely.

`mom config show <file>` prints the flattened matrix per ensemble, which is the quickest way to
sanity-check it.

---

## Value formats (durations, sizes, effort)

- **Durations** — `500ms`, `2s`, `20m`, `1h`, `14d` (units `ms`, `s`, `m`, `h`, `d`). A bare
  number is treated as seconds.
- **Byte sizes** — `512KB`, `64MB`, `1GB` (base-1024; units `B`, `KB`, `MB`, `GB`, `TB`).
- **Effort levels** — see the table above.
- **Env var names** — must match `^[A-Z][A-Z0-9_]*$` (upper snake case).

---

## Environment & settings

Secrets and machine-local facts come from the environment, never the YAML. Settings are read
from `MOM_`-prefixed variables; several **legacy v1 names** are still accepted as aliases (the
`MOM_` name wins when both are set).

| Purpose | Primary variable | Legacy alias | Notes |
| --- | --- | --- | --- |
| Config file path | `MOM_CONFIG` | `MOM_CONFIG_PATH` | Path to the YAML catalog. |
| Data directory | `MOM_DATA_DIR` | — | Overrides `storage.data_dir`; metrics DB + cache. |
| API token | `MOM_API_TOKEN` | `API_TOKEN` | The bearer token clients must present (when `auth: bearer`). |
| Listen host / port | `MOM_HOST` / `MOM_PORT` | — | Also settable via `mom serve --host/--port`. |
| Redis URL | `MOM_REDIS_URL` | `REDIS_URL` | Optional. |
| Log level | `MOM_LOG_LEVEL` | — | e.g. `INFO`, `DEBUG`. |
| Log format | `MOM_LOG_FORMAT` | — | `text` (default) or `json`. |
| LiteLLM debug | `MOM_LITELLM_DEBUG` | `LITELLM_VERBOSE` | Verbose upstream logging. |

**Provider API keys** are read directly from the environment, by the names inferred (or set
via `api_key_env`) in `llms`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...          # or GOOGLE_API_KEY
OPENROUTER_API_KEY=...
XAI_API_KEY=...             # MISTRAL_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY, COHERE_API_KEY, ...
```

**Per-model proxy URLs** are read from whatever env var a model's `proxy_url_env` names, e.g.:

```bash
MUSE_SPARK_PROXY_URL="http://user:password@us-proxy.example:8080"
```

**Langfuse** credentials are read from the env vars named in `observability.langfuse`
(`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` by default).

All of these may be placed in a `.env` file in the working directory; the process environment
takes precedence over `.env`.
