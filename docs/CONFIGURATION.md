# Configuration Reference (v2)

MoM is configured from two independent sources:

1. **YAML config files** — the *model catalog*: the individual models (`llms`), the
   synthesis prompts (`prompts`), and the ensembles clients actually call (`ensembles`),
   plus server/cache/observability settings. These files never contain secrets — only the
   **names** of the environment variables that hold them.
2. **The environment, and the secret files on the search path** — machine-local facts and
   secrets: the listen host/port, the API token, provider API keys, proxy URLs. Read from
   `MOM_`-prefixed variables (with a few legacy aliases), and from the `.env` / `auth.json`
   files described under [Where secrets come from](#where-secrets-come-from).

The YAML schema is Pydantic v2 with `extra = "forbid"` **everywhere**, so a misspelled key is
a hard error rather than a silently-ignored line.

```bash
mom config where                       # what was checked, what was found, and the merge order
mom config validate                    # loads, validates, resolves; non-zero on any problem
mom config show                        # prints the fully-resolved catalog (flattened efforts)
mom serve --host 0.0.0.0 --port 8000   # run the gateway
mom cache stats                        # response-cache entries / bytes / hits (under data_dir)
mom cache purge --yes                  # clear the response cache
```

None of those take a path, because mom finds its config itself. Every one of them also accepts
`--config <file>` (and honours `MOM_CONFIG`) when you want to say exactly which file to use.

## Where the config comes from

Two levels, deep-merged like git config — a **user** level holding the models and keys a machine
has once, and a **project** level holding only what a directory adds:

| Level | Looked for, in order — first **found** wins, they do not stack |
| --- | --- |
| user | `~/.mom/config.yaml`, then `$XDG_CONFIG_HOME/mom/config.yaml` (default `~/.config/mom/config.yaml`) |
| project | `./mom.yaml`, then `./.mom/config.yaml` |

Deliberately **not** `./config.yaml` — too generic a name to claim in an arbitrary directory.
And deliberately **no upward walk**: `./mom.yaml` means the working directory, and the user level
is what covers running from a subdirectory.

Each level also layers a **sibling override** for values specific to *this* machine or deployment
that have no business in a config you publish or share (`server.public_url` and your real domain,
say). The name is derived from the base file's own stem — `config.override.yaml` beside
`config.yaml`, and `mom.override.yaml` beside `mom.yaml`, so a project that names its config
`mom.yaml` does not grow a surprising `config.override.yaml`. A file pinned with `--config` gets
the same treatment (`prod.yaml` → `prod.override.yaml`). Overrides are gitignored by default,
along with `auth.json`.

The full merge order, lowest precedence first:

```
~/.mom/config.yaml  →  ~/.mom/config.override.yaml  →  ./mom.yaml  →  ./mom.override.yaml  →  $MOM_CONFIG_OVERLAY
```

**Validation runs once, after the merge**, so no single file has to be a complete config. That is
what makes the machine-wide catalog work — llms and prompts live in `~/.mom/config.yaml`, and a
project file is just the ensembles it adds:

```yaml
# ./mom.yaml — `llms: a` comes from the user level
version: 2
ensembles:
  panel:
    members: [{ llm: claude }, { llm: gpt }]
    synthesizer: { llm: claude }
```

Merge semantics are the same as `extends:` (see below), at every layer: nested maps merge
key-by-key, a scalar replaces the base value, and `null` **masks an inherited key** — which is
how a project drops an llm the user level defines.

### Pinning one file

`--config <file>` (or `MOM_CONFIG`) turns discovery **off entirely**. Only that file, its sibling
override, and `MOM_CONFIG_OVERLAY` apply — a server told exactly which config to serve must not
also pick up whatever happens to sit in `$HOME`:

```bash
export MOM_CONFIG=/etc/mom/config.yaml   # also reads /etc/mom/config.override.yaml
mom serve
```

`MOM_CONFIG` is read from the process environment and the working directory's `.env` only, never
from a discovered one — otherwise a file mom had not found yet could change where mom looks.

### Where secrets come from

Secrets are resolved separately from config, and the **first definition wins** — the process
environment always outranks a file:

| Precedence | Source |
| --- | --- |
| 1 | the process environment |
| 2 | `<project level dir>/.env`, then `<project level dir>/auth.json` |
| 3 | `./.env`, then `./auth.json` — the working directory, always |
| 4 | `~/.mom/.env` and `auth.json`, then `$XDG_CONFIG_HOME/mom/.env` and `auth.json` — skipped when pinned |
| 5 | `~/.local/share/opencode/auth.json` — only with `--auth-from-opencode` |

The working directory is always on the path, even when it is not the project level (a config in
`./.mom/`, or a pinned config elsewhere), because `./.env` is where people keep keys.

Unlike the config candidates, **both** user directories are searched for secrets, in the same
priority order (`~/.mom` first). Keys are additive and first-wins, so there is nothing to gain
from making them exclusive — and the common setup is a project `mom.yaml` for ensembles with
`~/.mom/.env` for keys and no user-level YAML at all.

**An empty value is treated as absent everywhere.** `KEY=` in a `.env`, or `"KEY": ""` in an
`auth.json`, defines nothing: it is not reported as a contribution, it does not shadow a real
value further down the path, and it does not stop a file from replacing an empty variable already
in the environment. Anything else would disagree with the code that reads these names — an empty
API key is indistinguishable from a missing one at the provider.

`auth.json` is a flat env-var-name → value object, the same vocabulary the config uses when it
names an `api_key_env` or `proxy_url_env`:

```json
{ "ANTHROPIC_API_KEY": "sk-ant-...", "MUSE_SPARK_PROXY_URL": "http://user:pw@proxy:8080" }
```

mom warns (never fails) when an `auth.json` is readable beyond its owner — `chmod 600` it. A
malformed one is a warning and a skip, not a startup failure: a missing key surfaces clearly on
the call that needed it, whereas a malformed *config* is fatal.

`MOM_*` settings in a `.env` reach mom's own settings but are deliberately **not** exported to the
process environment — `MOM_API_TOKEN` has no business being visible to every subprocess. Put
provider keys in `auth.json` or `.env`; put `MOM_*` in `.env`.

### Borrowing opencode's keys

`--auth-from-opencode` reads [opencode](https://github.com/sst/opencode)'s `auth.json`
(`$XDG_DATA_HOME/opencode/auth.json`, else `~/.local/share/opencode/auth.json`) and maps its
`type: "api"` entries onto the standard env var names, at the lowest precedence of all. The file
is ignored entirely unless the flag is set:

```bash
mom mcp --auth-from-opencode          # or MOM_AUTH_FROM_OPENCODE=1
mom config where --auth-from-opencode # preview what it would set, without setting it
```

`type: "oauth"` entries are skipped — they hold refresh tokens for a session opencode renews, not
API keys, and handing one to a provider fails with an opaque 401. Providers opencode authenticates
that mom has no model prefix for (bundled gateways, subscription plans) are skipped and reported.

### Debugging resolution

`mom config where` prints every path checked, which were found, the exact merge order, and the
secret files consulted. It reports env var **names** only — never values — and never applies the
secrets it describes, so asking where a key would come from cannot change where it comes from. It
also works when the merged config is broken or missing, which is when you need it most.

Each secret file is reported in three classes, because "what did this file do" has three
different answers:

| Line | Meaning |
| --- | --- |
| `would set: …` | names this file wins and publishes to the environment |
| `reaches settings: …` | `MOM_*` names, which configure mom itself without entering the environment |
| `already set elsewhere: …` | names a higher-precedence source already defined — this file lost |

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
  stream_heartbeat: null # emit an SSE keepalive comment after this much idle time; null = off
  cors:
    origins: []         # allowed origins; [] disables CORS
    allow_credentials: false
  dedupe:
    enabled: false       # in-flight request coalescing (see below); off by default
    orphan_grace: 90s    # cancel a run this long after its last subscriber leaves
    max_buffer: 8MB      # backstop against one run buffering unbounded history
```

- **`auth`** — `bearer` requires every request to carry the token (see
  [Environment & settings](#environment--settings)) as `Authorization: Bearer <token>` or an
  `x-api-key` header; comparison is timing-safe. `none` is an explicit opt-out for local dev —
  no token is checked. When `auth: bearer` but no token is configured, requests fail with a
  configuration error.
- **`public_url`** — the externally-reachable base URL of the gateway, used when it needs to
  emit an absolute link back to itself. Optional.
- **`stream_heartbeat`** — when set, every SSE response (`/v1/chat/completions`, `/v1/responses`,
  `/v1/messages`, and the progress feed) emits a `: keepalive` comment whenever the stream has
  gone this long without a real chunk, so a slow fan-out or synthesizer call doesn't trip a
  client's or intermediary's idle read-timeout. `null` (the default) disables it.
- **`cors`** — an empty `origins` list means CORS middleware is not installed at all. Listing
  `"*"` together with `allow_credentials: true` is rejected (a browser would refuse it anyway).
- **`dedupe`** — in-flight *request* coalescing: two concurrent, identical chat requests (same
  ensemble, messages, tools, sampling, effort, and resolved `<<SYSTEM>>` directives — see the
  `<<SYSTEM>>` section in [`README.md`](../README.md)) share one fan-out + synthesis instead of
  paying for it twice. The second (and any later) caller attaches to the first's live run and streams
  the exact same answer from token zero; its response carries an `X-MoM-Coalesced: 1` header and
  `X-Request-Id`/the progress link both point at the original (leading) request. This is distinct
  from [`cache.coalesce`](#cache), which dedupes identical *member calls* within/across ensembles
  at the LLM-call layer — `server.dedupe` operates one layer up, at the whole-request layer, and
  also catches a client's own retry-because-it-looked-stuck (the classic cause of doubled spend).
  `enabled` is the *default* policy rather than a hard switch: a `<<SYSTEM>> dedupe: on\|off`
  directive overrides it per request in both directions, so you can opt one duplicate-prone client
  into coalescing without committing the whole deployment, or force a genuinely fresh run when an
  identical turn is already in flight (re-rolling a panel rather than joining it).
  In-flight only: a run is dropped the instant it completes, so a deliberate regenerate afterward
  always starts fresh. `orphan_grace` is how long a run keeps going with zero attached
  subscribers (covers the gap between one client dropping and either a new one attaching or the
  original reconnecting) before it's cancelled outright. `max_buffer` is a backstop against one
  pathologically large run buffering unbounded history for a slow/absent subscriber; past it, the
  run keeps going but stops accepting new attachers (a request landing after that point just
  starts its own fresh run). Off by default — enable once validated against real traffic.
  Currently wired into `/v1/chat/completions` only.
- **`mcp`** — `enabled: true` mounts the [MCP tool surface](API.md#mcp-mcp-and-mom-mcp) at `/mcp`,
  in the same process and behind the same bearer auth as `/v1` (`auth: none` applies here too). Off
  by default: it is a second protocol on the same port, so turning it on should be deliberate. The
  flag gates the **HTTP** surface only — `mom mcp` serves the same tools over stdio regardless,
  since running that command is itself the opt-in. While disabled, `/mcp` answers 404 rather than
  403, so a switched-off surface doesn't announce itself. Everything but `consult` is read-only,
  and there is no purge or config-mutation tool on either transport.

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
settings applied to both members and the synthesizer. Retries are entirely mom's own — litellm's
built-in retry wrapper is never used (it gives zero backoff for most error classes and, when a
retry attempt also fails, silently discards that failure in favor of re-raising the *first*
attempt's error). mom instead retries only classified-transient failures (a rate limit or a
connection/server/timeout error — never a bad request, an auth failure, or a context-length
overflow, which fail identically on every attempt) with exponential backoff starting at
`retry_backoff` (honoring a provider's own `Retry-After` header when present), and always surfaces
the *last* attempt's error with an honest attempt count. A synthesizer's streamed answer is
retried only while establishing the connection — once the first token has streamed to the client,
a failure is never retried (it can't be un-sent).

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
  otel:
    enabled: false
    endpoint: http://localhost:4318       # OTLP collector endpoint
    protocol: http                        # http | grpc
    service_name: mom-llm
```

When Langfuse is `enabled: true`, each member and synthesizer call is recorded as a Langfuse
generation, grouped per request. Credentials are read from the environment variables **named** here
(the defaults match Langfuse's conventional names).

When `otel.enabled: true`, each call is emitted as an OpenTelemetry span following the GenAI
semantic conventions (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`) and exported over
OTLP to `endpoint`. Both backends may be enabled at once (calls fan out to each). OTel deps are
optional — install `mom-llm[otel]` (add `opentelemetry-exporter-otlp-proto-grpc` for `protocol:
grpc`). Tracing is fire-and-forget throughout: it never raises into the request path and silently
no-ops when a backend is disabled or its credentials/deps are missing.

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

Chains are resolved with cycle and missing-target detection. Reach for `extends` to reuse a model
for a one-off, differently-shaped purpose (a different `api`, a bespoke `params` shape) — for an
*effort family* of one base at several fixed levels, prefer `variants:` below; it's less
repetitive for that specific, very common case.

### `variants` — a compact effort family, nested

`variants:` is sugar over `extends` for the common case: one base model, several fixed-effort
siblings. Each key becomes its own llm named `<parent>-<key>`, inheriting the parent's `model` /
`api` / `api_key_env` / `proxy_url_env` unless overridden, with `params` deep-merged the same way
`extends` merges them:

```yaml
llms:
  sol:
    model: openai/gpt-5.6-sol
    variants:
      l: { params: { reasoning_effort: low } }
      m: { params: { reasoning_effort: medium } }
      h: { params: { reasoning_effort: high } }
      p: { api: responses, params: { reasoning: { effort: max, mode: pro } } }   # can override api too
```

This resolves exactly as if you had written `sol-l: {extends: sol, params: {...}}`,
`sol-m: {...}`, `sol-h: {...}`, `sol-p: {extends: sol, api: responses, params: {...}}` — one block
instead of four, and the model string appears once. Two things worth knowing:

- **Capability fields never propagate to a variant.** `search`, `pricing`, `capabilities`,
  `max_input_tokens`, `timeout`, and `cache_ttl` are deliberately excluded from what a variant can
  inherit — only generation-shaping fields do. A model that's both search-capable *and* has effort
  variants (see `k3` in a real config) keeps `search:` on its own bare identity; the variants stay
  plain. Set one of those fields explicitly on a variant in the rare case you actually want it.
- A variant name colliding with an existing top-level llm name is a config error at load time.

`members:` list items also accept a bare string as shorthand for `{llm: <name>}` — useful for an
ensemble that lists several llms with no per-member effort override:

```yaml
ensembles:
  everything:
    members: [sol, sol-l, sol-m, sol-h, sol-p]   # same as [{llm: sol}, {llm: sol-l}, ...]
    synthesizer: { llm: sol }
```

### `members: all` — a self-maintaining kitchen-sink panel

For a true "every llm, side by side" panel (a debug/eval ensemble), don't list members by hand at
all — `all` expands to every llm in the catalog (bases and expanded variants alike), so the panel
never falls out of sync as llms are added, renamed, or removed:

```yaml
ensembles:
  mom-debug:
    members: all
    synthesizer: { llm: sol, prompt: synth_default }
    show_work: inline
```

Opt specific llms out by name with `exclude` — e.g. a slow or costly special-purpose variant that
doesn't belong in a routine debug fan-out:

```yaml
ensembles:
  mom-debug:
    members: { all: true, exclude: [oai-dr, gem-dr] }   # skip the deep-research variants
    synthesizer: { llm: sol, prompt: synth_default }
    show_work: inline
```

An unknown name in `exclude` is a config error at load time (the same "never silently drift"
guarantee as everywhere else). `all` is only valid for `strategy: synthesize` — a `passthrough`
ensemble takes at most one member.

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
    tools:
      continuation: relay          # relay (default) | fanout
      member_tool_context: summary # summary (default) | none
      strategy: arbitrate          # arbitrate (default) | vote | first
      vote_threshold: 2            # min members agreeing, for strategy: vote
      stream_profile: compat       # compat (default) | strict
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
  (the per-tier matrix cell). Identities must be unique within the ensemble. Or `all` /
  `{all: true, exclude: [...]}` for a self-maintaining kitchen-sink panel — see
  [`members: all`](#members-all--a-self-maintaining-kitchen-sink-panel).
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
    cannot invoke tools themselves); `none` omits it. Ignored under `strategy: vote`/`first`,
    where members instead receive the real tool schemas.
  - `strategy: arbitrate` (default) | `vote` | `first` — how a tool call is chosen when the
    request carries tools:
    - `arbitrate`: the **synthesizer** decides; members stay advisory, and any calls they
      propose are surfaced to the synthesizer as context (the candidate envelope).
    - `vote`: members receive the real tools; if at least `vote_threshold` of them propose the
      **same** call (matched by name + normalized arguments), that call is returned directly and
      synthesis is skipped. Otherwise it falls back to `arbitrate`.
    - `first`: members receive the real tools; the first member (in config order) that proposes a
      tool call has its call(s) returned directly, skipping synthesis. Falls back to `arbitrate`
      when no member proposes one.
  - `vote_threshold` (default `2`, ≥ 1) — the minimum number of distinct members that must agree
    for `strategy: vote` to short-circuit.
  - `stream_profile: compat` (default) | `strict` — the streamed tool-call delta shape.
    `compat` re-emits `id`/`type`/`function.name` on **every** delta (safe for AI-SDK-style
    clients that read the header only once); `strict` sends them on the first delta only. A
    recognized AI-SDK `User-Agent` upgrades `strict` to `compat` automatically.
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
| Config file path | `MOM_CONFIG` | `MOM_CONFIG_PATH` | Pins one YAML catalog, disabling discovery — see [Pinning one file](#pinning-one-file). |
| Config overlay | `MOM_CONFIG_OVERLAY` | — | Optional file deep-merged last, over everything else. |
| opencode bridge | `MOM_AUTH_FROM_OPENCODE` | — | Also read API keys from opencode's `auth.json`, at lowest precedence. |
| Data directory | `MOM_DATA_DIR` | — | Overrides `storage.data_dir`; metrics DB + cache. |
| API token | `MOM_API_TOKEN` | `API_TOKEN` | The bearer token clients must present (when `auth: bearer`). |
| Listen host / port | `MOM_HOST` / `MOM_PORT` | — | Also settable via `mom serve --host/--port`. |
| Redis URL | `MOM_REDIS_URL` | `REDIS_URL` | Optional. |
| Log level | `MOM_LOG_LEVEL` | — | `INFO` (default) narrates each request — see [Reading the logs](#reading-the-logs). `WARNING` keeps only problems; `DEBUG` adds per-request wire detail. |
| Log format | `MOM_LOG_FORMAT` | — | `text` (default) or `json`. |
| LiteLLM debug | `MOM_LITELLM_DEBUG` | `LITELLM_VERBOSE` | Verbose upstream logging. Not yet wired in v2. |

### Reading the logs

At the default `INFO` level a request narrates itself as it runs, so `docker logs` shows what the
gateway is actually doing rather than only what went wrong. One request against a 2-member
ensemble prints its roster, a line per member as the call goes out, a line per member as it lands,
and a closing summary:

```
[info] fan-out started     request_id=req-9fd9… ensemble=mom members_total=2 members=['m1=openai/mock-a', 'm2=openai/mock-b']
[info] member dispatched   request_id=req-9fd9… llm=m1 model=openai/mock-a
[info] member dispatched   request_id=req-9fd9… llm=m2 model=openai/mock-b
[info] member completed    request_id=req-9fd9… llm=m1 status=ok cached=False duration_ms=302.0 tokens=18 cost_usd=0.0 attempts=1 completed=1 members_total=2
[info] member completed    request_id=req-9fd9… llm=m2 status=ok cached=False duration_ms=301.7 tokens=18 cost_usd=0.0 attempts=1 completed=2 members_total=2
[info] synthesis started   request_id=req-9fd9… llm=syn model=openai/mock-syn
[info] run completed       request_id=req-9fd9… status=stop members_ok=2 members_total=2 synthesis_ms=11.1 total_tokens=118 total_cost_usd=0.0 elapsed_seconds=0.31
```

Every line carries `request_id`, so concurrent requests stay separable under `grep`. The lines come
from the engine, so all three API surfaces and both streaming and non-streaming look the same. A
member that fails still gets its `member completed` line (with `status` and a classified
`error_kind`), and `run completed` reports `members_ok` against the **dispatched** roster, so a
member abandoned at the fan-out deadline shows up as a shortfall rather than vanishing from the
denominator. A member detached after a client disconnect finishes in the background and logs
`detached member completed` when it lands — after that request's `run completed`.

**No message content is ever logged.** Model names, identities, statuses, timings, token counts and
costs are; prompts, completions, reasoning, and tool arguments are not.

Provider **error** text is a separate matter. The lifecycle lines above carry only a classified
`error_kind`, and the client-visible error carries only a safe message. Internal-error lines report
an exception's type and source location (`pipeline.py:809 in run_ensemble`) rather than a
traceback, because a traceback's last line embeds the exception message — and an exception raised
while parsing a provider's response carries that provider's text in it.

The one exception is the operator-facing `member call failed` warning, which deliberately logs the
provider's own exception chain so a failure can actually be diagnosed. That text is **not**
scrubbed — it can contain API keys, bearer tokens, or internal URLs. Treat the log stream as
sensitive: it is fine on a host you control, but scrub or filter before shipping it to a
third-party aggregator.

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

All of these may be placed in a `.env` (or, for provider keys, an `auth.json`) anywhere on the
secrets search path — see [Where secrets come from](#where-secrets-come-from) for the full
precedence order. The process environment takes precedence over every file.
