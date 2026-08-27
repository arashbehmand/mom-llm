# MoM v2 API Reference

MoM exposes three drop-in-compatible chat protocols — **OpenAI Chat Completions**, **OpenAI
Responses**, and **Anthropic Messages** — plus model-discovery, metrics, and health endpoints, and
an optional **MCP** tool surface. A "model" in every request is the name of an **ensemble**: MoM
fans the prompt out to that ensemble's panel of members and returns one synthesized answer,
rendered in whichever protocol you called.

The chat protocols and everything alongside them are mounted under `/v1`; MCP is its own protocol
at `/mcp`. Point any OpenAI or Anthropic SDK at the gateway's base URL and set the model to an
ensemble name.

- [Authentication](#authentication)
- [Errors](#errors)
- [`POST /v1/chat/completions`](#post-v1chatcompletions)
- [`POST /v1/responses`](#post-v1responses)
- [`POST /v1/messages` and `/v1/messages/count_tokens`](#post-v1messages)
- [Model discovery](#model-discovery)
- [`GET /v1/metrics/usage`](#get-v1metricsusage)
- [`GET /v1/progress/{id}`](#get-v1progressid)
- [`GET /health`](#get-health)
- [MCP (`/mcp` and `mom mcp`)](#mcp-mcp-and-mom-mcp)
- [Effort tiers](#effort-tiers)
- [Usage and cost](#usage-and-cost)
- [Deviations](#deviations)

---

## Authentication

Auth is a per-request dependency on every `/v1` route, and the same check guards `/mcp`. Present
the token either way OpenAI and Anthropic SDKs send it:

```
Authorization: Bearer <token>      # OpenAI style
x-api-key: <token>                 # Anthropic style
```

The bearer header is checked first; `x-api-key` is the fallback. The comparison is timing-safe (a
SHA-256 digest compared with `secrets.compare_digest`). The token is the `MOM_API_TOKEN` (or legacy
`API_TOKEN`) env var.

- Set `server.auth: none` in the config to disable auth entirely (the dependency becomes a no-op).
- A missing or wrong token returns **401** (`authentication_error`, code `invalid_api_key`).
- If auth is enabled but no token is configured server-side, requests return **500**
  (`config_error`).

`/health` requires no auth.

## Errors

Every error is rendered as OpenAI-shaped JSON; provider/internal exception text is never leaked.

```json
{ "error": { "message": "unknown model 'gpt5-panel'", "type": "invalid_request_error", "code": "model_not_found" } }
```

| HTTP | `type` | `code` | When |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | `invalid_request` | bad params, unsupported tool type, stateful Responses features |
| 401 | `authentication_error` | `invalid_api_key` | missing/invalid token |
| 404 | `invalid_request_error` | `model_not_found` | unknown ensemble name |
| 502 | `upstream_error` | `upstream_error` | provider call failed |
| 504 | `upstream_error` | `timeout` | synthesizer stream stalled past its deadline |
| 500 | `api_error` | `internal_error` / `config_error` | unexpected / misconfiguration |

Plan-time errors (unknown model, invalid effort) are raised **before** any fan-out, so they arrive
as clean HTTP errors. A failure that occurs mid-stream is delivered as a terminal error frame in the
active protocol's grammar (an SSE `error`/`response.failed`/`message` frame), and non-streaming
requests re-raise it with the same HTTP status.

---

## `POST /v1/chat/completions`

The fullest surface. Unknown fields are ignored (not rejected).

**Honored request params**

| Param | Effect |
| --- | --- |
| `model` | ensemble name (required) |
| `messages` | conversation; `developer` role is folded to `system`; string or multi-part content; `image_url` parts; assistant `tool_calls`; `tool` results with `tool_call_id`; `name` |
| `stream` | SSE when true, else a single JSON body |
| `stream_options.include_usage` | when true, emit a trailing usage-only chunk before `[DONE]` |
| `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `stop`, `seed` | **applied to the synthesizer** (see note); `max_completion_tokens` wins over `max_tokens`; `stop` may be a string or list |
| `tools`, `tool_choice`, `parallel_tool_calls` | function tools given to the **synthesizer** (it owns the client-visible tool calls); `tool_choice` accepts `auto`/`none`/`required` or `{type:"function",function:{name}}` |
| `response_format` | structured-output spec passed to the synthesizer |
| `reasoning_effort` | resolves to an effort tier (see [Effort tiers](#effort-tiers)) |
| `web_search`, `web_search_options` | either one flags web search; MoM merges each search-capable member's provider search params |
| `user`, `metadata` | accepted and carried on the internal request |

**Sampling note.** The client's generation controls shape **only the synthesizer** — it produces
the visible answer. Advisory members keep their own tuned params, so a client `max_tokens` never
truncates a member's internal reasoning.

**Response (non-streaming)** — a `chat.completion` object. `choices[0].message` carries `content`,
optional `reasoning_content`, and optional `tool_calls`. `usage` includes
`prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`. Usage is the
**aggregate** across the whole ensemble (see [Deviations](#deviations)).

**Response (streaming)** — `chat.completion.chunk` SSE frames: the assistant `role` is sent exactly
once on the first delta; `reasoning_content` and `content` deltas follow; tool calls stream as
`tool_calls` deltas; a terminal `finish_reason` frame is always emitted (synthesized as `stop` if
the provider omits one); then the optional usage frame (`choices: []`); then `data: [DONE]`. With
`show_work: inline`, member perspectives are streamed first as a `<think>…</think>` block in the
content channel.

---

## `POST /v1/responses`

The **stateless** Open Responses subset. MoM holds no server-side state, so the stateful features of
the Responses API are rejected rather than silently mishandled.

**Honored request params**

| Param | Effect |
| --- | --- |
| `model` | ensemble name |
| `input` | a string, or a list of items: `message`, `function_call`, `function_call_output`; `input_image` parts are supported; `reasoning` items (our own replayed output) are dropped |
| `instructions` | prepended as a `system` message |
| `stream` | SSE lifecycle events when true |
| `tools`, `tool_choice`, `parallel_tool_calls` | `function` tools go to the synthesizer; `web_search`/`web_search_preview` set the web-search flag; an `mcp` tool returns **400** (run MCP client-side and pass plain function tools) |
| `reasoning.effort` | effort tier; `auto` is treated as unset |
| `max_output_tokens`, `temperature`, `top_p` | synthesizer sampling |
| `text.format` | used as the synthesizer's `response_format` |
| `metadata` | carried through |
| `store` | accepted but ignored — output always reports `store: false` |
| `previous_response_id` | **400** — resend the full input each turn |
| `background` | **400** — background responses are not supported |

**Response (non-streaming)** — a `response` object with `status: "completed"` and an `output` array:
a `message` item (with `output_text` parts) for the answer, plus one `function_call` item per tool
call. `usage` is `{input_tokens, output_tokens, total_tokens}`.

**Response (streaming)** — the lifecycle event sequence, each line carrying `event:` and `data:` with
a centralized `sequence_number`:

```
response.created → response.in_progress
  → response.output_item.added (message, in_progress)
  → response.content_part.added
  → response.output_text.delta …            (repeated)
  → response.output_text.done → response.content_part.done → response.output_item.done
  → response.output_item.added (function_call) → response.function_call_arguments.delta …
  → response.function_call_arguments.done → response.output_item.done
  → response.completed
```

A mid-stream failure closes any open items and emits `response.failed` with an `error` block.

---

## `POST /v1/messages`

Anthropic Messages compatibility. Used by Claude Code and the Anthropic SDK.

**Honored request params**

| Param | Effect |
| --- | --- |
| `model` | ensemble name |
| `messages` | user/assistant turns; `content` is a string or a block list |
| `max_tokens` | required by Anthropic; applied as the synthesizer's `max_tokens` |
| `system` | a string **or an array of text blocks** (joined with blank lines) → a `system` message |
| `stream` | SSE when true |
| `temperature`, `top_p`, `stop_sequences` | synthesizer sampling |
| `tools`, `tool_choice` | client-defined tools (with `input_schema`) go to the synthesizer; server tools (`web_search…`) are skipped as tools but flag web search; `tool_choice` maps `any→required`, `none→none`, `tool→` a specific tool |
| `thinking` | mapped to an effort tier by `budget_tokens` (see [Effort tiers](#effort-tiers)); `type:"disabled"` → `none` |
| `metadata` | carried through |

**Content blocks.** Incoming blocks are translated into the canonical IR: `text` and `image`
(base64 or URL source) into user content; `tool_use` into an assistant tool call; `tool_result`
(string or a list of text blocks) into a `tool` message keyed by `tool_use_id`. `thinking` /
`redacted_thinking` blocks (our own replayed output) are dropped.

**Response.** A `message` object whose `content` is a list of `text` and `tool_use` blocks
(`tool_use.input` is the parsed JSON of the streamed arguments). `stop_reason` maps from the internal
finish reason:

| internal | Anthropic `stop_reason` |
| --- | --- |
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `refusal` |
| `error` | `end_turn` |

**Streaming** folds into Anthropic's block grammar: `message_start` → `content_block_start/delta/stop`
for each block (a `thinking` block streams first when the synthesizer emits reasoning, then a `text`
block, then `tool_use` blocks with `input_json_delta` fragments) → `message_delta` (final
`stop_reason` + `output_tokens`) → `message_stop`.

**Usage.** `input_tokens` is a transcript-perspective estimate (see below); `output_tokens` is the
aggregate completion count; `cache_creation_input_tokens` / `cache_read_input_tokens` are reported as
`0`.

### `POST /v1/messages/count_tokens`

Returns `{"input_tokens": N}`, a deliberately rough estimate (~4 characters per token) over the
visible transcript (`system` + `messages` + `tools`). Claude Code uses this only for
context-window math. The **same** estimate seeds `message_start.usage.input_tokens` on
`/v1/messages`. This endpoint does not run the ensemble.

---

## Model discovery

Ensembles are advertised as models with **capability cards** computed from the panel (in
`config/capabilities.py`). Aggregation follows what actually happens at runtime:

| Capability | Rule |
| --- | --- |
| `supports_tools` | the **synthesizer** (it owns the final tool calls / structured output) |
| `supports_vision` | **any** member can see (image requests filter to vision-capable members) |
| `supports_reasoning` | the ensemble exposes effort tiers **or** `show_work != off` |
| `supports_web_search` | any member or the synthesizer declares a search block |
| `context_length` | **min** across the panel (the weakest member caps the safe window) |
| `max_output_tokens` | the synthesizer's declared limit |
| `reasoning_effort_levels` | the ensemble's tier labels |

An ensemble's `advertise:` block overrides any computed field with an explicit value.

### `GET /v1/models` and `GET /v1/models/{id}`

Lists the ensembles. Each entry carries OpenAI's fields plus OpenRouter-convention fields
(`architecture`, `supported_parameters`, `context_length`, `top_provider`) and a `mom` vendor block
(`members`, `synthesizer`, the `supports_*` flags, `client_managed_mcp: true`, `remote_mcp: false`).
`supported_parameters` grows with capability — tools/response_format when tool-capable,
`reasoning_effort`/`reasoning` when reasoning-capable, `web_search*` when search-capable.

Those two MCP flags are about MoM as an MCP *client* and are unrelated to
[MoM's own MCP surface](#mcp-mcp-and-mom-mcp): `client_managed_mcp` means you may pass Responses
`type: mcp` tool blocks through to a synthesizer that understands them, and `remote_mcp: false`
means MoM does not itself connect out to remote MCP servers on your behalf.

`GET /v1/models/{id}` returns the OpenAI single-model object, or **404** for an unknown ensemble.

#### Discovery dialects

Model discovery is the one place where otherwise-compatible clients disagree on the *envelope*, so
`GET /v1/models` answers in the dialect the request asks for. The models themselves are identical in
every dialect; only the wrapper changes.

| Request carries | Envelope |
| --- | --- |
| *(nothing)* | OpenAI — `{object:"list", data:[…]}` |
| `anthropic-version` or `x-api-key` header | Anthropic — `{data:[{type:"model", id, display_name, created_at}], has_more, first_id, last_id}` |
| `?client_version=` query param | Codex — `{"models": []}` |

Codex CLI refreshes its model picker with `GET /v1/models?client_version=<v>`. Given the OpenAI
envelope it logs `failed to refresh available models: … missing field 'models'` and falls back to
its bundled metadata; the parameter's *presence* (whatever its value) selects the Codex envelope.

```bash
curl "http://localhost:8000/v1/models?client_version=0.147.0" -H "Authorization: Bearer $TOKEN"
# {"models": []}
```

That catalog is empty on purpose. Codex requires every entry to carry `base_instructions` and uses
what it receives *verbatim* as that model's system prompt — measured against codex-cli 0.147.0, an
entry with `"base_instructions": ""` dropped the entire 20.7 KB agent prompt from Codex's request.
MoM has no business owning an agent's prompt, so it emits no entries: the refresh error goes away
and Codex keeps the metadata it already uses. Auth applies on this path like any other.

### `GET /v1/model/info`

A LiteLLM-shaped probe: `{data: [{model_name, litellm_params:{model:"mom/<id>"}, model_info:{…}}]}`
with `max_input_tokens`, `max_output_tokens`, `supports_function_calling`, `supports_vision`,
`supports_reasoning`, `supports_web_search`.

---

## `GET /v1/metrics/usage`

Single-pass aggregate over the metrics store. Optional query params: `start` and `end` (epoch
seconds) and `model` (filter to one ensemble).

```
GET /v1/metrics/usage?start=1753000000&model=council
```

Returns counts and sums: `calls`, token totals (`prompt_tokens`, `completion_tokens`,
`reasoning_tokens`, `cached_prompt_tokens`, `cache_write_tokens`), `cost_usd`, `errors`,
`cache_hits`, and `relay_calls` (tool-continuation turns that skipped fan-out). Returns
`{"calls": 0}` when metrics are unavailable.

Add `by=member`, `by=turn_type`, or `by=day` to group the aggregate (SQL `GROUP BY`). Grouped
responses are shaped `{"by": "<dimension>", "groups": [ {<dimension>: <key>, ...aggregate}, ... ]}`
— for example `by=member` returns one row per member/synthesizer, `by=day` one row per UTC day. The
`start` / `end` / `model` window filters apply to grouped queries too.

## `GET /v1/progress/{id}`

Server-sent stream of a request's coarse lifecycle milestones — `fanout_started`, one
`member_completed` per member, `synthesis_started`, and a terminal `completed` (or `failed`). Each
SSE frame is `event: <kind>` with a JSON `data:` payload; a `: ping` comment heartbeats an idle
connection. To watch a request, generate an id, open this stream, then issue the chat/messages/
responses call with the **`X-Request-Id`** header set to the same id (the gateway also echoes the
resolved id back in the `X-Request-Id` response header). Progress is buffered briefly per request,
so opening the stream a moment after the call still replays the events already emitted.

Backed by an in-memory bus by default; set `MOM_REDIS_URL` to fan progress across worker processes
via Redis (`pip install 'mom-llm[redis]'`).

### In-flight request coalescing (`server.dedupe`)

When enabled (off by default — see [`CONFIGURATION.md`](CONFIGURATION.md#server)), a chat
completions request identical to one already in flight attaches to that run instead of starting a
second one: both callers get the same answer, streamed from token zero, for the cost of one
fan-out and one synthesis. A coalesced response's `X-Request-Id` (and therefore the progress link
built from it) is the **original** request's id, not the second caller's own — that's the only
channel with anything published on it — and it carries an additional `X-MoM-Coalesced: 1` response
header so a client (or an operator reading logs) can tell the two apart. Coalescing is in-flight
only: a run is dropped from consideration the instant it completes, so an intentional identical
regenerate sent afterward always starts a fresh one. Currently wired into
`POST /v1/chat/completions` only.

## `GET /health`

Unauthenticated liveness probe. Returns `{"status": "ok", "version": "<v>"}`.

---

## MCP (`/mcp` and `mom mcp`)

The chat protocols make MoM a *model* you point a client at. MCP makes it a *tool* an agent can
call without leaving the model it is already running on: ask a panel for a second opinion mid-task,
assemble a panel from the catalog on the spot, and read spend, runs and cache state without a shell
on the gateway host.

**Transports.** Streamable HTTP at `/mcp` (same process, port, and bearer auth as `/v1`; the
`auth: none` dev opt-out applies here too), enabled with `server.mcp: { enabled: true }` — **off by
default**. And `mom mcp`, the same tools over stdio for a local client, backed by the same config
and data dir with no gateway running; consults it runs land in the same metrics DB and warm the
same cache. `mom mcp` ignores `server.mcp.enabled` (running the command is the opt-in) and is
unauthenticated: there are no headers on a pipe, and a process you started is inside the trust
boundary, exactly like the rest of the CLI.

```jsonc
// Claude Code / Cursor, over HTTP
{"mcpServers": {"mom": {"type": "http", "url": "https://mom.example.com/mcp",
                        "headers": {"Authorization": "Bearer $MOM_API_TOKEN"}}}}

// ...or locally over stdio
{"mcpServers": {"mom": {"command": "mom", "args": ["mcp"],
                        "env": {"MOM_CONFIG": "/etc/mom/config.yaml"}}}}
```

When the surface is disabled, `/mcp` answers **404** — a switched-off surface should look absent
rather than announce itself to anyone probing the path.

### Tools

Everything except `consult` is read-only. There is deliberately no purge and no config mutation: a
leaked token can already chat, and must not be able to destroy state.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_llms` | — | every catalog llm (bases and variants) with model string, capabilities, pricing and its `pricing_source` |
| `list_ensembles` | — | the resolved ensembles: members, effort tiers, synthesizer, advertised capabilities |
| `consult` | `prompt`, then either `ensemble` or `panel` + `synthesizer`; optional `effort`, `system`, `tools`, `include_member_answers` | one synthesized answer with a per-member cost breakdown (below) |
| `runs` | optional `request_id`, `limit` (clamped to 1–200) | in-flight, just-finished and recent runs, per-member status and cost |
| `usage` | `days` (0 or less = all time), optional `ensemble` | the same aggregation `mom metrics usage` prints, grouped by ensemble and by member |
| `cache_stats` | — | response-cache entry count, size, hits |

`consult` reports progress as each member is asked, as each one answers (with the running cost),
and again when synthesis begins — so a client can watch a slow panel work instead of guessing
whether the call has hung.

`effort` only applies to an ensemble that declares effort tiers. Passing it to one that doesn't —
including any inline panel, which is always tierless — is rejected rather than ignored, so an
agent never believes it bought deeper reasoning than it did.

`list_llms` reads capabilities and pricing from config first and litellm's bundled catalog
behind it, which is the same order the gateway uses at call time. Config declares those blocks
only to *override* litellm, so most deployments get their numbers from the catalog.

### Inline panels

`panel: ["gpt", "claude", "gemini"]` with `synthesizer: "gpt"` runs a panel assembled for that one
call. The llm names come from `list_llms`; everything else (strategy, tool arbitration, overflow
handling) takes the schema defaults a configured ensemble would get, because an inline panel is
resolved by the same resolver. Nothing is written to config — promoting a panel you like into a
named ensemble stays a human edit.

Inline panels are billed and grouped under the ensemble name **`mcp:adhoc`**, so `mom metrics usage
--by ensemble` can answer "what have ad-hoc panels cost me". The `:` is why that name is safe to
reuse across calls: config rejects `:` and `+` in ensemble names, so an inline panel can never
shadow one of yours. They also never coalesce: request identity keys on the ensemble *name* plus
the messages, so two different rosters asking the same question would otherwise collide and the
second would silently get the first one's answer.

### `consult` results

One envelope for every outcome, discriminated by `status`:

| `status` | Meaning | Shape |
| --- | --- | --- |
| `ok` | The panel answered | `answer` holds the synthesized text |
| `tool_calls` | The panel returned tool calls to execute | `tool_calls` populated (OpenAI wire shape); executing them is the caller's job, as there is no tool continuation over MCP. `answer` still holds any prose the model wrote alongside them |
| `failed` | The run died upstream (quorum not met, synthesis failure, timeout) | `error: {code, message, http_status}`, and the members that *did* complete are still listed with the tokens and money they spent |

`status` is decided by whether there are tool calls, not by `finish_reason` — a provider can
report `tool_calls` having emitted none we could parse, and that is an answer, not an empty
result. `finish_reason` is reported verbatim either way.

Every result also carries `ensemble`, `request_id`, `coalesced`, `progress_url`, `total_cost_usd`,
`usage`, `reasoning` (the synthesizer's, when it produced any), and `members[]` with each member's
`identity`, `status`, `cost_usd`, `duration_ms`, `cached` and client-safe `error`. A member the
fan-out deadline passed by appears with status `abandoned` rather than vanishing from the list.
`include_member_answers` adds each member's own text and reasoning; the synthesized answer and its
reasoning come back either way.

On a `failed` result the cost and token figures are a floor: they cover the members observed
before the failure, and a synthesizer that streamed and then failed emits nothing to count.
`usage` and `runs` remain the authority on what a run finally cost.

`progress_url` is `null` over stdio unless `server.public_url` is set (there is no request to
derive a host from). Unlike the `X-MoM-Progress-Url` response header, it never embeds the API
token: a tool result is data that lands in a model's context and travels with that agent's
transcript, and a stdio caller never presented a token in the first place. With `auth: bearer`
the link therefore needs your own token attached to open — an HTTP MCP client already has one.

Both a failed run and a malformed call come back as MCP `isError` results, and the difference is
what rides along: a **failed run** carries the full `ConsultResult` as structured content, so the
model can read what failed and what it cost and decide whether to retry or pick a different panel.
A **caller mistake** (both or neither of `ensemble`/`panel`, an unknown llm, a panel with no
synthesizer, an `effort` the ensemble has no tiers for) carries only a message, because nothing
ran and there is nothing to report; those spend nothing.

### What `runs` can and cannot see

Two sources, because neither knows the whole story. `recent` comes from the metrics ledger:
durable and shared across processes, but a call only lands there once the recorder's queue
drains. `in_flight` and `just_finished` come from an index the gateway keeps as progress events
pass through it — that index is what covers the window where a run has finished but the ledger
has not caught up, and it is the only place a still-running fan-out's spend is visible at all.

The index is **process-local**: a multi-worker gateway reports the worker that answered, and
`mom mcp` sees only consults it ran itself. It wraps whichever bus is configured, Redis included,
so this view survives a multi-process deployment for the process you asked. Only a gateway built
without one reports `in_flight_visibility: "none"`, which says "cannot tell" rather than letting
an empty list imply nothing is running.

Spend figures here are a lower bound: metrics are recorded off the hot path and dropped rather
than allowed to block a call (`GET /health` reports `metrics_dropped`).

Running `mom mcp` beside a live gateway is supported — SQLite is in WAL mode with a busy timeout,
so both processes read and write the same databases. The one unguarded case is two processes
racing a schema *migration*, which only happens on the first open after an upgrade.

---

## Effort tiers

Reasoning effort is one dial across all three protocols, resolved to a per-member, per-provider
parameter in two steps.

**1. Client effort → an ensemble tier.** Each protocol supplies effort differently:

| Protocol | Source | Notes |
| --- | --- | --- |
| Chat | `reasoning_effort` | a level name |
| Responses | `reasoning.effort` | `auto` is treated as unset |
| Anthropic | `thinking.budget_tokens` | bucketed → `2048:minimal, 8192:low, 16384:medium, 32768:high, 65536:xhigh, >65536:max`; `type:"disabled"` → `none` |

The ladder is `none < minimal < low < medium < high < xhigh < max`. If the ensemble defines
`effort_tiers`, the requested level snaps to the **nearest defined tier** (ties round up toward the
higher tier); with no client effort, the ensemble's `default_tier` applies. An ensemble without
`effort_tiers` ignores effort — members simply run their LLM's configured params.

**2. Tier → per-member provider params.** For the chosen tier, each member and the synthesizer have
an effort **cell**: a concrete level (`low`, `high`, …) or a sentinel —

- `off` — send no reasoning param (model default / non-reasoning model);
- `pass` — relay the client's own requested effort to this member;
- `skip` — exclude this member entirely at this tier.

**3. Provider clamping.** The resolved `reasoning_effort` is then clamped in the adapter to what each
provider actually accepts: OpenAI/Azure keep `minimal…high`; other providers map `minimal→low` and
cap at `high`; `xhigh`/`max` collapse to `high`; `none` drops the param. Combined with LiteLLM
`drop_params`, an unsupported effort is normalized or dropped instead of producing a provider 400.

---

## Usage and cost

**Cost per call** is resolved in priority order:

1. **Cache hit → $0** (and marked as a cache hit in metrics).
2. **Config pricing wins** — if the LLM has a `pricing:` block, cost is computed from token usage at
   those per-1M rates (cache-aware: cached-prompt tokens bill at the cache-read rate, cache-write
   tokens add at the cache-write rate).
3. **Otherwise the provider's real cost** — OpenRouter's returned usage cost, or LiteLLM's cost map.

The ensemble's total cost (every member plus the synthesizer) is summed internally and recorded to
metrics; query it via `GET /v1/metrics/usage`. Cost is not returned inline in chat/response bodies.

---

## Deviations

Behaviors worth knowing when comparing MoM to a single-model backend:

- **Usage accounting differs by surface.** The **Chat Completions** surface reports **aggregate
  ensemble usage** — `prompt_tokens` / `completion_tokens` / `total_tokens` (plus cached and
  reasoning detail) summed across every member call and the synthesizer, not the tokens of a single
  model. The **Responses** surface renders that same aggregate as `input_tokens` / `output_tokens`.
  The **Anthropic** surface instead reports a **transcript-perspective** `input_tokens` — the rough
  ~4-chars/token estimate over the visible request that `count_tokens` returns and that seeds
  `message_start` — while `output_tokens` is the aggregate completion count and the cache token
  fields are `0`.

- **`show_work` modes** (per-ensemble, `off` by default) control whether the panel's individual
  opinions are exposed alongside the synthesized answer:
  - `inline` — member perspectives are rendered as a `<think>…</think>` block prepended to the
    content stream (v1-compatible visibility of "the work");
  - `native` — no inline block; the answer's reasoning is surfaced through the provider-native
    reasoning channel (`reasoning_content` / Anthropic `thinking` blocks), and the ensemble is
    advertised as a reasoning model;
  - `off` — only the synthesized answer (the synthesizer's own reasoning deltas, if any, still pass
    through the native reasoning field).

- **Tool-call IDs pass through unchanged.** A tool call's `id` round-trips end to end — the
  synthesizer's emitted id becomes the `tool_calls[].id` (Chat), `call_id` (Responses `function_call`),
  or `tool_use.id` (Anthropic), and an incoming `tool_call_id` / `tool_use_id` on a tool result is
  preserved into the synthesizer's messages — so client-side tool execution correlates without the
  gateway rewriting identifiers.

- **Tools and structured output are the synthesizer's job.** Fan-out members are advisory; on a fresh
  turn they may receive a schema-free summary of the available tools (`member_tool_context: summary`)
  but cannot invoke them. Only the synthesizer emits client-visible tool calls or honors
  `response_format`.

- **Tool-continuation turns skip fan-out.** When the conversation tail is tool results, the request
  is a **relay**: it goes straight to the synthesizer (no fresh fan-out), recorded with
  `turn_type: relay`. A `FanoutSkipped` event marks it in the stream.
