# MoM v2 API Reference

MoM exposes three drop-in-compatible chat protocols — **OpenAI Chat Completions**, **OpenAI
Responses**, and **Anthropic Messages** — plus model-discovery, metrics, and health endpoints. A
"model" in every request is the name of an **ensemble**: MoM fans the prompt out to that ensemble's
panel of members and returns one synthesized answer, rendered in whichever protocol you called.

All application endpoints are mounted under `/v1`. Point any OpenAI or Anthropic SDK at the
gateway's base URL and set the model to an ensemble name.

- [Authentication](#authentication)
- [Errors](#errors)
- [`POST /v1/chat/completions`](#post-v1chatcompletions)
- [`POST /v1/responses`](#post-v1responses)
- [`POST /v1/messages` and `/v1/messages/count_tokens`](#post-v1messages)
- [Model discovery](#model-discovery)
- [`GET /v1/metrics/usage`](#get-v1metricsusage)
- [`GET /v1/progress/{id}`](#get-v1progressid)
- [`GET /health`](#get-health)
- [Effort tiers](#effort-tiers)
- [Usage and cost](#usage-and-cost)
- [Deviations](#deviations)

---

## Authentication

Auth is a per-request dependency on every `/v1` route. Present the token either way OpenAI and
Anthropic SDKs send it:

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

When the request carries `anthropic-version` or `x-api-key`, `GET /v1/models` returns the **Anthropic
list shape** instead (`{data: [{type:"model", id, display_name, created_at}], has_more, first_id,
last_id}`). `GET /v1/models/{id}` returns the OpenAI single-model object, or **404** for an unknown
ensemble.

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

## `GET /health`

Unauthenticated liveness probe. Returns `{"status": "ok", "version": "<v>"}`.

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
