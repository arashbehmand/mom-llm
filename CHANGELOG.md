# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Published container image** at `ghcr.io/arashbehmand/mom-llm`, built for `linux/amd64` and
  `linux/arm64` and tagged on release with the full version, `major.minor`, `major`, and `latest`.
- **PyPI releases** — `pip install mom-llm`, published from the release workflow via PyPI trusted
  publishing (OIDC, no stored API token).

## [2.0.0] - 2026-08-18

A ground-up rebuild of MoM as the `mom` package (src-layout, distribution `mom-llm`, CLI `mom`).

### Added

- **Three wire-compatible API surfaces** over a single engine: OpenAI Chat Completions
  (`POST /v1/chat/completions`), OpenAI Responses (`POST /v1/responses`), and Anthropic Messages
  (`POST /v1/messages`, plus `POST /v1/messages/count_tokens`). Also `GET /v1/models`,
  `/v1/models/{id}`, `/v1/model/info`, `GET /v1/metrics/usage`, and `GET /health`.
- **One orchestration pipeline** (`run_ensemble`) that emits a typed event stream, rendered by thin
  per-protocol encoders — so streaming and non-streaming responses share one code path and cannot
  drift.
- **Effort tiers**: a request's `reasoning_effort` selects a tier, and every member declares its own
  effort per tier (a level, `pass`, `off`, or `skip`). Replaces v1's per-effort alias models.
- **Tool calling** through the synthesizer, with relay continuations — a tool result skips a fresh
  fan-out and goes straight to the synthesizer, keeping multi-turn agent loops cheap.
- **Honest capability cards** on `/v1/models`: vision, tools, reasoning, and web-search support plus
  a minimum context window aggregated from the actual panel members. Discovery answers in the
  dialect the request asks for — the OpenAI list shape by default, the Anthropic list shape on an
  `anthropic-version`/`x-api-key` header, and Codex CLI's `{"models": []}` catalog on
  `?client_version=` (deliberately empty: Codex uses an entry's `base_instructions` verbatim as the
  model's system prompt, so any entry would replace its own agent prompt).
- **Automatic cost tracking**: per-call USD from litellm's cost map for direct providers and from
  OpenRouter's returned usage-cost; the config `pricing:` block is now an optional override only.
- **Provider prompt caching**: automatic Anthropic `cache_control` breakpoints and OpenAI/xAI
  `prompt_cache_key` affinity.
- **aiosqlite stores** for the response cache and usage metrics — WAL mode, `user_version`
  migrations, TTL + size-cap eviction, and batched writes drained off the request path.
- **Config v2** (`version: 2`): name-keyed `llms` / `prompts` / `ensembles` maps with a single
  `extends` inheritance primitive and a compact per-member effort matrix; inspected with
  `mom config validate` and `mom config show`.
- **Hardened auth**: bearer token or `x-api-key`, compared in constant time.

- **In-flight request coalescing** (`server.dedupe`, off by default): a chat completions request
  identical to one already running attaches to that run instead of starting a second fan-out +
  synthesis, streaming the same answer from token zero; the coalesced response carries
  `X-MoM-Coalesced: 1` and reports the original request's id. In-flight only — a run is dropped
  from consideration the instant it completes, so a deliberate regenerate afterward always starts
  fresh. `POST /v1/chat/completions` only for now.
- **SSE keepalive + anti-buffering on every streaming surface**: `server.stream_heartbeat` is now
  wired into `/v1/responses` and `/v1/messages` (previously chat completions only, so a slow
  member on those two surfaces streamed silently and could trip a client's idle timeout), and
  every SSE response now sets `Cache-Control: no-cache` / `X-Accel-Buffering: no` so an
  intermediary can't buffer the stream (and its heartbeats) away entirely.
- **`<<SYSTEM>>` directive block**, generalizing the old `<<CONCLUDING-INSTRUCTION>>` marker (which
  keeps working unchanged as an alias): an optional header of `exclude:`/`only:` (per-turn member
  selection), `show_work:`, and `synth:` (retarget the synthesizer), followed by the instruction
  text verbatim. Also fixes the legacy marker being silently dead on any multipart message (e.g.
  every Claude Code / image-bearing request), where it previously leaked its own raw markup into
  the panel instead of being stripped.
- **`mom metrics usage`** CLI (`--days`, `--ensemble`, `--by day|member|ensemble|status`): calls,
  billable calls, cache hit rate, cost, and a per-status breakdown (including previously-invisible
  `empty` and `timeout` outcomes) straight from the metrics store.
- Metrics schema v2: `finish_reason`, `error_kind`, `error_detail`, and `attempts` on every call
  row; `status` now records the full outcome (`ok`/`empty`/`timeout`/`error`/`detached`/`aborted`)
  instead of collapsing everything non-`ok` into one opaque `error`.
- **Tool-calling depth** (#14), building on the synthesizer-owned tool loop:
  - **Candidate envelope** — a member's proposed `tool_calls` are captured on its `ModelOutcome`
    (a tool-only proposal now counts as a real answer) and summarized to the synthesizer as
    advisory context.
  - **Tool strategies** `arbitrate` (default) | `vote` | `first` (`ensembles.<name>.tools.strategy`,
    with `vote_threshold`). `vote`/`first` hand members the real tool schemas and can return a
    member's call directly, skipping synthesis; both fall back to `arbitrate` when undecided.
  - **MoM-minted tool-call ids** with process-local custody: the client always sees a minted
    `call_…` id, never the provider-native one (notably Gemini's `__thought__` thought-signature),
    which is restored on a relay continuation to the same synthesizer.
  - **Streaming compat profiles** `compat` (default) | `strict` for tool-call deltas
    (`tools.stream_profile`; an AI-SDK `User-Agent` forces `compat`). `compat` re-emits
    `id`/`type`/`function.name` on every delta.
  - Responses `type: mcp` tool blocks are **forwarded** to a synthesizer whose provider supports
    remote MCP (else the clean 400 is kept), and streamed Anthropic thinking blocks now carry an
    opaque `signature_delta`.

### Changed

- Rearchitected around a hexagonal, layered design with a pure domain; `import-linter` enforces the
  layers and quarantines `litellm` to a single adapter module.
- Client sampling controls (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`) are now honored on
  the synthesizer instead of being dropped silently.

### Fixed

- **Provider failures were invisible.** A member call's real error was silently discarded on the
  hot path that actually matters — `UpstreamError` is a `MomError`, so the one branch that *did*
  log never ran for it — leaving operators with zero signal on real fan-out failures. Errors are
  now classified into a stable `ErrorKind` and a scrubbed, log/metrics/trace-only `error_detail`
  (API keys, bearer tokens, and query strings/paths are stripped before it's ever persisted),
  logged at the point of failure, and threaded through to Langfuse/OTel.
- **A whole failure mode was mislabelled.** A member that returned no content and no tool call
  (`status: "empty"`) was collapsed into the same opaque `error` bucket as a hard failure —
  indistinguishable in the metrics store even though it still billed tokens for nothing. Now
  recorded as its own status, with its own visual state on the progress dashboard.
- **mom was silently retrying worse than not retrying at all.** litellm's built-in retry wrapper
  (still enabled, invisibly) gave zero backoff for most error classes, retried non-retryable
  errors (auth, bad request, context-length overflow) just as hard as transient ones, and — on a
  retry that also failed — discarded that failure and re-raised the *first* attempt's error,
  hiding what actually went wrong. Retries are now entirely mom's own: only classified-transient
  errors, exponential backoff honoring a provider's `Retry-After`, the *last* attempt's error
  surfaced, and an honest `attempts` count recorded end to end (including previously-unrecorded
  failed synthesis calls).
- **The progress dashboard could show "1 pending" forever.** A member that finished in the same
  scheduling round as a client disconnect could be dropped with no event and no metric at all — a
  narrower window than "between the deadline and the next check," specifically the suspension
  point at `yield` itself. Fan-out accounting now treats "not yet reported to the consumer" as the
  single source of truth, so a member handed to the same-request cleanup path is always either
  reported or explicitly detached/cancelled, never silently lost.
- **The progress link stayed invisible for the length of the first fan-out call.** The
  `show_work: inline` think-block preamble (which carries the progress URL) opened only on the
  first completed member — now it opens as soon as fan-out starts, so the link is in the first
  streamed bytes regardless of how long any member takes.
- A dashboard watching a `vote`/`first` tool-call turn hung open forever (no terminal progress
  event was ever published on that path); a `passthrough`/relay turn showed a blank page
  indistinguishable from a stuck request (nothing was published at all).
- The progress event bus could evict and sentinel-close a request's own still-live channel the
  moment it tried to publish to itself again after a long gap (e.g. a slow synthesizer call) — the
  sweep ran before the touch that would have kept the channel alive. Same bug on the subscribe
  side, where a late subscriber could evict the very history it came to replay.
- Synthesis candidates were ordered by fan-out *completion* time, making a synthesizer's answer
  non-deterministic run to run for identical inputs; now ordered by ensemble config order.
- A `<<SYSTEM>>`/`<<CONCLUDING-INSTRUCTION>>` instruction was silently dropped on every
  passthrough and tool-continuation (relay) turn instead of reaching the model.

### Quality

- 500+ tests, including SDK-in-the-loop contract tests that parse MoM's real ASGI streams with the
  official OpenAI and Anthropic SDKs. Gates: `ruff`, `mypy --strict`, `import-linter`, a coverage
  floor on the core layers, and a Docker build + boot smoke test.
