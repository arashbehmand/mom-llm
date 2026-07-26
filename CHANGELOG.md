# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

## [2.0.0] - 2026-07-26

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
  a minimum context window aggregated from the actual panel members.
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

### Changed

- Rearchitected around a hexagonal, layered design with a pure domain; `import-linter` enforces the
  layers and quarantines `litellm` to a single adapter module.
- Client sampling controls (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`) are now honored on
  the synthesizer instead of being dropped silently.

### Quality

- 150+ tests, including SDK-in-the-loop contract tests that parse MoM's real ASGI streams with the
  official OpenAI and Anthropic SDKs. Gates: `ruff`, `mypy --strict`, and `import-linter`.
