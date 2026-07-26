# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
