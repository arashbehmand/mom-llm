# MoM — Mixture of Models

MoM is a self-hosted LLM gateway that speaks the OpenAI and Anthropic wire protocols but answers
each request with a *panel* of models instead of one. It fans a single request out to several
models in parallel, then a designated **synthesizer** reads all their candidate answers and writes
one consolidated response. Because MoM is drop-in wire-compatible, any tool that already talks to
OpenAI or Anthropic — SDKs, Claude Code, Codex, Cursor, Aider — gets ensemble answers with no code
changes: just point its base URL at MoM.

## The core idea

```
                     ┌─ model A ─┐
   your request ────▶├─ model B ─┤──▶ synthesizer ──▶ one answer
                     └─ model C ─┘   (owns the final text + tool calls)
```

Members answer independently and are *advisory*; the synthesizer owns the client-visible output.
Internally there is exactly **one** pipeline (`run_ensemble`) that emits a typed event stream, and
thin per-protocol encoders render that stream as Chat Completions, Responses, or Anthropic
Messages. Streaming and non-streaming are two consumers of the same events, so they cannot drift.

## Quickstart

Requires Python 3.12+ and an API key for at least one provider.

```bash
uv sync                        # install into .venv
export MOM_CONFIG=config.example.yaml
export MOM_API_TOKEN=dev-secret
export OPENAI_API_KEY=sk-...   # plus any other providers your config uses
mom serve                      # http://127.0.0.1:8000
```

Or with Docker:

```bash
docker compose up              # reads .env for secrets and MOM_CONFIG
```

Secrets come from the environment (or a gitignored `.env`); the YAML config only ever names the
env vars, never the values. Check the server with `curl localhost:8000/health`.

## API surfaces

The `model` you send is an **ensemble** from your config (e.g. `bmom`), not a raw provider model.

**OpenAI Chat Completions** — `POST /v1/chat/completions`

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dev-secret")
resp = client.chat.completions.create(
    model="bmom",
    messages=[{"role": "user", "content": "Compare Postgres and SQLite for a small app."}],
)
print(resp.choices[0].message.content)
```

**Anthropic Messages / Claude Code** — `POST /v1/messages`

Point the Anthropic SDK (or Claude Code) at MoM with two environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=dev-secret
claude                         # Claude Code now runs against your ensemble
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="dev-secret")
msg = client.messages.create(
    model="bmom", max_tokens=1024,
    messages=[{"role": "user", "content": "Explain WAL mode in SQLite."}],
)
print(msg.content[0].text)
```

**OpenAI Responses** — `POST /v1/responses` (used by Codex). Any surface also works over raw HTTP:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" \
  -d '{"model": "bmom", "messages": [{"role": "user", "content": "hi"}]}'
```

Also served: `GET /v1/models`, `/v1/models/{id}`, `/v1/model/info` (capability cards),
`POST /v1/messages/count_tokens`, `GET /v1/metrics/usage`, and `GET /health`. Auth is a bearer
token or an `x-api-key` header, compared in constant time.

## Features

- **Three compatible surfaces** — Chat Completions, Responses, and Anthropic Messages, all over one
  pipeline and a shared typed event stream, so streaming and non-streaming stay in lockstep.
- **Effort tiers** — a request's `reasoning_effort` selects a tier; each member declares its own
  effort per tier (a level, `pass` to relay the client's, `off`, or `skip`). No alias models.
- **Tool calling** — the synthesizer owns the tool calls; a tool result relays straight to it and
  skips a fresh fan-out, so multi-turn agent loops stay cheap.
- **Honest capability cards** — `/v1/models` reports vision, tools, reasoning, and web-search
  support plus a *minimum* context window computed from the actual panel, not aspirational numbers.
- **Automatic cost tracking** — per-call USD from litellm's cost map for direct providers and from
  OpenRouter's returned usage-cost; a config `pricing:` block is an optional override only.
- **Provider prompt caching** — Anthropic `cache_control` breakpoints and OpenAI/xAI
  `prompt_cache_key` affinity are injected automatically.
- **aiosqlite stores** — a response cache (TTL + size-cap eviction, optional coalescing of
  identical concurrent calls) and a usage/metrics table, both on WAL SQLite with batched writes
  drained off the request path.

## Configuration

One YAML file, `version: 2`, with three name-keyed maps: `llms:` (individual models, with `extends`
inheritance), `prompts:` (synthesis instructions), and `ensembles:` (what clients call). Each
ensemble lists advisory `members`, a `synthesizer`, a `strategy` (`synthesize` or `passthrough`),
and an effort-tier matrix.

Start from **[`config.example.yaml`](config.example.yaml)** and see
**[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** for the full reference. Validate and inspect with:

```bash
mom config validate config.example.yaml
mom config show config.example.yaml
```

## Use MoM as a provider

MoM is a drop-in `base_url` for the OpenAI SDK, the Anthropic SDK and Claude Code
(`ANTHROPIC_BASE_URL`), Codex (Responses API), and Cursor / Cline / Aider. See
**[docs/PROVIDERS.md](docs/PROVIDERS.md)** for per-tool setup.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the pipeline, event stream, and hexagonal layering
- [docs/API.md](docs/API.md) — endpoint and wire-format reference
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — the v2 config schema
- [docs/PROVIDERS.md](docs/PROVIDERS.md) — using MoM from SDKs and agent tools
- [docs/MIGRATION.md](docs/MIGRATION.md) — upgrading from v1

## Development

```bash
uv sync                 # dev + test dependency groups
uv run pytest           # 150+ tests, incl. SDK-in-the-loop contract tests
```

Quality gates: `ruff`, `mypy --strict`, and `import-linter`, which enforces the layered
architecture and keeps `litellm` quarantined in a single adapter module.

## License

MIT — see [LICENSE](LICENSE).
