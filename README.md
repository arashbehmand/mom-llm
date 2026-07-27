# 🎭 MoM — Mixture of Models

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arashbehmand/mom-llm)

> **Transform multiple AI perspectives into one superior answer through intelligent synthesis.**

MoM is a self-hosted LLM gateway that speaks the OpenAI and Anthropic wire protocols but answers
each request with a *panel* of models instead of one. It fans a single request out to several models
in parallel, then a designated **concluding model** (the *synthesizer*) reads every candidate answer
and writes one consolidated response.

Think of it as assembling an expert panel: the creativity of a frontier GPT, the reasoning of Claude
Opus, and the breadth of Gemini Pro — combined into a single answer that is more reliable and nuanced
than any one model produces alone. Because MoM is drop-in wire-compatible, any tool that already
talks to OpenAI or Anthropic — the SDKs, Claude Code, Codex, Cursor, Cline, Aider — gets ensemble
answers with **no code changes**: just point its base URL at MoM.

## 🌟 Why a Mixture of Models?

In a landscape of hundreds of specialized LLMs, leaning on a single model is a self-imposed ceiling.

![Different AI models offer unique perspectives on the same question](docs/neo-fork.png)
*Each model brings its own perspective and reasoning style; MoM synthesizes them into one
comprehensive answer.*

| Benefit | How MoM delivers it |
|---|---|
| **🎯 Superior quality** | Synthesis reconciles several perspectives, dropping one model's hallucinations and biases in favor of the panel's strongest reasoning. |
| **🛡️ Resilience** | If a member is slow or fails, a quorum of the others still answers — one bad seat never sinks the request. |
| **💰 Cost control** | Cheap models on the panel, a stronger one to conclude; per-tier effort, prompt caching, and relay continuations keep spend honest. |
| **🔄 Flexibility** | Hot-swap models and reshape ensembles in one YAML file — no code changes. Build task-specific "meta-models" (`bmom`, `mom-code`, …). |

### Real-world use cases

- **💻 Coding agents** — point Claude Code / Codex / Cursor at MoM and let a panel deliberate on each step.
- **🔍 Research & analysis** — consult multiple AI "experts" and get one reconciled answer.
- **📝 Content creation** — combine creative and factual models for balanced, grounded writing.
- **🎓 Education** — well-rounded explanations drawn from diverse reasoning styles.

## 🔄 How it works

A **fan-out / fan-in** architecture: members answer in parallel and are *advisory*; the synthesizer
owns the client-visible output (text **and** tool calls).

```mermaid
graph LR
    A[Client request<br/>OpenAI / Anthropic wire] --> B{MoM}
    B --> M1[member A]
    B --> M2[member B]
    B --> M3[member C]
    M1 --> S{{synthesizer<br/>concluding model}}
    M2 --> S
    M3 --> S
    S --> R[one answer<br/>streamed back]
```

Internally there is exactly **one** pipeline (`run_ensemble`) that emits a typed event stream, and
thin per-protocol encoders render that stream as Chat Completions, Responses, or Anthropic Messages.
Streaming and non-streaming are two consumers of the *same* events, so they can never drift.

## 🚀 Quickstart

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

Secrets come from the environment (or a gitignored `.env`); the YAML config only ever *names* the
env vars, never the values. Check the server with `curl localhost:8000/health`.

## 🔌 API surfaces

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

**Anthropic Messages / Claude Code** — `POST /v1/messages`. Point the Anthropic SDK (or Claude Code)
at MoM with two environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=dev-secret
claude                         # Claude Code now runs against your ensemble
```

**OpenAI Responses** — `POST /v1/responses` (used by Codex). Any surface also works over raw HTTP:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" \
  -d '{"model": "bmom", "messages": [{"role": "user", "content": "hi"}]}'
```

Also served: `GET /v1/models`, `/v1/models/{id}`, `/v1/model/info` (capability cards),
`POST /v1/messages/count_tokens`, `GET /v1/metrics/usage`, `GET /v1/progress/{id}` (SSE), and
`GET /health`. Auth is a bearer token or an `x-api-key` header, compared in constant time.

## ✨ Features

- **Three compatible surfaces** — Chat Completions, Responses, and Anthropic Messages over one
  pipeline and a shared typed event stream, so streaming and non-streaming stay in lockstep.
- **🖼️ Multimodal / vision** — send images (OpenAI or Anthropic format); a vision request runs on
  the members that can see, and incapable members drop out cleanly.
- **Effort tiers** — a request's `reasoning_effort` selects a tier; each member declares its own
  effort per tier (a level, `pass` to relay the client's, `off`, or `skip`). No alias models.
- **Tool calling** — the synthesizer owns the tool calls; a tool result relays straight to it and
  skips a fresh fan-out, so multi-turn agent loops stay cheap.
- **🔎 Request-triggered web search** — mark a model search-capable with a `search:` block; it goes
  online only when the client asks (see [Web search](#web-search) below). No always-on search models.
- **Honest capability cards** — `/v1/models` reports vision, tools, reasoning, and web-search support
  plus a *minimum* context window computed from the real panel, not aspirational numbers.
- **Automatic cost tracking** — per-call USD from litellm's cost map for direct providers and from
  OpenRouter's returned usage-cost; a config `pricing:` block is an optional override only.
- **Provider prompt caching** — Anthropic `cache_control` breakpoints and OpenAI/xAI
  `prompt_cache_key` affinity are injected automatically.
- **aiosqlite stores** — a response cache (TTL + size-cap eviction, optional coalescing of identical
  concurrent calls) and a usage/metrics table, both on WAL SQLite with batched off-path writes.

## 🎯 Advanced features

### Thinking context — see the panel's work

Set `show_work: inline` on an ensemble to prepend a `<think>` block that shows every member's own
answer before the synthesized one — useful for transparency and debugging:

```
<think>
Model: openai/gpt-5.6-sol
Content: [that member's answer]
---
Model: anthropic/claude-opus-4-8
Content: [that member's answer]
</think>

[the synthesized answer]
```

`show_work: native` routes it through the provider's reasoning channel instead; `off` (the default)
hides it.

### Concluding instruction — steer the synthesis, not the panel

Wrap a directive in `<<CONCLUDING-INSTRUCTION>>…<</CONCLUDING-INSTRUCTION>>` anywhere in your
message. MoM **strips it from what the fan-out members see** and hands it **only to the concluding
model**, so you can steer the final synthesis without biasing the panel that feeds it:

```
Summarize the trade-offs of WAL mode in SQLite.
<<CONCLUDING-INSTRUCTION>>Answer as a terse bullet list, no preamble.<</CONCLUDING-INSTRUCTION>>
```

The members answer the plain question; the synthesizer additionally obeys the instruction.

### Web search

A model becomes search-capable by carrying a `search:` block; its provider search params are merged
in **only when the client requests web search** (`web_search` / `web_search_options`, an Anthropic
`web_search` tool, or a Responses `web_search` tool). Offline requests leave it untouched.

```yaml
llms:
  k3:  # OpenRouter: the web plugin, request-triggered (no separate always-on ":online" model)
    model: openrouter/moonshotai/kimi-k3
    search: { extra_body: { plugins: [{ id: web }] } }
  gemini:  # Google Search grounding
    model: gemini/gemini-3.1-pro
    search: { web_search_options: { search_context_size: high } }
```

## ⚙️ Configuration

One YAML file, `version: 2`, with three name-keyed maps: `llms:` (individual models, with `extends`
inheritance), `prompts:` (synthesis instructions), and `ensembles:` (what clients call). Each
ensemble lists advisory `members`, a `synthesizer`, a `strategy` (`synthesize` or `passthrough`),
and per-member effort. Secrets never appear — only env-var names.

Start from **[`config.example.yaml`](config.example.yaml)** and see
**[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** for the full reference. Validate and inspect with:

```bash
mom config validate config.example.yaml
mom config show config.example.yaml bmom     # flattened, resolved view of one ensemble
```

## 🔗 Use MoM as a provider

MoM is a drop-in `base_url` for the OpenAI SDK, the Anthropic SDK and Claude Code
(`ANTHROPIC_BASE_URL`), Codex (Responses API), and Cursor / Cline / Aider. See
**[docs/PROVIDERS.md](docs/PROVIDERS.md)** for copy-paste per-tool setup.

## 📚 Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the pipeline, event stream, and hexagonal layering
- [docs/API.md](docs/API.md) — endpoint and wire-format reference
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — the v2 config schema
- [docs/PROVIDERS.md](docs/PROVIDERS.md) — using MoM from SDKs and agent tools
- [docs/MIGRATION.md](docs/MIGRATION.md) — upgrading from v1

## 🛠️ Development

```bash
uv sync                 # dev + test dependency groups
uv run pytest           # incl. SDK-in-the-loop contract tests
```

Quality gates: `ruff`, `mypy --strict`, and `import-linter`, which enforces the layered architecture
and keeps `litellm` quarantined in a single adapter module.

## License

MIT — see [LICENSE](LICENSE).
