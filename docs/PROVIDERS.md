# Use MoM as a Provider

MoM speaks the ordinary **OpenAI** and **Anthropic** wire protocols, so any client that can talk
to those APIs can point at MoM with **no special SDK** — you only change the base URL, the API
key, and the model name. Behind that single "model" is your whole ensemble.

Three things to set in every client:

| Setting | Value |
| --- | --- |
| **Base URL** | OpenAI-style clients: `http://HOST:8000/v1` · Anthropic-style clients: `http://HOST:8000` |
| **API key** | your **MoM token** (the value of `MOM_API_TOKEN`), *not* a provider key |
| **Model** | the **ensemble name** from your config (`bmom`, `mom-code`, …) |

Replace `HOST:8000` with wherever MoM listens (`localhost:8000` locally). The two base-URL shapes
exist because the Anthropic SDK appends `/v1/messages` to its base URL while OpenAI SDKs append
`/chat/completions` (or `/responses`) to a base URL that already ends in `/v1`. If you configured
`server.auth: none`, the API key can be any non-empty string.

MoM exposes, all under `/v1`: `POST /chat/completions`, `POST /responses`, `POST /messages`
(+ `/messages/count_tokens`), and `GET /models`. `GET /health` needs no auth. There is also an
optional [MCP surface](#mcp-clients) at `/mcp`, which is a different way to use MoM: a tool your
agent calls while staying on its own model, rather than a model it runs on.

---

## Claude Code

Claude Code talks the Anthropic Messages API. Set three environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=<your MoM token>
export ANTHROPIC_MODEL=mom-code          # any ensemble name from your config
claude
```

`ANTHROPIC_BASE_URL` has **no** `/v1` suffix — Claude Code adds `/v1/messages` itself. A
`passthrough` ensemble (like `mom-code`) is the usual choice here: one strong tool-capable model
with `tools.continuation: relay`, so agentic tool loops stay fast and coherent.

---

## Codex CLI

Codex uses the OpenAI **Responses** API. Point it at MoM's `/v1` base URL in
`~/.codex/config.toml`:

```toml
model = "bmom"                            # an ensemble name
model_provider = "mom"

[model_providers.mom]
name = "MoM"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "MOM_API_TOKEN"                 # Codex reads the token from this env var
```

```bash
export MOM_API_TOKEN=<your MoM token>
codex
```

MoM's `/v1/responses` endpoint implements the stateless subset of the Responses API (streaming,
tools, and reasoning), which is what Codex needs.

---

## Cursor / Cline / Aider / Continue

These are OpenAI-compatible chat clients — set an OpenAI base URL of `http://localhost:8000/v1`,
the MoM token as the OpenAI API key, and an ensemble name as the model.

**Cursor** (Settings → Models → OpenAI API Key → "Override OpenAI Base URL"):

```
Base URL:  http://localhost:8000/v1
API Key:   <your MoM token>
Model:     bmom
```

**Cline** (VS Code extension → API Provider: "OpenAI Compatible"):

```
Base URL:  http://localhost:8000/v1
API Key:   <your MoM token>
Model ID:  bmom
```

**Aider**:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=<your MoM token>
aider --model openai/bmom
```

**Continue** (`~/.continue/config.yaml`):

```yaml
models:
  - name: MoM Panel
    provider: openai
    model: bmom
    apiBase: http://localhost:8000/v1
    apiKey: <your MoM token>
```

---

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="<your MoM token>",
)

# Chat Completions
resp = client.chat.completions.create(
    model="bmom",  # ensemble name
    messages=[{"role": "user", "content": "Compare two sorting algorithms."}],
    reasoning_effort="high",  # snapped to the ensemble's nearest tier
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")

# Responses API works too, against the same base URL:
r = client.responses.create(model="bmom", input="Give me three startup ideas.")
print(r.output_text)
```

---

## Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",  # NO /v1 — the SDK adds /v1/messages
    api_key="<your MoM token>",
)

msg = client.messages.create(
    model="mom-code",  # ensemble name
    max_tokens=1024,
    messages=[{"role": "user", "content": "Refactor this function for clarity."}],
)
print(msg.content[0].text)
```

Streaming (`client.messages.stream(...)`) and tool use work the same way.

---

## MCP clients

Everything above makes MoM the *model* a client runs on. MCP is the other arrangement: your agent
keeps its own model and calls the panel as a tool when a question is worth more than one opinion.
Turn the surface on with `server.mcp: { enabled: true }` (off by default), then:

```bash
# Claude Code, over HTTP — same port and token as /v1
claude mcp add --transport http mom http://localhost:8000/mcp \
  --header "Authorization: Bearer <your MoM token>"

# ...or locally over stdio, with no gateway running at all
claude mcp add mom -- mom mcp
```

Clients that read a JSON config (Cursor, Cline, Continue, Codex) take the same two shapes:

```jsonc
{"mcpServers": {
  "mom": {"type": "http", "url": "http://localhost:8000/mcp",
          "headers": {"Authorization": "Bearer <your MoM token>"}},
  "mom-local": {"command": "mom", "args": ["mcp"]}
}}
```

The agent then has `consult` (a configured ensemble, or a panel it assembles from `list_llms` for
that one question) plus read-only `list_llms`, `list_ensembles`, `runs`, `usage`, and
`cache_stats`. The stdio form shares the gateway's databases, so a consult run there shows up in
`mom metrics usage` and warms the same cache. Full tool and result reference:
[API.md](API.md#mcp-mcp-and-mom-mcp).

---

## Compatibility matrix

All ensembles stream. "Tools" and "Reasoning" columns mark protocol support on that surface; the
*actual* availability for a given ensemble is what its `/v1/models` capability card advertises
(tools follow the synthesizer; reasoning is present when the ensemble defines effort tiers or
shows work).

| Client | Surface used | Base URL | Streaming | Tools | Reasoning |
| --- | --- | --- | --- | --- | --- |
| Claude Code | Anthropic Messages (`/v1/messages`) | `http://HOST:8000` | ✅ | ✅ | ✅ native thinking / effort tiers |
| Codex CLI | OpenAI Responses (`/v1/responses`) | `http://HOST:8000/v1` | ✅ | ✅ | ✅ `reasoning` / effort |
| Cursor | OpenAI Chat (`/v1/chat/completions`) | `http://HOST:8000/v1` | ✅ | ✅ | ✅ `reasoning_effort` |
| Cline | OpenAI Chat | `http://HOST:8000/v1` | ✅ | ✅ | ✅ |
| Aider | OpenAI Chat | `http://HOST:8000/v1` | ✅ | ✅ | ✅ |
| Continue | OpenAI Chat | `http://HOST:8000/v1` | ✅ | ✅ | ✅ |
| `openai` SDK | OpenAI Chat **or** Responses | `http://HOST:8000/v1` | ✅ | ✅ | ✅ |
| `anthropic` SDK | Anthropic Messages | `http://HOST:8000` | ✅ | ✅ | ✅ |
| any MCP client | MCP (`/mcp` or `mom mcp`) | `http://HOST:8000/mcp` | — (progress notifications) | n/a — MoM *is* the tool | ✅ `effort` argument |

### Notes

- **Model = ensemble name.** List what's available at `GET /v1/models` (send your token). Each
  entry includes a `mom` block with the members, the synthesizer, and the supported parameters.
- **Reasoning effort.** Clients that send `reasoning_effort` (OpenAI) or extended-thinking
  requests (Anthropic) have that mapped onto the ensemble's nearest effort tier; each member then
  applies its own configured depth for that tier.
- **Web search.** If the client sends a web-search request and the ensemble has search-capable
  members, those members answer with search enabled.
- **Auth.** Send the token as `Authorization: Bearer <token>` (OpenAI clients) or `x-api-key`
  (Anthropic clients) — both are accepted. With `server.auth: none`, no token is required.
