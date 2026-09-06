# Upgrading from v1 to v2

v2 is a ground-up rebuild of MoM. Two things change for you, and one thing deliberately doesn't:

1. **How you run it** — a new package (`mom`, not `mom_service`), a `mom` CLI instead of a raw
   `uvicorn` invocation, one service instead of two, and a real data directory.
2. **How you configure it** — v1's parallel *lists* keyed by string names (`base_name:suffix`)
   become v2 *maps* keyed by name, with a single inheritance primitive (`extends`) and a compact
   per-member effort matrix.
3. **Your clients don't change.** They keep speaking the ordinary OpenAI and Anthropic APIs against
   MoM (see [PROVIDERS.md](./PROVIDERS.md)); there was never a proprietary MoM client and there
   still isn't. Existing base URLs and bearer tokens keep working.

Migration is a one-time, by-hand job: swap the run command, move your data directory, and convert
the YAML. Budget half an hour. Part 1 covers the deployment, Part 2 the config file.

> **Roll back at any time** by checking out the `v1.11.1` tag — v2 writes its state to a new
> location and never touches v1's database files, so a v1 process can be brought back up untouched.

---

# Part 1 — the deployment

## Requirements

Python **3.12+**, now declared and enforced (`requires-python = ">=3.12"`; v1 claimed 3.9+).
[uv](https://docs.astral.sh/uv/) is the supported installer; plain pip works.

## Install and run

| | v1 | v2 |
| --- | --- | --- |
| Install | `pip install -r requirements.txt` | `uv sync` (or `pip install .`) |
| Import package | `mom_service` | `mom` |
| Distribution | — (not packaged) | `mom-llm` |
| Run the gateway | `uvicorn mom_service.main:app --port 8000` | `mom serve --port 8000` |
| Run the dashboard | `uvicorn mom_service.reporting.main:app --port 8001` | *(same process — see below)* |
| Dev reload | `UVICORN_RELOAD=true` | `mom serve --reload` |
| Health probe | `curl localhost:8000/health` | `mom healthcheck` (or the same curl) |

```bash
uv sync
export MOM_CONFIG=config.yaml
export MOM_API_TOKEN=…            # your old API_TOKEN value still works under that name too
mom serve --host 0.0.0.0 --port 8000
```

`requirements.txt` is gone; dependencies are pinned in `uv.lock` and declared in `pyproject.toml`.

## The reporting service is gone — it's one app now

v1 ran a **second** process on port 8001 (`mom_service.reporting.main:app`) that served the
progress dashboard, and the two processes talked over Redis. v2 folds it into the gateway:

- The progress page moved from `http://…:8001/progress/{request_id}` to
  **`GET /v1/progress/{request_id}`** on the main app. A browser (`Accept: text/html`) gets a live
  page; anything else gets the raw SSE event stream.
- It is authenticated. Send the bearer token, or — since a plain link can't carry a header —
  append `?token=…`. MoM hands you a ready-made link per request in the **`X-MoM-Progress-Url`**
  response header (and inside the think block when the ensemble sets `show_work: inline`); the
  token in that link is a per-request [link token](API.md#how-the-link-authenticates), not your
  API token.
- Behind a reverse proxy, set `server.public_url` in the config so that link points at your public
  hostname rather than the internal one.
- **Stop running the 8001 service** and drop `REPORTING_SERVICE_URL` from your environment; it is
  no longer read by anything.
- **Redis is now optional.** It was mandatory in v1's compose file purely to carry progress events
  between the two processes. v2's event bus is in-memory by default; set `MOM_REDIS_URL` only if you
  run several worker processes and want the dashboard to see events from all of them.

## Move your data directory

v1 wrote its SQLite files *inside the package directory*:

```
mom_service/llm_cache.db        # response cache
mom_service/usage_metrics.db    # usage + cost metrics
```

v2 writes to a proper data directory — `MOM_DATA_DIR`, else `storage.data_dir` from the config,
else the platform default (`~/.local/share/mom-llm` on Linux, `/data` in the Docker image,
`~/Library/Application Support/mom-llm` on macOS):

```
$MOM_DATA_DIR/cache.db
$MOM_DATA_DIR/metrics.db
```

**Both schemas changed and there is no automatic import.** The cache table was rebuilt (`cache` →
`entries`, with TTL and size-cap eviction), and the metrics table gained per-call `finish_reason`,
`error_kind`, `error_detail`, and `attempts` plus a real `status` (`ok`/`empty`/`timeout`/`error`/
`detached`/`aborted`) where v1 collapsed everything non-OK into one opaque bucket. Practically:

- **Cache** — start empty and let it re-warm. Nothing is lost but a few repeated calls.
- **Metrics** — v2 starts a fresh history. Keep the old `usage_metrics.db` around if you want your
  v1 cost history; query it with any SQLite client. Don't copy it over `metrics.db` — the schema
  won't match and MoM's `user_version` migrations will not rescue it.

Point `MOM_DATA_DIR` at a directory that survives redeploys (a volume, not the container layer).

## Environment variables

Every legacy name is still **accepted as an alias**, so an existing `.env` keeps working; the
`MOM_`-prefixed spelling wins when both are set. See
[Environment & settings](./CONFIGURATION.md#environment--settings) for the full table.

| v1 | v2 | Status |
| --- | --- | --- |
| `API_TOKEN` | `MOM_API_TOKEN` | alias kept |
| `MOM_CONFIG_PATH` | `MOM_CONFIG` | alias kept |
| `REDIS_URL` | `MOM_REDIS_URL` | alias kept, now optional |
| `LITELLM_VERBOSE` | `MOM_LITELLM_DEBUG` | alias kept |
| — | `MOM_DATA_DIR` | new — where the SQLite files live |
| — | `MOM_CONFIG_OVERLAY` | new — a file deep-merged last, over everything else |
| — | `MOM_AUTH_FROM_OPENCODE` | new — also read API keys from opencode's `auth.json` |
| — | `MOM_HOST` / `MOM_PORT` / `MOM_LOG_LEVEL` / `MOM_LOG_FORMAT` | new |
| `ALLOWED_CORS_ORIGINS` | *(none)* | **removed** — CORS moved into the config under `server.cors` |
| `REPORTING_SERVICE_URL` | *(none)* | **removed** — no second service |
| `UVICORN_APP` / `UVICORN_PORT` / `UVICORN_RELOAD` / `UVICORN_WORKERS` | *(none)* | **removed** — use `mom serve` flags |

Provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) and per-model proxy URLs work exactly
as before. [`.env.example`](../.env.example) is the refreshed reference.

## Config discovery is additive

`MOM_CONFIG` still works and now means something slightly stronger: it **pins** one file and turns
the search path off entirely, so an existing deployment resolves exactly the config it always did.
What is new is that omitting it no longer fails — mom looks for `~/.mom/config.yaml` (or
`$XDG_CONFIG_HOME/mom/config.yaml`) and `./mom.yaml` (or `./.mom/config.yaml`), merges them, and
serves the result. `mom config where` shows what it found. See
[Where the config comes from](./CONFIGURATION.md#where-the-config-comes-from).

Two behaviour changes worth knowing about:

- **The uvicorn target moved.** `mom serve` now runs `mom.api.app:serve_app`, which resolves the
  search path in the serving process. `uvicorn mom.api.app:create_app --factory` still works and
  still discovers its config, but it no longer installs CORS from that config — middleware has to
  be added before the app is first called, and `create_app` deliberately reads nothing. Point
  runbooks at `mom.api.app:serve_app` (or just use `mom serve`).
- **`null` in `MOM_CONFIG_OVERLAY` now deletes an inherited key**, matching `extends:` and what
  this document and [CONFIGURATION.md](./CONFIGURATION.md) already described. It previously set
  the key to `null` instead.

## Docker

The image is uv-based, runs as non-root, and its entrypoint is the `mom` CLI:

```bash
docker build -t mom-llm:latest .
docker run -p 8000:8000 \
  -e MOM_CONFIG=/config.yaml -e MOM_API_TOKEN=… \
  -v "$PWD/config.yaml:/config.yaml:ro" -v mom-data:/data \
  mom-llm:latest
```

Compose changes, if you used it:

| v1 | v2 |
| --- | --- |
| services `mom-service` + `mom-reporting` + `redis` | one `mom` service; `redis` behind `--profile redis` |
| ports 8000 **and** 8001 | 8000 only |
| `- ./config.yaml:/app/config.yaml:ro` | `- ${MOM_CONFIG_FILE:-./config.example.yaml}:/app/config.yaml:ro` |
| `- ./data:/app/data` | named volume `mom-data:/data` |
| `API_TOKEN=${API_TOKEN:-your-secret-token-here}` | `API_TOKEN: ${API_TOKEN:?…}` — fails fast, no insecure default |
| `docker/start-uvicorn.sh` + `UVICORN_*` | `ENTRYPOINT ["mom"]`, `CMD ["serve", …]` |

Set `MOM_CONFIG_FILE` in your `.env` to the path of your real config; `docker compose up` otherwise
boots the shareable `config.example.yaml`.

## Endpoint changes

Everything v1 served is still served, at the same paths, except two ops endpoints:

| Endpoint | v1 | v2 |
| --- | --- | --- |
| `POST /v1/chat/completions` | ✅ | ✅ |
| `GET /v1/models` | ✅ | ✅ (now with capability cards) |
| `GET /v1/metrics/usage` | ✅ | ✅ |
| `GET /health` | ✅ | ✅ |
| `GET /health/detailed` | ✅ | **removed** — `/health` reports status + version; use `mom cache stats` / `mom metrics usage` to inspect the stores |
| `GET /v1/metrics/usage/raw` | ✅ | **removed** — use `mom metrics usage` |
| `GET /progress/{id}` (port 8001) | ✅ | moved to `GET /v1/progress/{id}` |
| `POST /v1/responses` | — | **new** (OpenAI Responses, stateless subset) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | — | **new** (Anthropic Messages) |
| `GET /v1/models/{id}`, `GET /v1/model/info` | — | **new** |

## New CLI

Beyond `mom serve`, v2 gives you offline tools v1 had no equivalent for:

```bash
mom config validate config.yaml     # schema + reference check, before restarting anything
mom config show config.yaml mom     # the resolved ensemble, including the effort matrix
mom metrics usage --days 7 --by member
mom cache stats
mom cache purge --yes
```

Run `mom config validate` against your converted file **before** you cut over.

---

# Part 2 — the config file

See [CONFIGURATION.md](./CONFIGURATION.md) for the full v2 reference; this is the field-by-field
mapping.

## What changed, at a glance

| v1 | v2 |
| --- | --- |
| (no version marker) | `version: 2` (required) |
| `llm_definitions:` — a **list** with `base_name` + `variants`/`suffix` | `llms:` — a **map** keyed by name, with `extends` (or a nested `variants:` block) |
| `prompt_definitions:` — a list of `{name, content}` | `prompts:` — a map of `{name: content}` |
| `models:` — a list of ensembles | `ensembles:` — a map keyed by name |
| `llms_to_query:` (list of names) | `members:` (list, each with per-tier `effort`) |
| `concluding_llm:` + `concluding_prompt:` | `synthesizer: { llm, prompt }` |
| `include_thinking_context: true` | `show_work: off \| inline \| native` |
| baked-in `reasoning_effort` variants (`oai-o3:m`) | separate `llms`, **or** an ensemble `effort_tiers` matrix |
| per-**token** `pricing` | per-**1M** `pricing`, and now **optional** (cost is automatic) |
| `api_mode: responses` | `api: responses` |
| `service.timeout_seconds: 30` | `defaults.call.timeout: 30s` (duration string) |
| top-level `langfuse:` | `observability.langfuse:` |
| CORS via `ALLOWED_CORS_ORIGINS` | `server.cors:` |
| names like `oai-o3:m` | `:` and `+` are **reserved** — rename (e.g. `oai-o3-m`) |

## `llm_definitions` (list) → `llms` (map)

In v1, one entry could spawn several models via `variants`, each named `<base_name>:<suffix>`. In
v2 every model is its own map entry, and shared settings come from `extends` (which deep-merges
`params` — a `null` value deletes an inherited key). This removes the `base:suffix` string sprawl
and makes each callable model explicit.

The old variant mechanism was often used to bake a fixed `reasoning_effort` into a suffix (an
`oai-o3:m` for "medium"). In v2 you have three choices:

- **Consolidate into an ensemble effort matrix** — define the model once and let the ensemble's
  `effort_tiers` pick the depth per request. This is the idiomatic v2 approach and removes the need
  for `mom-high`-style alias models entirely. See
  [the effort matrix](./CONFIGURATION.md#the-effort-matrix).
- **Keep them as distinct llms** via a nested
  [`variants:`](./CONFIGURATION.md#variants--a-compact-effort-family-nested) block — sugar for one
  base plus several fixed-effort children, which is the closest thing to v1's `suffix` and the
  least noisy way to express an effort family.
- **Write each one out** with `extends` from a common base, when the siblings differ by more than
  effort.

## `models` (list) → `ensembles` (map)

`llms_to_query` becomes `members`; `concluding_llm` + `concluding_prompt` become a single
`synthesizer: { llm, prompt }`. Each member may now declare its own effort per tier rather than
inheriting a fixed variant. v1's `include_thinking_context` (exposing member reasoning) maps to
the v2 `show_work: off | inline | native` knob on the ensemble.

If a v1 ensemble simply listed *every* model you had defined, v2 has shorthand for that:
[`members: { all: true, exclude: [...] }`](./CONFIGURATION.md#members-all--a-self-maintaining-kitchen-sink-panel),
which keeps the panel current as you add models.

## Pricing: per-token → per-1M, and now optional

v1 required per-**token** rates (`prompt_cost_per_token`, `completion_cost_per_token`,
`reasoning_cost_per_token`). v2 uses per-**1M-token** rates (`input_per_1m`, `output_per_1m`,
`reasoning_per_1m`, plus `cache_read_per_1m` / `cache_write_per_1m`) — multiply your old numbers
by 1,000,000.

More importantly, **pricing is now optional**. MoM measures real cost automatically (LiteLLM's
cost map for direct providers, OpenRouter's returned usage cost for OpenRouter models). Keep a
`pricing:` block only for a model neither source can price; otherwise delete it.

## Before / after

**v1**

```yaml
llm_definitions:
  - base_name: "oai-g4.1"
    model: "openai/gpt-4.1"
    params: { temperature: 0.7 }
  - base_name: "oai-o3"
    model: "openai/o3-mini"
    variants:
      - suffix: "m"          # -> "oai-o3:m", a fixed medium-effort variant
  - base_name: "g25f"
    model: "gemini/gemini-2.5-flash-preview-04-17"

prompt_definitions:
  - name: "synth_default"
    content: |
      Synthesize the expert responses above into one superior answer.

models:
  - name: "mom"
    llms_to_query: ["oai-g4.1", "oai-o3:m", "g25f"]
    concluding_llm: "g25f"
    concluding_prompt: "synth_default"
    include_thinking_context: true

service:
  timeout_seconds: 30

langfuse:
  public_key_env: "LANGFUSE_PUBLIC_KEY"
```

**v2**

```yaml
version: 2

defaults:
  call: { timeout: 30s }

observability:
  langfuse: { enabled: true }   # key env-var names default to the LANGFUSE_* conventions

llms:
  oai-g4.1: { model: openai/gpt-4.1, params: { temperature: 0.7 } }
  oai-o3:   { model: openai/o3-mini }        # effort now comes from the tier, not a suffix
  g25f:     { model: gemini/gemini-2.5-flash-preview-04-17 }

prompts:
  synth_default: |
    Synthesize the expert responses above into one superior answer.

ensembles:
  mom:
    effort_tiers: [low, medium, high]
    default_tier: medium
    members:
      - { llm: oai-g4.1, effort: pass }        # relay the client's requested effort
      - { llm: oai-o3,   effort: [l, m, h] }   # follows the tier (replaces the ":m" variant)
      - { llm: g25f,     effort: off }         # no reasoning param
    synthesizer: { llm: g25f, prompt: synth_default }
    show_work: native                          # replaces include_thinking_context
```

Note that `oai-o3:m` disappeared: the model is defined once as `oai-o3` and the ensemble's
`effort_tiers` supply the depth. The old `:` in the variant name would be rejected by v2 anyway
(`:` and `+` are reserved in `llms`/`ensembles` names).

---

## Checklist

**Config**

1. Add `version: 2` at the top.
2. Turn `llm_definitions` into an `llms:` map; replace `variants`/`suffix` with a nested `variants:`
   block or explicit `extends` entries, and rename any name containing `:` or `+`.
3. Turn `prompt_definitions` into a `prompts:` map.
4. Turn `models` into an `ensembles:` map: `llms_to_query` → `members`, `concluding_llm` +
   `concluding_prompt` → `synthesizer`, `include_thinking_context` → `show_work`, and decide effort
   per member (tiers vs. fixed).
5. Convert any `pricing` to per-1M rates — or drop it and rely on automatic cost tracking.
6. Move `service.timeout_seconds` to `defaults.call.timeout` (as a duration string), top-level
   `langfuse` under `observability`, and `ALLOWED_CORS_ORIGINS` into `server.cors`.
7. Change `api_mode: responses` to `api: responses` on any model that used it.
8. Validate: `mom config validate <file>`, then `mom config show <file>` to eyeball the resolved
   effort matrix.

**Deployment**

9. Move to Python 3.12+ and install with `uv sync` (or `pip install .`).
10. Replace the `uvicorn mom_service.main:app` command with `mom serve`.
11. Stop the port-8001 reporting service; drop `REPORTING_SERVICE_URL`; point dashboard users at
    `/v1/progress/{request_id}` (and set `server.public_url` if you're behind a proxy).
12. Set `MOM_DATA_DIR` to a persistent directory. Expect a fresh cache and a fresh metrics history;
    archive `mom_service/usage_metrics.db` if you want the old numbers.
13. Drop Redis unless you run multiple workers.
14. Clients need no changes.
