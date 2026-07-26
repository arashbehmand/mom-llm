# Live testing

Two ways to exercise the running server end to end.

## 1. Offline smoke (no keys, no cost)

Drives the full stack — uvicorn → plan resolution → LiteLLM → HTTP → fan-out synthesis → SSE →
SQLite persistence + metrics — against a local mock provider.

```bash
# terminal 1: the mock OpenAI endpoint
uv run python -m tools.mock_openai_server 9099

# terminal 2: the gateway, pointed at the mock
export MOM_CONFIG=$PWD/tools/live_config.example.yaml
export MOM_API_TOKEN=livetest MOM_DATA_DIR=/tmp/mom-live
export OPENAI_API_BASE=http://127.0.0.1:9099/v1 OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP=true
uv run mom serve --host 127.0.0.1 --port 8099

# terminal 3: drive it
TOKEN="Authorization: Bearer livetest"
curl -s http://127.0.0.1:8099/health
curl -s -H "$TOKEN" http://127.0.0.1:8099/v1/models
curl -s -H "$TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"mom","messages":[{"role":"user","content":"hi"}]}' \
  http://127.0.0.1:8099/v1/chat/completions
curl -sN -H "$TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"mom","messages":[{"role":"user","content":"hi"}],"stream":true}' \
  http://127.0.0.1:8099/v1/chat/completions
curl -s -H "$TOKEN" http://127.0.0.1:8099/v1/metrics/usage
```

Expect a fan-out (both members shown in the `<think>` block), a synthesized answer, an SSE stream
whose first delta carries the assistant role, and `/v1/metrics/usage` reporting the recorded calls.

## 2. Real providers + Langfuse

Uses your real config and keys, and lights up Langfuse tracing.

1. Migrate your v1 config once: `uv run python -m tools.migrate_v1_config config.yaml -o models.yaml`.
2. Set `observability.langfuse.enabled: true` in the config, and put the provider keys +
   `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` + `MOM_API_TOKEN` in `.env`.
3. `MOM_CONFIG=$PWD/models.yaml uv run mom serve` and drive it with the same curls, choosing a
   cheap ensemble (e.g. `mom-cheap`/`mom-mini`) for the first run.
4. Each request appears in Langfuse as a trace with one generation per member and the synthesis,
   and `/v1/metrics/usage` shows tokens + cost, incl. `cached_prompt_tokens` on repeat turns.
