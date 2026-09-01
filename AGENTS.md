# MoM — Agent Guidelines

MoM (Mixture of Models) is a self-hosted, OpenAI/Anthropic-compatible LLM gateway: one request
fans out to a panel of models, and a designated synthesizer consolidates their answers into a
single reply. Package `mom` (src-layout, under `src/mom/`), distribution `mom-llm`, CLI `mom`.

**Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before any non-trivial change.** It is the
authoritative account of the pipeline, the layering, and *why* they are shaped that way. This file
is the short operating manual on top of it.

> **v1 is retired.** There is no `mom_service` package, no `requirements.txt`, and no
> `uvicorn mom_service.main:app`. A reference to any of those is stale — fix it rather than
> following it. See [docs/MIGRATION.md](docs/MIGRATION.md) for what changed.

## Environment

Always work inside the project virtualenv:

```bash
make install              # == uv sync --group dev -> ./.venv (Python 3.12+, resolved from uv.lock)
source .venv/bin/activate
```

- `uv` owns dependencies. Add one with `uv add <pkg>` / `uv add --group dev <pkg>`; never
  `pip install` into `.venv`, and never hand-edit `uv.lock`.
- Prefer the persistent `.venv` over ephemeral `uv run …` invocations — the `make` targets call
  `.venv/bin/*` directly, and CI installs the same lockfile with `uv sync --locked`.
- Secrets live in the environment, or in a `.env` / `auth.json` on the config search path
  (`docs/CONFIGURATION.md`); copy `.env.example`. The YAML config only ever *names* env vars,
  never holds values. Never commit a key, and never read a real `.env` into a test — the suite is
  hermetic against `$HOME` and the working directory as well as the environment.

## Build / Lint / Test commands

| Command | What it does |
| --- | --- |
| `make fmt` | Ruff format + `ruff check --fix` over `src/mom tests` |
| `make fmt-check` | Ruff format check (what CI gates on) |
| `make lint` | `ruff check .` + `lint-imports` (layer contracts) |
| `make typecheck` | `mypy --strict` over `src/mom` |
| `make test` | Full pytest suite |
| `make cov` | Tests + coverage; must stay at or above `fail_under` in `[tool.coverage.report]` |
| `make check` | fmt-check + lint + typecheck + test — run this before handing work back |
| `make run` | `mom serve --reload` on http://127.0.0.1:8000 |

Narrower runs while iterating:

```bash
pytest tests/test_chat_api.py                                   # one file
pytest tests/test_chat_api.py::test_non_streaming_completion     # one test
pytest -k "cache and not metrics" -q                             # by expression
```

The CLI itself is a fast feedback loop: `mom config where` (the search path — what was checked,
found, and in what order it merges), `mom config validate` (exits non-zero on any problem),
`mom config show [ensemble]` (fully-resolved catalog), `mom healthcheck`, `mom cache …`,
`mom metrics …`. Each takes an optional path to pin one file instead of discovering.

For an end-to-end run with no real provider or keys, start the mock upstream, point
`OPENAI_API_BASE` at it, and serve the matching config — every member is `openai/*`, so nothing
leaves the machine:

```bash
python -m tools.mock_openai_server 8899 &
export OPENAI_API_BASE=http://127.0.0.1:8899 OPENAI_API_KEY=mock-key
export MOM_CONFIG=tools/live_config.example.yaml MOM_API_TOKEN=dev-secret
mom serve                 # then POST to /v1/chat/completions with model "mom"
```

## Map of the package

| Package | Holds |
| --- | --- |
| `mom.domain` | Pure value types + pure functions: request IR, `StreamEvent`s, results, cost math, cache key. No I/O, no framework imports. |
| `mom.domain.ports` | The Protocols the engine depends on (`LLMClient`, `Clock`, `IdFactory`, `CacheStore`, `Tracer`) and their DTOs. |
| `mom.engine` | `resolve_plan` (request + catalog → `ExecutionPlan`) and `run_ensemble` / `collect`. |
| `mom.adapters` | Port implementations that touch the world: `LiteLLMClient`, `CachingClient`, tracer, event bus. |
| `mom.config` | Pydantic v2 YAML schema, `extends` resolution, the immutable `ResolvedCatalog`, capability cards. |
| `mom.store` | The two aiosqlite databases: response cache and usage metrics. |
| `mom.api` | FastAPI routers, wire schemas, the three protocol encoders, and wire→IR translation. |
| `mom.runtime` | Composition root: `Settings`, `Container`, `build_container`, logging. |
| `mom.cli` | The `mom` Typer app. |
| `mom.testing` | Shipped test doubles (`FakeLLM`, `ManualClock`, `SequentialIds`, `RecordingTracer`). |

## Invariants — do not break these

1. **One orchestration path.** `run_ensemble` emits a typed event stream; streaming encoders and
   `collect()` + `build_*` both fold *that same stream*. Never add a second path for
   non-streaming — that is exactly the drift v2 was built to eliminate.
2. **`StreamEvent` is a closed union.** Adding a variant must make `mypy --strict` fail in every
   encoder that does not handle it. Handle it everywhere rather than widening a type.
3. **The domain stays pure**, and layers point one way (`cli` → `api` → `runtime`). `lint-imports`
   enforces both contracts; if it fails, the design is wrong, not the linter.
4. **Only `mom.adapters.litellm_client` imports `litellm`.** Everything above it speaks the
   domain's neutral `CallSpec` / `Completion` / `CompletionChunk`.
5. **The cache key is bit-compatible with the v1 golden** (`src/mom/domain/cachekey.py`, pinned by
   `tests/test_cachekey.py` against `tests/golden/cache_keys.json`). Changing its output silently
   invalidates every cached entry and re-spends real money. Deliberate divergences belong in
   `tests/golden/DEVIATIONS.md`.
6. **No import-time side effects.** Building the app must not open sockets, files, or config;
   tests inject a prebuilt `Container` and skip the lifespan entirely.
7. **Nothing blocks the token stream.** Metrics writes go through a bounded queue drained off the
   request path; no blocking call belongs in an async path (Ruff's `ASYNC` rules police this).
8. **No raw provider detail reaches a client.** Upstream failures become a typed `MomError` /
   `UpstreamError` with a safe message; the operator sees the detail in logs only.

## Code style

- **Line length** 100, double quotes, spaces. Ruff is both formatter and linter — do not add Black.
- `from __future__ import annotations` at the top of every module; imports grouped stdlib /
  third-party / first-party (`mom` is first-party, `force-sort-within-sections`).
- **Types everywhere.** `mypy --strict` covers `src/mom`; prefer precise types over `Any`.
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `_private` for internal.
- **Errors**: raise the specific domain error from `mom.domain.errors`; no bare `except`.
- **Logging**: structlog via `get_logger("mom.<area>")`, called with a short event name plus
  key/value kwargs (`logger.info("event name", request_id=…)`) — not `%`-formatting or f-strings.
- **Docstrings**: a module docstring saying what the module is *for*, and one on anything whose
  reason for existing is not obvious from the signature. Explain *why*, not *what*.
- Ruff's rule selection is deliberate: `S` (this is an auth gateway), `ASYNC`, `DTZ` (no naive
  datetimes), `LOG`/`G`, `PT`, `PERF`, `RET`, `SIM`, `ARG`. Max McCabe complexity 10. Fix the code
  rather than adding a `noqa`; if a suppression is genuinely right, comment why.

## Testing conventions

- Tests live in `tests/`, files named `test_*.py`, plain module-level functions (no test classes).
- `asyncio_mode = "auto"` — an `async def test_…` needs no marker.
- The suite is **hermetic**: `pytest-socket` blocks the network except `127.0.0.1`/`::1` (for the
  in-process ASGI transport), and `conftest.py` strips ambient `MOM_*` and provider keys so a local
  `.env` cannot leak in. `pytest-randomly` shuffles order, so tests must not depend on each other.
- Prefer the shipped doubles in `mom.testing` over `unittest.mock` — they implement the real ports,
  so a signature change breaks them loudly.
- Drive endpoints with `create_app()` over `httpx.ASGITransport` (`tests/test_app.py`);
  `tests/test_sdk_openai.py` / `test_sdk_anthropic.py` parse real streams with the official SDKs to
  keep the wire formats honest.
- SQLite stores open under pytest's `tmp_path` (`tests/test_store_metrics.py`).
- Coverage is gated on `mom.domain`, `mom.engine`, `mom.adapters`, `mom.config`.

## When you change behavior

Update the artifacts that describe it, in the same change:

- `CHANGELOG.md` (Keep a Changelog format; the project is on SemVer)
- `docs/` — `ARCHITECTURE.md`, `CONFIGURATION.md`, `API.md`, `PROVIDERS.md`, `MIGRATION.md`
- `config.example.yaml` and `.env.example` for new config or env surface
- `README.md` when the user-visible story changes

There is no docs generator; these are hand-maintained, so drift is on you to prevent.

## Tooling

The MCP servers *you* use to work on this repo, and other agent tooling, are per-developer
environment config and are not committed here — use whatever your environment provides, and fall
back to ripgrep and the docs in `docs/` when it doesn't. Do not add local-tooling references to
committed files. (Distinct from `mom.api.mcp`, which is the product: mom serving *its own* tools
over MCP. That is committed, tested, and documented like any other surface.)
