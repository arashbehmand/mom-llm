# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-09-06

Shape a panel without editing its roster — per turn from the chat box, per machine from a config
layer — plus a model list that says what each ensemble contains, and a progress link that no longer
hands out the gateway credential.

### Added

- **Config discovery — a search path, not one env var.** `mom` now finds its own config. Two
  levels, deep-merged like git config: a user level (`~/.mom/config.yaml`, else
  `$XDG_CONFIG_HOME/mom/config.yaml`) holding the models and keys a machine has once, and a
  project level (`./mom.yaml`, else `./.mom/config.yaml`) holding only what a directory adds.
  Each layers a sibling override named from its own file (`mom.yaml` → `mom.override.yaml`), and
  `MOM_CONFIG_OVERLAY` still merges last. Validation runs **once, after the merge**, so a project
  file can be nothing but `version: 2` and the `ensembles:` it adds over user-level `llms`; `null`
  masks an inherited key. Deliberately not `./config.yaml` — too generic a name to claim in an
  arbitrary directory — and deliberately no upward walk from the working directory.

  This exists because `mom mcp` made the old model untenable: MCP clients launch it from an
  arbitrary directory with a near-empty environment, so every client entry had to carry
  `--config /abs/path` and its own `env` block, and `mom cache` / `mom metrics` answered
  differently depending on where you ran them. An MCP entry is now
  `{"command": "mom", "args": ["mcp"]}`.

  `--config` / `MOM_CONFIG` still works and now means something stronger: it **pins** one file and
  turns discovery off entirely, so a server told which config to serve does not also pick up
  whatever sits in `$HOME`. `MOM_CONFIG` is read from the process environment and the working
  directory's `.env` only, never from a discovered one — a file mom has not found yet must not be
  able to change where mom looks.

- **Secrets on the same search path, and `auth.json`.** `.env` and `auth.json` are read from each
  level's directory and always from the working directory, project before user, first definition
  wins, with the process environment outranking every file. `auth.json` is a flat
  env-var-name → value object (`{"ANTHROPIC_API_KEY": "sk-…"}`), the same vocabulary the config
  already uses when it names an `api_key_env`; mom warns when one is readable beyond its owner,
  and treats a malformed one as a skip rather than a startup failure.

  This also fixes something that only ever worked by accident: nothing in mom put provider keys
  into the environment, so a key in `.env` reached `os.getenv` only because litellm calls
  `load_dotenv()` on import — and that resolves relative to *litellm's own installed directory*,
  not the working directory. It found a repo's `.env` when `.venv/` happened to sit inside the
  repo, and nothing at all from a system-wide install. Keys are now published deliberately.
  `MOM_*` names are deliberately excluded from that: they reach `Settings` through its dotenv
  source instead, so `MOM_API_TOKEN` is not visible to every subprocess mom spawns.

  An **empty value is treated as absent everywhere** — it is not a contribution, it does not
  shadow a real value further down the path, and it does not prevent a file from replacing an
  empty variable already in the environment. Any other rule disagrees with the code that reads
  these names, where an empty API key is indistinguishable from a missing one. That covers mom's
  own settings too: `Settings` is handed the values the files defined rather than the file paths,
  because handing over paths let pydantic re-read them raw, so a bare `MOM_API_TOKEN=` in a
  project `.env` beat a real token at the user level and left the gateway unable to authenticate
  anyone. Keeping settings out of the process environment likewise covers the legacy spellings
  (`API_TOKEN`, `REDIS_URL`, `LITELLM_VERBOSE`), which carry the same secrets as their prefixed
  names and were being published while the prefixed ones were not.

- **`--auth-from-opencode`.** Borrows API keys from [opencode](https://github.com/sst/opencode)'s
  `auth.json` (`$XDG_DATA_HOME/opencode/auth.json`, else `~/.local/share/opencode/auth.json`),
  mapping its `type: "api"` entries onto the standard env var names at the lowest precedence of
  all. `oauth` entries are skipped — they hold refresh tokens for a session opencode renews, not
  API keys, and a provider answers one with an opaque 401. The file is ignored entirely unless the
  flag (or `MOM_AUTH_FROM_OPENCODE`) is set.

- **`mom config where`.** Prints every path checked, which were found, the exact merge order, and
  the secret files consulted — env var **names** only, never values. It never applies the secrets
  it describes, so asking where a key would come from cannot change where it comes from, and it
  answers even when the merged config is broken or missing, which is when it is most needed.
  Each secret file reports in three classes, because "what did this file do" has three different
  answers: `would set:` (names it publishes to the environment), `reaches settings:` (`MOM_*`
  names, which configure mom without entering the environment), and `already set elsewhere:`
  (names a higher-precedence source had already defined). Reporting only the first would have
  made a `~/.mom/.env` holding just `MOM_API_TOKEN` — the file authenticating the gateway — read
  as having contributed nothing.

- **`mom serve` takes `--config` / `--overlay` / `--auth-from-opencode`**, and the positional path
  on `mom config validate` / `mom config show` is now optional (omitted means "discover"; given, it
  pins). Every existing invocation keeps working. `mom config show <ensemble>` works too — a lone
  argument that is not a file on disk is read as an ensemble name.

- **MCP surface — the panel as a tool call.** An agent can now ask an ensemble for a second opinion
  without re-pointing its model endpoint mid-session, assemble a panel from the catalog for a single
  question, and read gateway state without a shell on the host. Two transports over one definition:
  streamable HTTP at `/mcp` (same process, port, and bearer auth as `/v1`; enabled with
  `server.mcp: { enabled: true }`, **off by default**, and 404 while disabled), and `mom mcp` over
  stdio for local clients — same config and data dir, no running gateway needed.

  Six tools: `consult` runs a configured `ensemble` *or* an inline `panel` of catalog llms plus a
  `synthesizer`, reporting progress as each member is asked and again as it answers (with the
  running cost), and returning the synthesized answer with a per-member cost breakdown and a
  `/v1/progress/{id}` link. `list_llms`, `list_ensembles`, `runs`, `usage`, and `cache_stats` are
  read-only views. There is no purge or config-mutation tool on either transport: a leaked token
  can already chat, and must not be able to destroy state.

  `consult` returns one envelope for every outcome, discriminated by `status`: `ok` with the answer,
  `tool_calls` when the panel returns calls to execute (keeping any prose the model wrote alongside
  them), or `failed` with a client-safe `{code, message, http_status}` — and, in that last case, the
  members that did complete along with the tokens and money they spent, so a run that dies upstream
  still accounts for what it cost. Inline panels are call-scoped (nothing is written to config),
  bill under the ensemble name `mcp:adhoc` (in config's reserved-character namespace, so they can
  never shadow a configured ensemble), and never coalesce, since request identity keys on the
  ensemble name rather than the roster. `effort` is rejected on an ensemble that declares no tiers
  rather than quietly ignored, so an agent never believes it bought reasoning it didn't get.

  `runs` reports live runs, ones this process just finished, and the recent ledger. The middle
  list matters because metrics are written by a queue drained off the hot path: without it, a run
  that had just returned looked like it had never happened.

  Everything runs through the one orchestration path (`resolve_plan` → `run_ensemble` → `collect`);
  the MCP surface is a fourth fold over the same typed event stream the wire encoders fold.

- **Published container image** at `ghcr.io/arashbehmand/mom-llm`, built for `linux/amd64` and
  `linux/arm64` and tagged on release with the full version, `major.minor`, `major`, and `latest`.
- **PyPI releases** — `pip install mom-llm`, published from the release workflow via PyPI trusted
  publishing (OIDC, no stored API token).
- **Request lifecycle logging at the default level.** A request now narrates itself in `docker
  logs`: the fan-out roster, a line per member as its call goes out, a line per member as it lands
  (status, cache hit, duration, tokens, cost, attempts), synthesis start, and a closing summary
  with totals and elapsed time. Every line carries `request_id`, and the lines come from the
  engine, so all three API surfaces and both streaming and non-streaming behave alike. Model
  output, prompts, and tool arguments are never logged.

- **`<<SYSTEM>> include:` — add a model to one turn's panel.** The counterpart to `only:` and
  `exclude:`, which could only ever subtract. `include: g31p` seats a member the ensemble skips at
  this effort tier (at its llm's own params, since `skip` is a roster instruction and not a
  provider effort value), or any `llm` in the catalog that isn't a member at all, under its own
  name. Applied after the other two, so `include:` wins over `exclude:` on the same name and
  `only: fast` + `include: k3` builds a panel from scratch. Names already on the panel are no-ops,
  never a second seat.

- **A model's description now names the panel behind it.** `emom` in a model picker told a human
  nothing about what answers it. Every discovery surface with a description field — `/v1/models`
  (OpenAI and Anthropic dialects), `/v1/models/{id}`, `/v1/model/info` — now returns the ensemble's
  configured `description:` followed by the models themselves: *"Fans out to 13 models — gpt-5.6-sol,
  gpt-6-astra, claude-opus-5, … , +1 more — then synthesizes with claude-sonnet-5."* Names are
  deduplicated (one llm seated twice is one model) and capped at twelve spelled out with the rest
  counted, so a `members: all` panel stays readable; an ensemble with no configured description
  still gets the panel line, and a `passthrough` one reports the single model that answers. The
  `mom` vendor block gained `member_models`, `synthesizer_model` and `strategy` alongside the
  identities it already carried. `description` is not part of Anthropic's model object — it rides
  along for the clients that render one, and `display_name` stays the id.

- **`members_exclude` / `members_include` — shape a panel from a layer that didn't author it.**
  Config layering deep-merges maps, but a **list** replaces wholesale, and `ensembles.<name>.
  members` is a list: an override that wanted a panel minus one model had to restate the entire
  roster, which then silently stops tracking the roster it was copied from. That is why models
  kept getting dropped from tracked ensembles for machine-local reasons. These two keys patch
  whatever roster the merge produced, so the tracked config keeps the full panel and the
  gitignored override next to it decides what this machine runs:

  ```yaml
  # ~/.mom/config.override.yaml
  ensembles:
    emom:
      members_exclude: [fable, astra]
  ```

  `members_exclude` takes identities (a bare name, or a list); `members_include` takes members in
  the same shape `members:` does, and one whose identity is already seated is redeclared **in
  place** rather than seated twice — which is also how a layer retunes a single member's effort.
  Inclusions apply last, so `members_include` wins over `members_exclude` on the same name, the
  same rule the per-turn `<<SYSTEM>> include:` directive follows. Excluding a name that isn't on
  the roster is a deliberate no-op: the exclusion outlives edits to the config it patches, and a
  base config dropping that model on its own must not take the gateway down. Including an llm that
  doesn't exist is still an error, as is an exclusion that empties the panel.

### Changed

- **A `<<SYSTEM>>` directive MoM can't honor no longer costs you the turn.** An unknown member
  name, an unknown `synth:` target, a value outside a directive's vocabulary (`show_work:
  verbose`, `dedupe: maybe`), `only:`/`exclude:`/`include:` on a passthrough ensemble, a mistyped
  key — each used to be a 400 that threw away the whole request. Each is now ignored, with the
  reason carried on `ExecutionPlan.notices` and rendered at the top of the think block: as the
  `<think>` preamble on `/v1/chat/completions`, in the leading reasoning item on `/v1/responses`,
  as a `thinking` block on `/v1/messages` (which has no member-dump convention to hang it on), in
  `notices[]` on an MCP `consult` result — and in the logs. It renders whatever `show_work` says,
  since a panel that quietly ran the wrong roster is exactly what this was built to prevent, and
  near misses name the likely culprit (`Did you mean 'k3'?`).

  These names get typed from memory into a chat box, and one wrong character shouldn't cost a
  panel run. The one case still fatal is a selection with nothing left to run — an `only:` that
  matches nothing, or an `exclude:` below the ensemble's quorum — which stays a pre-flight 400
  before any fan-out spend, since there's no answer to be had either way.

  A mistyped *key* now ends the directive header rather than failing: that line and everything
  after it become the instruction verbatim (with a warning naming the key), so nothing typed is
  ever dropped. `<<SYSTEM>>Note: be careful<</SYSTEM>>` therefore works as prose now, where it
  used to 400.

### Fixed

- **`null` in `MOM_CONFIG_OVERLAY` now deletes an inherited key**, which is what
  `docs/CONFIGURATION.md` has always said it does and what `extends:` has always done. There were
  two deep-merge implementations that disagreed on exactly this; there is now one
  (`mom.config.merge`), used by config layering, `extends`/`variants`, and discovery alike.

- **Config is resolved once, in one place.** It used to be loaded at four independent sites with
  three different policies, and two of them silently dropped `MOM_CONFIG_OVERLAY`: the CORS
  bootstrap in `create_app` and the data-dir resolution behind `mom cache` / `mom metrics`. An
  overlay that set `server.cors` was invisible to CORS, and one that set `storage.data_dir` sent
  `mom cache stats` looking in a different directory than the gateway was writing to. Everything
  now goes through `mom.runtime.bootstrap`.

- **A config you name is a config you get.** `mom cache` / `mom metrics` resolve their data
  directory through the shared resolver, and that resolver used to swallow every config error and
  fall back to the platform default. A typo in `MOM_CONFIG` therefore retargeted the command
  silently — `mom cache purge --yes` would have purged a cache the operator never named. Finding
  *nothing* still falls back quietly (these commands answered without a config before discovery
  existed); a file that was named and could not be loaded now fails.

- **A supplied catalog turns discovery off.** `create_app(catalog=…)` without settings paired the
  caller's catalog with settings discovered from the host's `~/.mom`, so an embedder that had
  taken over config resolution still had its data directory, API token and Redis URL decided by
  whatever happened to be in `$HOME`. Its process environment configures it now, and nothing else.

- **`create_app` honours the `Settings` it is handed.** Passing `Settings(config_file=X)` without
  a catalog — the library-embedder and `uvicorn …:create_app --factory` path — re-derived the pin
  from the environment and served whatever discovery turned up, while still reporting `X` as
  `container.settings.config_file`. It now serves `X`, and adopts the bootstrapped settings so a
  `MOM_API_TOKEN` in a discovered `.env` authenticates on that path too.

- **`settings.config_file` is only ever the pin.** A discovered `.env` carrying `MOM_CONFIG` is
  correctly ignored for discovery, but still bound into `Settings` — leaving `config_file` naming
  a file that was never loaded while `sources.files` held the merge that actually ran. It is now
  cleared explicitly when discovery did the work.

- **An `MOM_CONFIG_OVERLAY` that names a discovered file keeps its last place.** De-duplication
  kept the first occurrence, which is right for the secret directories (ordered
  highest-precedence-first) and backwards for the config stack (ordered lowest-first): the file
  the operator asked to apply last was demoted, and an intervening layer overrode it.

- **Building the app reads no files.** `create_app` used to re-load `settings.config_file` in its
  body to install CORS — a second read that could disagree with the catalog the lifespan went on
  to serve. It is now handed the resolved catalog. `mom serve` points uvicorn at the new
  `mom.api.app:serve_app`, which resolves config and secrets in the process that serves (the
  child, under `--reload`). `uvicorn mom.api.app:create_app --factory` still works and still
  discovers its config, but no longer installs CORS from it — see `docs/MIGRATION.md`.

- **`MOM_LOG_LEVEL` and `MOM_LOG_FORMAT` had no effect.** `configure_logging` was never called, so
  the gateway ran on structlog's defaults and `MOM_LOG_FORMAT=json` was unreachable. It is now
  invoked from the app lifespan.
- **Durations recorded as zero.** Synthesis was recorded as `duration_ms=0.0` in both metrics and
  tracing, and a member that timed out or errored — or a synthesis call that failed outright —
  recorded no duration at all, zero on exactly the outcomes worth timing. All now record real
  elapsed time.
- **Detached members were invisible in the log.** A member left running after a client disconnect
  finishes and records a metric for its spend, but nothing was logged, so the logs under-reported
  cost relative to the metrics DB. It now logs `detached member completed` when it lands.

### Security

- **The progress link no longer carries your API token.** A browser opening a link cannot attach an
  `Authorization` header, so something has to ride in the query string — and what MoM put there was
  the gateway credential, in a URL it then printed into the `X-MoM-Progress-Url` header, the
  think-block `Progress:` line, and every transcript a client saved. It is now a **link token**:
  `HMAC(api_token, request_id)`, minted per request and worth exactly one run's progress feed.
  Forging one needs the API token; leaking one costs a progress page. Nothing to configure and
  nothing extra to rotate — the key is the API token itself, so rotating it invalidates every
  outstanding link. The endpoint still accepts the API token by header or `?token=`, so a client
  that holds it can still watch any run and older links keep opening.

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
