# Intentional v2 deviations from v1 behavior

Each row is a place where MoM v2 **deliberately** differs from the frozen v1 goldens. The v2-side
test asserts the *new* behavior and links back here; the golden diff is an annotated changelog, not
a regression. (Populated as v2 phases land; captured here up front so nothing drifts silently.)

| Area | v1 (captured) | v2 (intended) | Why |
|---|---|---|---|
| Cache key | includes `stream` / `_api_route` in the hashed input | excludes `stream`/`stream_options` | streaming and non-streaming responses should share a cache entry |
| Fan-out ordering in synthesis | candidate blocks emitted in completion order (nondeterministic) | ordered by ensemble-config member order | deterministic synthesis prompt → provider prompt-cache hits + stable SSE goldens |
| Streaming terminal chunk | provider `finish_reason` relied upon; usage chunk dropped | guaranteed terminal `finish_reason`; usage chunk emitted when `stream_options.include_usage` | spec-correct OpenAI streaming; agents read usage |
| First streaming delta | no `role` delta | first delta carries `{"role":"assistant"}` | spec-correct; some clients require it |
| Errors mid-stream | `str(exc)` leaked in-band | typed `MomError.safe_message`; always followed by `[DONE]` | no internal detail leakage; clean termination |
| Effort params | silently dropped (request model didn't accept them) | mapped to a per-member effort tier | "respect modern reasoning params" |
| Cached tokens | billed at full input rate | priced at provider cache read/write rates | correctness (live v1 cost bug) |

> Note: the config-expansion golden is expected to reproduce **exactly** under the v2 loader (the
> config format is redesigned, but `scripts/migrate_v1_config.py` must yield a v2 config whose
> resolved projection is semantically equal to this golden).
