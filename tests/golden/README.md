# Characterization goldens

These artifacts freeze the **load-bearing behavior of the v1 service** so the MoM v2 rebuild can
be proven equivalent where it matters. They are captured from the current code *before* v2 exists,
and they are the specification the rebuild targets — not the other way around.

Some goldens are intentionally captured over v1 behavior that is *buggy*; where v2 deliberately
diverges, the change is recorded in [`DEVIATIONS.md`](DEVIATIONS.md) and the v2-side test asserts
the new, documented behavior.

## Artifacts

| Golden | Captured from | Pins |
|---|---|---|
| `config_expansion.json` | `mom_service.config.load_config` on `fixtures/live_config_snapshot.yaml` | full base+variant+alias expansion, `api_key_env` inference, service/langfuse blocks — the spec for the v2 loader and the one-time migration script |
| `cache_keys.json` | `mom_service.llm_calls._generate_cache_key` | the exact response-cache SHA256s + invariants (ignore-list params, presigned-URL stripping, key-order canonicalization, alias sensitivity) — the v2 cache-key port must be **bit-compatible** |

`fixtures/live_config_snapshot.yaml` is a frozen copy of the owner's real `config.yaml` at the
start of the rebuild. It contains no secrets (the config uses env-var-name indirection). Goldens
load the snapshot, never the live `config.yaml`, so editing the live config never breaks them.

## Regenerating

Only regenerate on a **deliberate, understood** behavior change (regenerating `cache_keys.json`
invalidates every entry in the on-disk response cache):

```bash
REGEN_GOLDENS=1 .venv/bin/python -m pytest tests/test_golden_config.py tests/test_golden_cache_keys.py
```

Review the diff before committing.
