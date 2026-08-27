# Proposal: expose the gateway over MCP

Status: proposed.

## Why

MoM's only client surface is the model endpoint (OpenAI/Anthropic wire) plus the CLI on the
server host. That leaves two gaps.

1. **The panel as a tool.** An agent already running on some model can't ask an ensemble for a
   second opinion without re-pointing its model endpoint at the gateway mid-session. The fan-out
   is exactly the kind of thing you want available as a tool call ("consult the panel on this
   decision"), not only as a model swap.
2. **Operating the gateway needs a shell.** Spend, cache state, and the catalog are only visible
   through `mom metrics` / `mom cache` / `mom config show` on the host. The one client-side
   control channel is the in-band `<<SYSTEM>>` directive block: useful, but it is control
   smuggled through message content.

One server, both gaps, nothing per-client.

## Desired behaviour

- **Transport**: streamable HTTP mounted on the existing FastAPI app at `/mcp`. Same process,
  port, and bearer auth; the dev `auth: none` opt-out applies here too. Enabled by
  `server.mcp: { enabled: true }`, default off.
- **Tools**, deliberately few:
  - `consult(ensemble, prompt, effort?)`: run the named ensemble, return the synthesized
    answer; `show_work` behaviour follows the ensemble config. Must run through the one
    orchestration path (`run_ensemble` + `collect`), like every other surface.
  - `list_ensembles()`: the resolved catalog with names, descriptions, members, tiers.
  - `usage(window?)`: spend by ensemble/llm, same query the CLI's `metrics usage` runs.
  - `cache_stats()`: entry count, size, hit counts.
- **Read-only except `consult`.** Mutations (purge, config changes) stay CLI-only; a leaked
  token can already chat but must not be able to destroy state.
- **Layering**: the MCP surface is a sibling of the routers in `mom.api` and consumes the
  same typed event stream the encoders do. No second orchestration path, nothing new in the
  domain.

## Non-goals (for now)

- stdio transport. The gateway is a long-running server; HTTP is the natural fit. Revisit if
  a local-only use case shows up.
- MCP *client* support (members calling out to MCP servers). That is a tools-pipeline
  feature, unrelated to this surface.
