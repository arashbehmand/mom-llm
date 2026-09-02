"""``mom`` command-line interface.

Thin Typer app. Subcommands delegate to the library; nothing heavy runs at import.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
import urllib.error
import urllib.request

import typer

from mom import __version__


if TYPE_CHECKING:  # imports stay lazy at runtime — nothing heavy runs on `mom --help`
    from mom.config.resolve import ResolvedCatalog
    from mom.runtime.bootstrap import Bootstrapped


app = typer.Typer(
    name="mom",
    help="MoM — a Mixture-of-Models OpenAI/Anthropic-compatible LLM gateway.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """MoM command-line interface."""


# One declaration per option, shared by every command that takes it. These used to be four
# near-identical sets with drifting help text, which is how `mom serve` ended up with no
# `--config` at all and `mom cache`'s silently ignoring `MOM_CONFIG_OVERLAY`.
_ConfigOpt = typer.Option(
    None,
    "--config",
    "-c",
    exists=True,
    dir_okay=False,
    help="Config YAML to use. Pins resolution: discovery is skipped entirely (else MOM_CONFIG).",
)
_OverlayOpt = typer.Option(
    None,
    "--overlay",
    exists=True,
    dir_okay=False,
    help="Extra YAML deep-merged last, over everything else (else MOM_CONFIG_OVERLAY).",
)
_DataDirOpt = typer.Option(
    None, "--data-dir", help="Data directory for cache.db/metrics.db (overrides config/env)."
)
_OpencodeOpt = typer.Option(
    False,
    "--auth-from-opencode",
    help="Also read API keys from opencode's auth.json, at lowest precedence.",
)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Dev auto-reload (never in prod)."),
    config: Path | None = _ConfigOpt,
    overlay: Path | None = _OverlayOpt,
    auth_from_opencode: bool = _OpencodeOpt,
) -> None:
    """Run the MoM server."""
    import uvicorn

    # uvicorn imports the factory and calls it with no arguments, in a *child* process under
    # --reload. Flags therefore travel by environment or not at all — the same ordering problem
    # `mom/__init__.py` solves for LITELLM_LOCAL_MODEL_COST_MAP. Export the raw flags, never the
    # resolved file list: exporting the result would pin the child to one file and collapse the
    # merge discovery just performed. Resolve to absolute, because the child's working directory
    # is not guaranteed to be this one.
    if config is not None:
        os.environ["MOM_CONFIG"] = str(config.resolve())
    if overlay is not None:
        os.environ["MOM_CONFIG_OVERLAY"] = str(overlay.resolve())
    if auth_from_opencode:
        os.environ["MOM_AUTH_FROM_OPENCODE"] = "1"

    # Not `create_app`: `serve_app` resolves config and secrets in the process that serves. Doing
    # it here in the parent would hand the child a pre-populated environment, turning its own
    # first-definition-wins pass into a no-op and logging every warning twice.
    uvicorn.run(
        "mom.api.app:serve_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def mcp(
    config: Path | None = _ConfigOpt,
    data_dir: Path | None = _DataDirOpt,
    overlay: Path | None = _OverlayOpt,
    auth_from_opencode: bool = _OpencodeOpt,
) -> None:
    """Serve the MoM tools over MCP stdio (for a local MCP client; no running gateway needed).

    Consults run here are recorded to the same metrics.db and warm the same cache as the
    gateway's, so `mom metrics usage` accounts for them too. Unlike the HTTP surface at /mcp,
    this ignores `server.mcp.enabled` — running the command is the opt-in.
    """
    from mom.api.mcp.stdio import run_stdio

    asyncio.run(
        run_stdio(
            config=config,
            data_dir=data_dir,
            overlay=overlay,
            auth_from_opencode=auth_from_opencode,
        )
    )


@app.command()
def healthcheck(
    url: str = typer.Option("http://127.0.0.1:8000/health", help="Health endpoint to probe."),
    timeout: float = typer.Option(3.0, help="Timeout in seconds."),
) -> None:
    """Probe the server's health endpoint (stdlib only — safe as a Docker HEALTHCHECK)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed localhost)
            if resp.status == 200:
                raise typer.Exit(0)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        typer.echo(f"health check failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo("health check failed: non-200 response", err=True)
    raise typer.Exit(1)


config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

# Optional: omitted means "resolve the search path", which is the whole point of discovery.
# Given, it pins — so every `mom config validate <path>` in a script or a doc keeps working.
_PathArg = typer.Argument(
    None, exists=True, dir_okay=False, help="Config YAML to pin (default: discover)."
)


def _resolve_catalog(
    path: Path | None, overlay: Path | None, *, auth_from_opencode: bool = False
) -> ResolvedCatalog:
    """Resolve a catalog the way every entry point does, turning failure into a clean exit."""
    from mom.config.resolve import ConfigError
    from mom.runtime.bootstrap import bootstrap

    try:
        return bootstrap(
            config=path, overlay=overlay, auth_from_opencode=auth_from_opencode, apply=False
        ).catalog()
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.secho(f"invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@config_app.command("validate")
def config_validate(
    path: Path | None = _PathArg,
    overlay: Path | None = _OverlayOpt,
    auth_from_opencode: bool = _OpencodeOpt,
) -> None:
    """Load, validate, and resolve a config; exit non-zero on any problem."""
    catalog = _resolve_catalog(path, overlay, auth_from_opencode=auth_from_opencode)
    typer.secho(
        f"OK — {len(catalog.llms)} llms, {len(catalog.ensembles)} ensembles",
        fg=typer.colors.GREEN,
    )


_ShowPathArg = typer.Argument(
    None, dir_okay=False, help="Config YAML to pin, or an ensemble name (default: discover)."
)


@config_app.command("show")
def config_show(
    path: Path | None = _ShowPathArg,
    ensemble: str | None = typer.Argument(None, help="Show just this ensemble."),
    overlay: Path | None = _OverlayOpt,
    auth_from_opencode: bool = _OpencodeOpt,
) -> None:
    """Print the fully-resolved catalog (flattened effort matrix per ensemble)."""
    from mom.config.resolve import ResolvedEnsemble

    # Both positionals are optional now, which makes a lone `mom config show bmom` ambiguous
    # with `mom config show ./mom.yaml`. Resolve it by looking: an argument that is not a file
    # on disk was meant as an ensemble name. That keeps `show <path> <ensemble>` (docs, tests),
    # `show <path>`, `show <ensemble>` and bare `show` all working.
    if ensemble is None and path is not None and not path.is_file():
        path, ensemble = None, str(path)
    if path is not None and not path.is_file():
        typer.secho(f"no such file: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    catalog = _resolve_catalog(path, overlay, auth_from_opencode=auth_from_opencode)

    def render(name: str, ens: ResolvedEnsemble) -> None:
        tiers = ens.effort_tiers
        tier_labels = "/".join(t.label for t in tiers) if tiers else "(none)"
        typer.secho(f"\nensemble {name}", bold=True)
        typer.echo(
            f"  strategy={ens.strategy}  "
            f"default_tier={ens.default_tier.label if ens.default_tier else '-'}  "
            f"show_work={ens.show_work}  tiers={tier_labels}"
        )
        typer.echo("  members:")
        for member in ens.members:
            llm = catalog.llms[member.llm]
            effort = "/".join(member.effort_by_tier[t] for t in tiers) if tiers else "(llm params)"
            typer.echo(f"    {member.identity:<14} {llm.model:<34} effort: {effort}")
        syn = ens.synthesizer
        syn_llm = catalog.llms[syn.llm]
        syn_effort = "/".join(syn.effort_by_tier[t] for t in tiers) if tiers else "(llm params)"
        typer.echo(
            f"  synthesizer: {syn.llm:<12} {syn_llm.model:<34} effort: {syn_effort}"
            f"  prompt: {syn.prompt or '-'}"
        )

    if ensemble is not None:
        target = catalog.ensembles.get(ensemble)
        if target is None:
            typer.secho(f"no such ensemble: {ensemble}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        render(ensemble, target)
    else:
        typer.echo(f"{len(catalog.llms)} llms, {len(catalog.ensembles)} ensembles")
        for name, ens in catalog.ensembles.items():
            render(name, ens)


@config_app.command("where")
def config_where(
    path: Path | None = _PathArg,
    overlay: Path | None = _OverlayOpt,
    auth_from_opencode: bool = _OpencodeOpt,
) -> None:
    """Show what was checked, what was found, and in what order it merges.

    Reports without resolving: a search path is hardest to reason about exactly when the config
    it produced is broken or missing, so this must still answer then. It also never applies the
    secrets it describes — asking where a key would come from must not change where it comes
    from — and prints env var *names* only, never values.
    """
    from mom.runtime.bootstrap import bootstrap

    booted = bootstrap(
        config=path, overlay=overlay, auth_from_opencode=auth_from_opencode, apply=False
    )
    _render_config_sources(booted)
    _render_secret_sources(booted)
    typer.echo(f"\ndata dir: {_data_dir_from(booted)}")


def _render_config_sources(booted: Bootstrapped) -> None:
    sources = booted.sources
    typer.secho("config", bold=True)
    typer.echo(f"  mode: {'pinned (discovery off)' if sources.pinned else 'discovery'}")
    for note in sources.notes:
        typer.echo(f"  note: {note}")
    for probe in sources.checked:
        mark = "found" if probe.found else (probe.note or "not found")
        typer.echo(f"    {probe.path!s:<52} {probe.role:<16} {mark}")
    typer.echo("\n  merge order (low -> high precedence):")
    for i, file in enumerate(sources.files, 1):
        typer.echo(f"    {i}. {file}")
    if not sources.files:
        typer.secho("    (nothing — mom has no config to serve)", fg=typer.colors.YELLOW)


def _render_secret_sources(booted: Bootstrapped) -> None:
    """Every class of contribution, because `applied` alone cannot express two of them.

    `MOM_*` names never enter `os.environ` (they reach `Settings` via its dotenv source), and a
    name a higher-precedence source already defined is beaten rather than absent. Reporting only
    `applied` made a `~/.mom/.env` holding just `MOM_API_TOKEN` — the file authenticating the
    gateway — read as "nothing new".
    """
    typer.secho("\nsecrets  (env var NAMES only — values are never printed)", bold=True)
    for source in booted.secrets:
        if not source.found:
            note = f"not found{'' if source.warning is None else f' ({source.warning})'}"
            typer.echo(f"    {source.path!s:<52} {source.kind:<10} {note}")
            continue
        typer.echo(f"    {source.path!s:<52} {source.kind:<10}")
        for label, names in (
            ("would set", source.applied),
            ("reaches settings", source.settings_names),
            ("already set elsewhere", source.shadowed),
        ):
            if names:
                typer.echo(f"      {label}: {', '.join(names)}")
        if not (source.applied or source.settings_names or source.shadowed):
            typer.echo("      contributes nothing")
        if source.warning:
            typer.secho(f"      ⚠ {source.warning}", fg=typer.colors.YELLOW)
    if not booted.settings.auth_from_opencode:
        typer.echo(
            f"    {'opencode bridge':<52} {'opencode':<10} not checked (pass --auth-from-opencode)"
        )
    typer.echo("\n  the process environment outranks every file above.")


cache_app = typer.Typer(help="Inspect and manage the response cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


def _resolve_data_dir(
    config: Path | None, data_dir: Path | None, *, overlay: Path | None = None
) -> Path:
    """Resolve the data directory exactly as the server does (explicit flag wins).

    Goes through the same resolver as `mom serve`, so `mom cache stats` and the gateway cannot
    disagree about which database they mean — they used to, because this read one file and
    ignored `MOM_CONFIG_OVERLAY` while the server merged the overlay in.

    Finding **nothing** is not fatal: these commands only ever wanted a directory, and they
    answered without a config before discovery existed. A config that was *named* and could not
    be loaded is a different matter — falling back there would silently retarget the command at
    the default database, and `mom cache purge --yes` with a typo in `MOM_CONFIG` would then
    purge a cache the operator never asked about.
    """
    if data_dir is not None:
        return data_dir
    from mom.runtime.bootstrap import bootstrap

    return _data_dir_from(bootstrap(config=config, overlay=overlay, apply=False))


def _data_dir_from(booted: Bootstrapped) -> Path:
    """The data dir implied by an already-resolved bootstrap."""
    from mom.config.resolve import ConfigError
    from mom.runtime.wiring import resolve_data_dir

    if booted.settings.data_dir is not None:
        return Path(booted.settings.data_dir)
    try:
        return resolve_data_dir(booted.settings, booted.catalog())
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        if booted.sources.files:  # something was named or found, and it did not load
            typer.secho(f"invalid config: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        import platformdirs

        return Path(platformdirs.user_data_dir("mom-llm"))


async def _cache_stats(db: Path) -> dict[str, int]:
    from mom.store.cache import SqliteCacheStore

    store = await SqliteCacheStore.open(db, ttl_seconds=0.0, max_bytes=1 << 62)
    try:
        return await store.stats()
    finally:
        await store.close()


async def _cache_clear(db: Path) -> int:
    from mom.store.cache import SqliteCacheStore

    store = await SqliteCacheStore.open(db, ttl_seconds=0.0, max_bytes=1 << 62)
    try:
        return await store.clear()
    finally:
        await store.close()


@cache_app.command("stats")
def cache_stats(
    config: Path | None = _ConfigOpt,
    data_dir: Path | None = _DataDirOpt,
    overlay: Path | None = _OverlayOpt,
) -> None:
    """Print response-cache statistics (entries / bytes / hits)."""
    db = _resolve_data_dir(config, data_dir, overlay=overlay) / "cache.db"
    if not db.exists():
        typer.echo(f"cache: empty (no database at {db})")
        return
    stats = asyncio.run(_cache_stats(db))
    typer.echo(f"path:    {db}")
    typer.echo(f"entries: {stats.get('entries', 0)}")
    typer.echo(f"bytes:   {stats.get('bytes', 0)}")
    typer.echo(f"hits:    {stats.get('hits', 0)}")


@cache_app.command("purge")
def cache_purge(
    config: Path | None = _ConfigOpt,
    data_dir: Path | None = _DataDirOpt,
    overlay: Path | None = _OverlayOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete every entry in the response cache."""
    db = _resolve_data_dir(config, data_dir, overlay=overlay) / "cache.db"
    if not db.exists():
        typer.echo(f"cache: empty (no database at {db})")
        return
    if not yes:
        typer.confirm(f"Purge all cached entries at {db}?", abort=True)
    removed = asyncio.run(_cache_clear(db))
    typer.secho(
        f"purged {removed} cache entr{'y' if removed == 1 else 'ies'}", fg=typer.colors.GREEN
    )


metrics_app = typer.Typer(help="Inspect recorded call metrics (usage/cost).", no_args_is_help=True)
app.add_typer(metrics_app, name="metrics")


async def _usage_report(
    db: Path, *, start: float | None, ensemble: str | None, by: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, float]:
    from mom.store.metrics import MetricsStore

    store = await MetricsStore.open(db)
    try:
        agg = await store.aggregate(start=start, ensemble=ensemble)
        groups = await store.aggregate_by(by, start=start, ensemble=ensemble) if by else None
        savings = await store.estimated_cache_savings(start=start, ensemble=ensemble)
        return agg, groups, savings
    finally:
        await store.close()


@metrics_app.command("usage")
def metrics_usage(
    config: Path | None = _ConfigOpt,
    data_dir: Path | None = _DataDirOpt,
    overlay: Path | None = _OverlayOpt,
    days: float = typer.Option(7.0, help="Look back this many days (0 or negative = all time)."),
    ensemble: str | None = typer.Option(None, help="Restrict to one ensemble."),
    by: str | None = typer.Option(
        None, "--by", help="Also group by: day, member, ensemble, status, or turn_type."
    ),
) -> None:
    """Print usage/cost: calls, billable calls, cache hit rate, per-status breakdown, and
    estimated cache savings. NOTE: `MetricsRecorder` drops rows under sustained load (see
    `GET /health`'s `metrics_dropped`) — treat this as a lower bound on real spend, not exact."""
    db = _resolve_data_dir(config, data_dir, overlay=overlay) / "metrics.db"
    if not db.exists():
        typer.echo(f"metrics: empty (no database at {db})")
        return
    start = time.time() - days * 86400 if days > 0 else None
    try:
        agg, groups, savings = asyncio.run(_usage_report(db, start=start, ensemble=ensemble, by=by))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"path:    {db}")
    typer.echo(f"window:  last {days:g} days" if start is not None else "window:  all time")
    if ensemble:
        typer.echo(f"ensemble: {ensemble}")
    calls = int(agg.get("calls", 0) or 0)
    billable = int(agg.get("billable_calls", 0) or 0)
    cache_hits = int(agg.get("cache_hits", 0) or 0)
    hit_rate = (cache_hits / calls * 100) if calls else 0.0
    cost = float(agg.get("cost_usd", 0.0) or 0.0)
    typer.echo(f"calls:        {calls}  (billable: {billable})")
    typer.echo(f"cache hits:   {cache_hits}  ({hit_rate:.1f}% hit rate)")
    typer.echo(f"cost:         ${cost:.4f}")
    typer.echo(f"  errors:     {int(agg.get('errors', 0) or 0)}")
    typer.echo(f"  empty:      {int(agg.get('empty', 0) or 0)}")
    typer.echo(f"  timeouts:   {int(agg.get('timeouts', 0) or 0)}")
    typer.echo(f"  relay:      {int(agg.get('relay_calls', 0) or 0)}")
    typer.echo(f"estimated cache savings: ${savings:.4f}  (see --help for what 'estimated' means)")

    if groups is not None and by is not None:
        typer.echo(f"\nby {by}:")
        for row in groups:
            key = row.get(by, "?")
            row_calls = int(row.get("calls", 0) or 0)
            row_cost = float(row.get("cost_usd", 0.0) or 0.0)
            row_errors = int(row.get("errors", 0) or 0)
            row_hits = int(row.get("cache_hits", 0) or 0)
            typer.echo(
                f"  {key!s:<16} calls={row_calls:<6} cost=${row_cost:<10.4f} "
                f"errors={row_errors:<4} cache_hits={row_hits}"
            )


if __name__ == "__main__":
    app()
