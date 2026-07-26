"""``mom`` command-line interface.

Thin Typer app. Subcommands delegate to the library; nothing heavy runs at import.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import urllib.error
import urllib.request

import typer

from mom import __version__


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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Dev auto-reload (never in prod)."),
) -> None:
    """Run the MoM server."""
    import uvicorn

    uvicorn.run(
        "mom.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
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

_PathArg = typer.Argument(..., exists=True, dir_okay=False, help="Path to the config YAML.")
_OverlayOpt = typer.Option(None, "--overlay", help="Optional local override file to deep-merge.")


@config_app.command("validate")
def config_validate(path: Path = _PathArg, overlay: Path | None = _OverlayOpt) -> None:
    """Load, validate, and resolve a config; exit non-zero on any problem."""
    from mom.config.loader import load_config
    from mom.config.resolve import ConfigError

    try:
        catalog = load_config(path, overlay=overlay)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.secho(f"invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"OK — {len(catalog.llms)} llms, {len(catalog.ensembles)} ensembles",
        fg=typer.colors.GREEN,
    )


@config_app.command("show")
def config_show(
    path: Path = _PathArg,
    ensemble: str | None = typer.Argument(None, help="Show just this ensemble."),
    overlay: Path | None = _OverlayOpt,
) -> None:
    """Print the fully-resolved catalog (flattened effort matrix per ensemble)."""
    from mom.config.loader import load_config
    from mom.config.resolve import ConfigError, ResolvedEnsemble

    try:
        catalog = load_config(path, overlay=overlay)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.secho(f"invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

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


cache_app = typer.Typer(help="Inspect and manage the response cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

_CacheConfigOpt = typer.Option(
    None, "--config", "-c", exists=True, dir_okay=False, help="Config YAML (to resolve data_dir)."
)
_CacheDataDirOpt = typer.Option(
    None, "--data-dir", help="Data directory holding cache.db (overrides config/env)."
)


def _resolve_data_dir(config: Path | None, data_dir: Path | None) -> Path:
    """Resolve the data directory the same way the server does (explicit flag wins)."""
    if data_dir is not None:
        return data_dir
    from mom.runtime.settings import Settings

    settings = Settings()
    if settings.data_dir is not None:
        return Path(settings.data_dir)
    config_path = config or settings.config_file
    if config_path is not None:
        from mom.config.loader import load_config
        from mom.runtime.wiring import resolve_data_dir

        return resolve_data_dir(settings, load_config(config_path))
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
    config: Path | None = _CacheConfigOpt, data_dir: Path | None = _CacheDataDirOpt
) -> None:
    """Print response-cache statistics (entries / bytes / hits)."""
    db = _resolve_data_dir(config, data_dir) / "cache.db"
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
    config: Path | None = _CacheConfigOpt,
    data_dir: Path | None = _CacheDataDirOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete every entry in the response cache."""
    db = _resolve_data_dir(config, data_dir) / "cache.db"
    if not db.exists():
        typer.echo(f"cache: empty (no database at {db})")
        return
    if not yes:
        typer.confirm(f"Purge all cached entries at {db}?", abort=True)
    removed = asyncio.run(_cache_clear(db))
    typer.secho(
        f"purged {removed} cache entr{'y' if removed == 1 else 'ies'}", fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
