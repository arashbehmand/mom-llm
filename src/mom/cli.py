"""``mom`` command-line interface.

Thin Typer app. Subcommands delegate to the library; nothing heavy runs at import.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
