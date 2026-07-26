"""``mom`` command-line interface.

Thin Typer app. Subcommands delegate to the library; nothing heavy runs at import.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
