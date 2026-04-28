from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from rag_blast import __version__
from rag_blast.diff import diff_manifests
from rag_blast.manifest import ManifestLoadError, load_manifest, write_starter_manifest
from rag_blast.report import (
    build_report,
    normalize_fail_on,
    render_json_report,
    render_text_report,
    should_fail_report,
)
from rag_blast.rules import get_rule

app = typer.Typer(
    help="Pre-deploy safety checks for RAG changes.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rag-blast {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed rag-blast version.",
    ),
) -> None:
    """Run rag-blast commands."""


@app.command("init")
def init_command(
    output: Path = typer.Option(
        Path(".rag-manifest.json"),
        "--output",
        "-o",
        help="Path where the starter manifest should be written.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing manifest.",
    ),
) -> None:
    """Create a starter RAG manifest."""
    try:
        write_starter_manifest(output, force=force)
    except FileExistsError:
        console.print(f"[red]Manifest already exists:[/red] {output}")
        console.print("Use --force to overwrite it.")
        raise typer.Exit(1) from None

    console.print(f"[green]Created starter manifest:[/green] {output}")


@app.command("check")
def check_command(
    old_manifest: Path = typer.Option(
        ...,
        "--old",
        help="Path to the baseline RAG manifest.",
    ),
    new_manifest: Path = typer.Option(
        ...,
        "--new",
        help="Path to the proposed RAG manifest.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Report format: text or json.",
    ),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="Exit with code 1 when risk is at least: none, low, medium, or high.",
    ),
) -> None:
    """Compare two RAG manifests."""
    if output_format not in {"text", "json"}:
        console.print("[red]Unsupported format.[/red] Use 'text' or 'json'.")
        raise typer.Exit(1)

    fail_threshold = normalize_fail_on(fail_on)
    if fail_threshold is None:
        console.print(
            "[red]Unsupported fail-on threshold.[/red] Use 'none', 'low', 'medium', or 'high'."
        )
        raise typer.Exit(1)

    try:
        old_data = load_manifest(old_manifest)
        new_data = load_manifest(new_manifest)
    except ManifestLoadError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from None

    changes = diff_manifests(old_data, new_data)
    report = build_report(changes)

    if output_format == "json":
        typer.echo(render_json_report(report))
    else:
        console.print(render_text_report(report), markup=False)

    if should_fail_report(report, fail_threshold):
        raise typer.Exit(1)


@app.command("explain")
def explain_command(rule_id: str = typer.Argument(..., help="Rule identifier to explain.")) -> None:
    """Explain a blast-radius rule."""
    rule = get_rule(rule_id)
    if rule is None:
        console.print(f"[red]Unknown rule:[/red] {rule_id}")
        raise typer.Exit(1)

    console.print(f"[bold]{rule.id}[/bold]")
    console.print(f"Severity: {rule.severity}")
    console.print(f"Summary: {rule.summary}")
    console.print(f"Recommendation: {rule.recommendation}")
