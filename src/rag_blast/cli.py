from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from rag_blast import __version__
from rag_blast.diff import diff_manifests
from rag_blast.integrations import (
    IntegrationScanError,
    render_partial_manifest,
    scan_llamaindex_qdrant,
)
from rag_blast.manifest import (
    ManifestLoadError,
    load_manifest,
    manifest_json_schema,
    write_starter_manifest,
)
from rag_blast.report import (
    REPORT_FORMATS,
    ReportLoadError,
    build_report,
    load_report,
    normalize_fail_on,
    normalize_format,
    parse_report,
    render_report,
    should_fail_report,
)
from rag_blast.rules import get_rule, rules_payload

integrations_app = typer.Typer(
    help="Generate manifest drafts from known RAG framework patterns.",
    no_args_is_help=True,
)
app = typer.Typer(
    help="Pre-deploy safety checks for RAG changes.",
    no_args_is_help=True,
)
app.add_typer(integrations_app, name="integrations")
console = Console()
err_console = Console(stderr=True)


FORMAT_HELP = f"Report format: {', '.join(REPORT_FORMATS)}."


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rag-blast {__version__}")
        raise typer.Exit()


def _resolve_format(value: str) -> str:
    resolved = normalize_format(value)
    if resolved is None:
        console.print(f"[red]Unsupported format.[/red] Use one of: {', '.join(REPORT_FORMATS)}.")
        raise typer.Exit(1)
    return resolved


def _emit_report(report: dict[str, Any], output_format: str, output: Path | None) -> None:
    """Write a rendered report to a file, or print it to stdout."""
    rendered = render_report(report, output_format).rstrip("\n")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"[green]Wrote report:[/green] {output}")
        return

    if output_format == "text":
        console.print(rendered, markup=False)
        return

    typer.echo(rendered)


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
        help=FORMAT_HELP,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the report to a file instead of stdout. Existing files are overwritten.",
    ),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="Exit with code 1 when risk is at least: none, low, medium, or high.",
    ),
) -> None:
    """Compare two RAG manifests."""
    resolved_format = _resolve_format(output_format)

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

    _emit_report(report, resolved_format, output)

    if should_fail_report(report, fail_threshold):
        raise typer.Exit(1)


@app.command("report")
def report_command(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="JSON report from 'rag-blast check --format json'. Use '-' to read stdin.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help=FORMAT_HELP,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the report to a file instead of stdout. Existing files are overwritten.",
    ),
) -> None:
    """Re-render a saved JSON report in another format."""
    resolved_format = _resolve_format(output_format)

    try:
        if str(input_path) == "-":
            report = parse_report(sys.stdin.read(), source="<stdin>")
        else:
            report = load_report(input_path)
        _emit_report(report, resolved_format, output)
    except ReportLoadError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from None


@app.command("validate")
def validate_command(
    manifests: list[Path] = typer.Argument(
        ...,
        help="One or more manifest paths to validate.",
    ),
) -> None:
    """Validate RAG manifests without comparing them."""
    invalid = False
    for manifest_path in manifests:
        try:
            load_manifest(manifest_path)
        except ManifestLoadError as error:
            invalid = True
            console.print(f"[red]{error}[/red]")
            continue

        console.print(f"[green]Valid manifest:[/green] {manifest_path}")

    if invalid:
        raise typer.Exit(1)


@app.command("rules")
def rules_command(
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
    ),
) -> None:
    """List the deterministic blast-radius rules."""
    normalized = output_format.strip().lower()
    if normalized not in {"text", "json"}:
        console.print("[red]Unsupported format.[/red] Use 'text' or 'json'.")
        raise typer.Exit(1)

    payload = rules_payload()
    if normalized == "json":
        typer.echo(json.dumps(payload, indent=2))
        return

    for rule in payload:
        console.print(f"[bold]{rule['rule_id']}[/bold] ({rule['severity']})")
        console.print(f"  {rule['summary']}", markup=False)

    console.print("")
    console.print("Run 'rag-blast explain RULE_ID' for the recommended action.")


@app.command("schema")
def schema_command(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the JSON Schema to a file instead of stdout. Existing files are overwritten.",
    ),
) -> None:
    """Print the JSON Schema for the RAG manifest."""
    rendered = json.dumps(manifest_json_schema(), indent=2)

    if output is None:
        typer.echo(rendered)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    console.print(f"[green]Wrote manifest schema:[/green] {output}")


@integrations_app.command("llamaindex-qdrant")
def llamaindex_qdrant_command(
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        help="Python file or directory to inspect for LlamaIndex + Qdrant config.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the partial manifest. Defaults to stdout.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing output file.",
    ),
) -> None:
    """Generate a partial manifest from LlamaIndex + Qdrant configuration."""
    try:
        scan = scan_llamaindex_qdrant(source)
    except IntegrationScanError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from None

    rendered_manifest = render_partial_manifest(scan.manifest)
    if output is None:
        typer.echo(rendered_manifest, nl=False)
        message_console = err_console
    else:
        if output.exists() and not force:
            console.print(f"[red]Output already exists:[/red] {output}")
            console.print("Use --force to overwrite it.")
            raise typer.Exit(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered_manifest, encoding="utf-8")
        console.print(f"[green]Wrote partial manifest:[/green] {output}")
        message_console = console

    message_console.print(f"Scanned {len(scan.scanned_files)} Python file(s).")
    if scan.warnings:
        message_console.print("[yellow]Manual review required:[/yellow]")
        for warning in scan.warnings:
            message_console.print(f"- {warning}")


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
