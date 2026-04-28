from pathlib import Path

from typer.testing import CliRunner

from rag_blast.cli import app

ROOT = Path(__file__).parent.parent
runner = CliRunner()


def test_readme_has_public_release_sections() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for heading in (
        "# rag-blast-radius",
        "## Why RAG Changes Are Risky",
        "## Quickstart",
        "## Example Check Output",
        "## GitHub Action",
        "## Manifest Schema",
        "## Diff Output",
        "## Rules",
        "## Non-Goals",
        "## Roadmap",
    ):
        assert heading in readme

    for phrase in (
        "rag-blast check --old examples/openai_ada_to_3_large/old.json",
        "Risk: HIGH",
        "REEMBED_REQUIRED",
        "VECTOR_INDEX_INCOMPATIBLE",
        "Automatic migrations",
        "Add integrations when real users ask",
    ):
        assert phrase in readme


def test_documented_initial_release_smoke_command() -> None:
    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(ROOT / "examples/openai_ada_to_3_large/old.json"),
            "--new",
            str(ROOT / "examples/openai_ada_to_3_large/new.json"),
        ],
    )

    assert result.exit_code == 0
    assert "Risk: HIGH" in result.output
    assert "embedding.model" in result.output
    assert "embedding.dimensions" in result.output
    assert "REEMBED_REQUIRED" in result.output
    assert "VECTOR_INDEX_INCOMPATIBLE" in result.output
    assert "SHADOW_INDEX_RECOMMENDED" in result.output
    assert "Recommended rollout:" in result.output
