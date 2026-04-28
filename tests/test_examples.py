import json
from pathlib import Path

from rag_blast.diff import diff_manifests
from rag_blast.manifest import load_manifest
from rag_blast.report import build_report

EXAMPLE_NAMES = {
    "openai_ada_to_3_large",
    "chunk_size_change",
    "reranker_added",
    "semantic_cache_namespace_bug",
    "weaviate_alias_migration",
    "qdrant_dual_collection_migration",
}


def test_examples_catalog_is_complete() -> None:
    examples_dir = _examples_dir()
    example_dirs = {path.name for path in examples_dir.iterdir() if path.is_dir()}

    assert example_dirs == EXAMPLE_NAMES


def test_examples_include_required_files() -> None:
    for example_dir in _example_dirs():
        assert (example_dir / "old.json").exists()
        assert (example_dir / "new.json").exists()
        assert (example_dir / "expected-summary.json").exists()
        assert (example_dir / "README.md").exists()


def test_example_expected_summaries_match_current_reports() -> None:
    for example_dir in _example_dirs():
        report = build_report(
            diff_manifests(
                load_manifest(example_dir / "old.json"), load_manifest(example_dir / "new.json")
            )
        )
        expected = json.loads((example_dir / "expected-summary.json").read_text(encoding="utf-8"))

        assert _summary(report) == expected


def test_example_readmes_show_how_to_run_the_example() -> None:
    for example_dir in _example_dirs():
        readme = (example_dir / "README.md").read_text(encoding="utf-8")

        assert "Expected outcome:" in readme
        assert f"examples/{example_dir.name}/old.json" in readme
        assert f"examples/{example_dir.name}/new.json" in readme


def _summary(report: dict) -> dict:
    return {
        "risk": report["risk"],
        "change_count": report["change_count"],
        "categories": report["categories"],
        "finding_count": report["finding_count"],
        "findings": [finding["rule_id"] for finding in report["findings"]],
        "unassessed_change_count": report["unassessed_change_count"],
        "unassessed_change_paths": report["unassessed_change_paths"],
    }


def _example_dirs() -> list[Path]:
    return sorted(path for path in _examples_dir().iterdir() if path.is_dir())


def _examples_dir() -> Path:
    return Path(__file__).parent.parent / "examples"
