import json

from rag_blast.diff import ManifestChange, ManifestDiff
from rag_blast.report import (
    build_report,
    normalize_fail_on,
    render_json_report,
    render_text_report,
    should_fail_report,
)


def test_render_text_report_lists_changes() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="embedding.model",
                    old="text-embedding-ada-002",
                    new="new-model",
                    category="embedding_model_changed",
                    summary="Embedding model changed",
                ),
            )
        )
    )

    text = render_text_report(report)

    assert "RAG BLAST RADIUS REPORT" in text
    assert "Risk: HIGH" in text
    assert (
        "embedding.model (embedding_model_changed): "
        "Embedding model changed; text-embedding-ada-002 -> new-model" in text
    )
    assert "Invalidation rules triggered:" in text
    assert "HIGH: REEMBED_REQUIRED" in text
    assert "Recommended rollout:" in text
    assert report["categories"] == ["embedding_model_changed"]
    assert report["finding_count"] == 5
    assert report["recommended_rollout"]


def test_render_text_report_handles_no_changes() -> None:
    text = render_text_report(build_report(ManifestDiff(changes=())))

    assert "Risk: NONE" in text
    assert "  - none" in text
    assert "Invalidation rules triggered:" in text


def test_build_report_keeps_unknown_change_risk_unassessed() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="app",
                    old="support-rag",
                    new="support-rag-v2",
                    category="manifest_field_changed",
                    summary="Manifest field changed",
                ),
            )
        )
    )

    assert report["risk"] == "UNASSESSED"
    assert report["change_count"] == 1
    assert report["finding_count"] == 0
    assert report["findings"] == []
    assert report["recommended_rollout"] == [
        "Review unassessed manifest changes before deployment."
    ]


def test_render_json_report_is_parseable() -> None:
    report = build_report(ManifestDiff(changes=()))

    assert json.loads(render_json_report(report))["risk"] == "NONE"


def test_normalize_fail_on_accepts_known_values() -> None:
    assert normalize_fail_on("HIGH") == "high"
    assert normalize_fail_on("none") == "none"


def test_normalize_fail_on_rejects_unknown_values() -> None:
    assert normalize_fail_on("critical") is None


def test_should_fail_report_uses_severity_thresholds() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="retriever.top_k",
                    old=8,
                    new=12,
                    category="retriever_top_k_changed",
                    summary="Retriever top_k changed",
                ),
            )
        )
    )

    assert should_fail_report(report, "medium") is True
    assert should_fail_report(report, "high") is False


def test_should_fail_report_fails_unassessed_changes_when_threshold_enabled() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="app",
                    old="support-rag",
                    new="support-rag-v2",
                    category="manifest_field_changed",
                    summary="Manifest field changed",
                ),
            )
        )
    )

    assert should_fail_report(report, "high") is True
    assert should_fail_report(report, "none") is False


def test_should_fail_report_does_not_fail_empty_reports() -> None:
    report = build_report(ManifestDiff(changes=()))

    assert should_fail_report(report, "low") is False
