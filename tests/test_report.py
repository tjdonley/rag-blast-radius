from rag_blast.diff import ManifestChange, ManifestDiff
from rag_blast.report import build_report, render_text_report


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
    assert "Recommended actions:" in text
    assert report["categories"] == ["embedding_model_changed"]
    assert report["finding_count"] == 5


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
