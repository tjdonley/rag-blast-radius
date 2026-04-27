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
    assert "Risk: UNASSESSED" in text
    assert (
        "embedding.model (embedding_model_changed): "
        "Embedding model changed; text-embedding-ada-002 -> new-model" in text
    )
    assert report["categories"] == ["embedding_model_changed"]


def test_render_text_report_handles_no_changes() -> None:
    text = render_text_report(build_report(ManifestDiff(changes=())))

    assert "Risk: NONE" in text
    assert "  - none" in text
