from rag_blast.diff import ManifestChange
from rag_blast.report import build_report, render_text_report


def test_render_text_report_lists_changes() -> None:
    report = build_report(
        [ManifestChange(path="embedding.model", old="text-embedding-ada-002", new="new-model")]
    )

    text = render_text_report(report)

    assert "RAG BLAST RADIUS REPORT" in text
    assert "Risk: UNASSESSED" in text
    assert "embedding.model: text-embedding-ada-002 -> new-model" in text


def test_render_text_report_handles_no_changes() -> None:
    text = render_text_report(build_report([]))

    assert "Risk: NONE" in text
    assert "  - none" in text
