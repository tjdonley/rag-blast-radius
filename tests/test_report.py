import json

import pytest

from rag_blast.diff import ManifestChange, ManifestDiff
from rag_blast.report import (
    REPORT_FORMATS,
    ReportLoadError,
    build_report,
    load_report,
    normalize_fail_on,
    normalize_format,
    parse_report,
    render_github_output,
    render_html_report,
    render_json_report,
    render_markdown_report,
    render_report,
    render_text_report,
    should_fail_report,
)


def _sample_report() -> dict:
    return build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="embedding.model",
                    old="text-embedding-ada-002",
                    new="text-embedding-3-large",
                    category="embedding_model_changed",
                    summary="Embedding model changed",
                ),
            )
        )
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
    assert report["unassessed_change_count"] == 1
    assert report["unassessed_change_paths"] == ["app"]
    assert report["recommended_rollout"] == [
        "Review unassessed manifest changes before deployment."
    ]


def test_build_report_tracks_mixed_assessed_and_unassessed_changes() -> None:
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

    assert report["risk"] == "MEDIUM"
    assert report["finding_count"] == 2
    assert report["unassessed_change_count"] == 1
    assert report["unassessed_change_paths"] == ["app"]
    assert "Review unassessed manifest changes before deployment." in report["recommended_rollout"]


def test_render_json_report_is_parseable() -> None:
    report = build_report(ManifestDiff(changes=()))

    assert json.loads(render_json_report(report))["risk"] == "NONE"


def test_render_markdown_report_lists_github_summary() -> None:
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

    markdown = render_markdown_report(report)

    assert "## RAG Blast Radius" in markdown
    assert "| Risk | <code>HIGH</code> |" in markdown
    assert "| Changes | <code>1</code> |" in markdown
    assert "### Detected Changes" in markdown
    assert "<code>embedding.model</code>" in markdown
    assert "### Findings" in markdown
    assert "<code>REEMBED_REQUIRED</code>" in markdown
    assert "### Recommended Rollout" in markdown
    assert "Risk is based on deterministic local rules." in markdown


def test_render_markdown_report_handles_empty_report() -> None:
    markdown = render_markdown_report(build_report(ManifestDiff(changes=())))

    assert "| Risk | <code>NONE</code> |" in markdown
    assert "| Changes | <code>0</code> |" in markdown
    assert "### Detected Changes\n- none" in markdown
    assert "### Findings\n- none" in markdown


def test_render_markdown_report_escapes_table_values() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="app`name",
                    old="<old|app>",
                    new="new\napp",
                    category="manifest_field_changed",
                    summary="Manifest | field changed",
                ),
            )
        )
    )

    markdown = render_markdown_report(report)

    assert "<code>app&#96;name</code>" in markdown
    assert "\\`" not in markdown
    assert "Manifest \\| field changed" in markdown
    assert "&lt;old\\|app&gt;" in markdown
    assert "new<br>app" in markdown
    assert "### Unassessed Changes" in markdown


def test_render_html_report_lists_summary_sections() -> None:
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

    rendered = render_html_report(report)

    assert rendered.startswith("<!doctype html>")
    assert "<title>RAG Blast Radius Report</title>" in rendered
    assert "<h1>RAG Blast Radius Report</h1>" in rendered
    assert '<span class="risk-badge">HIGH</span>' in rendered
    assert '<section class="section" aria-labelledby="changes-heading">' in rendered
    assert '<h2 id="changes-heading">Detected Changes</h2>' in rendered
    assert "<code>embedding.model</code>" in rendered
    assert "<code>REEMBED_REQUIRED</code>" in rendered
    assert '<ol class="rollout">' in rendered
    assert "<summary>Show report payload</summary>" in rendered
    assert 'src="' not in rendered
    assert "<script" not in rendered


def test_render_html_report_handles_empty_report() -> None:
    rendered = render_html_report(build_report(ManifestDiff(changes=())))

    assert '<span class="risk-badge">NONE</span>' in rendered
    assert "No manifest changes detected." in rendered
    assert "No invalidation rules triggered." in rendered
    assert "No unassessed changes." in rendered
    assert "No rollout steps required." in rendered


def test_render_html_report_escapes_values_and_raw_json() -> None:
    report = build_report(
        ManifestDiff(
            changes=(
                ManifestChange(
                    path="app<script>",
                    old="<old&app>",
                    new='new "app"',
                    category="manifest_field_changed",
                    summary="Manifest <field> changed",
                ),
            )
        )
    )

    rendered = render_html_report(report)

    assert "app&lt;script&gt;" in rendered
    assert "&lt;old&amp;app&gt;" in rendered
    assert "new &quot;app&quot;" in rendered
    assert "Manifest &lt;field&gt; changed" in rendered
    assert "<script>" not in rendered
    assert '"path": "app&lt;script&gt;"' in rendered
    assert "Review unassessed manifest changes before deployment." in rendered


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


def test_should_fail_report_fails_mixed_unassessed_changes_when_threshold_enabled() -> None:
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

    assert should_fail_report(report, "high") is True
    assert should_fail_report(report, "none") is False


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


def test_report_formats_all_render_through_the_dispatcher() -> None:
    """Every advertised format must have a renderer behind it."""
    report = _sample_report()

    for output_format in REPORT_FORMATS:
        assert render_report(report, output_format).strip()


def test_render_report_matches_the_dedicated_renderers() -> None:
    report = _sample_report()

    assert render_report(report, "text") == render_text_report(report)
    assert render_report(report, "json") == render_json_report(report)
    assert render_report(report, "markdown") == render_markdown_report(report)
    assert render_report(report, "html") == render_html_report(report)
    assert render_report(report, "github-output") == render_github_output(report)


def test_render_report_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported report format"):
        render_report(_sample_report(), "xml")


def test_render_github_output_emits_summary_fields() -> None:
    rendered = render_github_output(_sample_report())

    assert rendered.splitlines() == [
        "risk=HIGH",
        "change_count=1",
        "finding_count=5",
        "unassessed_change_count=0",
    ]


def test_render_github_output_rejects_multiline_values() -> None:
    report = _sample_report()
    report["risk"] = "HIGH\nrisk=INJECTED"

    with pytest.raises(ReportLoadError, match="single-line"):
        render_github_output(report)


def test_normalize_format_accepts_known_values() -> None:
    assert normalize_format("HTML") == "html"
    assert normalize_format("  markdown  ") == "markdown"
    assert normalize_format("github-output") == "github-output"


def test_normalize_format_rejects_unknown_values() -> None:
    assert normalize_format("xml") is None
    assert normalize_format("") is None


def test_parse_report_round_trips_a_rendered_json_report() -> None:
    report = _sample_report()

    parsed = parse_report(render_json_report(report), source="<test>")

    assert parsed == report
    assert render_markdown_report(parsed) == render_markdown_report(report)


def test_parse_report_rejects_malformed_json() -> None:
    with pytest.raises(ReportLoadError, match="Invalid JSON in report"):
        parse_report("{", source="<test>")


def test_parse_report_rejects_non_object_payloads() -> None:
    with pytest.raises(ReportLoadError, match="must be a JSON object"):
        parse_report("[]", source="<test>")


def test_parse_report_rejects_payloads_missing_render_fields() -> None:
    payload = _sample_report()
    del payload["findings"]
    del payload["note"]

    with pytest.raises(ReportLoadError, match=r"findings: missing\n- note: missing"):
        parse_report(json.dumps(payload), source="<test>")


def test_load_report_reads_a_report_from_disk(tmp_path) -> None:
    path = tmp_path / "report.json"
    report = _sample_report()
    path.write_text(render_json_report(report), encoding="utf-8")

    assert load_report(path) == report


def test_load_report_reports_unreadable_files(tmp_path) -> None:
    with pytest.raises(ReportLoadError, match="Unable to read report"):
        load_report(tmp_path / "missing.json")


def test_parse_report_rejects_mistyped_top_level_fields() -> None:
    payload = _sample_report()
    payload["risk"] = ["HIGH"]
    payload["change_count"] = "1"
    payload["findings"] = {}

    with pytest.raises(ReportLoadError) as error:
        parse_report(json.dumps(payload), source="<test>")

    message = str(error.value)
    assert "risk: expected a string" in message
    assert "change_count: expected an integer" in message
    assert "findings: expected an array" in message


def test_parse_report_rejects_booleans_where_counts_are_required() -> None:
    payload = _sample_report()
    payload["change_count"] = True

    with pytest.raises(ReportLoadError, match="change_count: expected an integer"):
        parse_report(json.dumps(payload), source="<test>")


def test_parse_report_rejects_malformed_change_and_finding_entries() -> None:
    payload = _sample_report()
    payload["changes"] = [None]
    payload["findings"] = [{"rule_id": "X"}]

    with pytest.raises(ReportLoadError) as error:
        parse_report(json.dumps(payload), source="<test>")

    message = str(error.value)
    assert "changes[0]: expected an object" in message
    assert "findings[0]: missing severity, summary, change_paths" in message


def test_parse_report_rejects_non_array_change_paths() -> None:
    payload = _sample_report()
    payload["findings"][0]["change_paths"] = "embedding.model"

    with pytest.raises(ReportLoadError, match=r"findings\[0\]\.change_paths: expected an array"):
        parse_report(json.dumps(payload), source="<test>")


def test_any_payload_parse_report_accepts_renders_in_every_format() -> None:
    """Whatever survives validation must never make a renderer raise."""
    payload = {
        "risk": "HIGH",
        "change_count": 0,
        "categories": [None, {"a": 1}, [1]],
        "changes": [
            {
                "path": {"x": 1},
                "category": [1, 2],
                "summary": None,
                "old": {"k": [1]},
                "new": True,
            }
        ],
        "finding_count": 0,
        "findings": [
            {
                "rule_id": None,
                "severity": {"a": 1},
                "summary": [1],
                "change_paths": [None, {"z": 1}],
            }
        ],
        "unassessed_change_count": 0,
        "unassessed_change_paths": [None, {"q": 2}],
        "recommended_rollout": [None, {"s": 1}],
        "note": "n",
    }

    report = parse_report(json.dumps(payload), source="<test>")

    for output_format in REPORT_FORMATS:
        assert render_report(report, output_format) is not None
