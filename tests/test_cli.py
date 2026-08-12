import json

from typer.testing import CliRunner

from rag_blast.cli import app
from rag_blast.manifest import starter_manifest
from rag_blast.rules import RULE_ORDER

runner = CliRunner()


def test_cli_help_runs() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Pre-deploy safety checks" in result.output


def test_cli_init_writes_manifest(tmp_path) -> None:
    output = tmp_path / "rag-manifest.json"

    result = runner.invoke(app, ["init", "--output", str(output)])

    assert result.exit_code == 0
    assert output.exists()


def test_cli_init_refuses_to_overwrite_existing_manifest(tmp_path) -> None:
    output = tmp_path / "rag-manifest.json"
    output.write_text("existing", encoding="utf-8")

    result = runner.invoke(app, ["init", "--output", str(output)])

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_init_force_overwrites_existing_manifest(tmp_path) -> None:
    output = tmp_path / "rag-manifest.json"
    output.write_text("existing", encoding="utf-8")

    result = runner.invoke(app, ["init", "--output", str(output), "--force"])

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["app"] == "customer-support-rag"


def test_cli_check_json_reports_changes(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    new_manifest = starter_manifest()
    new_manifest["embedding"]["model"] = "text-embedding-3-large"

    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", "--old", str(old_path), "--new", str(new_path), "--format", "json"],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["risk"] == "HIGH"
    assert report["change_count"] == 2
    assert report["finding_count"] == 6
    assert report["unassessed_change_count"] == 0
    assert report["unassessed_change_paths"] == []
    assert report["categories"] == [
        "embedding_model_changed",
        "semantic_cache_namespace_unchanged",
    ]
    assert [change["path"] for change in report["changes"]] == [
        "caches[support_rag_prod_v4].namespace",
        "embedding.model",
    ]
    assert [change["summary"] for change in report["changes"]] == [
        "Semantic cache namespace unchanged after embedding, chunking, or retrieval change",
        "Embedding model changed",
    ]
    assert [finding["rule_id"] for finding in report["findings"]] == [
        "REEMBED_REQUIRED",
        "VECTOR_INDEX_INCOMPATIBLE",
        "SEMANTIC_CACHE_UNSAFE",
        "RETRIEVAL_BASELINE_STALE",
        "SHADOW_INDEX_RECOMMENDED",
        "ROLLBACK_REQUIRES_OLD_INDEX",
    ]
    assert report["recommended_rollout"]


def test_cli_check_fail_on_high_exits_with_failure_after_json_report(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    new_manifest = starter_manifest()
    new_manifest["embedding"]["model"] = "text-embedding-3-large"

    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--fail-on",
            "high",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["risk"] == "HIGH"


def test_cli_check_fail_on_high_exits_for_unassessed_changes(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    new_manifest = starter_manifest()
    new_manifest["app"] = "customer-support-rag-v2"

    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", "--old", str(old_path), "--new", str(new_path), "--fail-on", "high"],
    )

    assert result.exit_code == 1
    assert "Risk: UNASSESSED" in result.output


def test_cli_check_fail_on_high_exits_for_mixed_unassessed_changes(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    old_manifest["caches"] = []
    new_manifest = starter_manifest()
    new_manifest["caches"] = []
    new_manifest["app"] = "customer-support-rag-v2"
    new_manifest["retriever"]["top_k"] = 12

    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--fail-on",
            "high",
        ],
    )

    report = json.loads(result.output)
    assert result.exit_code == 1
    assert report["risk"] == "MEDIUM"
    assert report["unassessed_change_paths"] == ["app"]


def test_cli_check_fail_on_high_allows_no_changes(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", "--old", str(manifest_path), "--new", str(manifest_path), "--fail-on", "high"],
    )

    assert result.exit_code == 0
    assert "Risk: NONE" in result.output


def test_cli_check_text_preserves_keyed_paths(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    new_manifest = starter_manifest()
    new_manifest["embedding"]["model"] = "text-embedding-3-large"

    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")

    result = runner.invoke(app, ["check", "--old", str(old_path), "--new", str(new_path)])

    assert result.exit_code == 0
    assert "caches[support_rag_prod_v4].namespace" in result.output
    assert "Invalidation rules triggered:" in result.output


def test_cli_check_rejects_invalid_format(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", "--old", str(manifest_path), "--new", str(manifest_path), "--format", "xml"],
    )

    assert result.exit_code == 1
    assert "Unsupported format" in result.stderr


def test_cli_check_rejects_invalid_fail_on_threshold(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(manifest_path),
            "--new",
            str(manifest_path),
            "--fail-on",
            "critical",
        ],
    )

    assert result.exit_code == 1
    assert "Unsupported fail-on threshold" in result.stderr


def test_cli_check_rejects_malformed_json(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text("{", encoding="utf-8")
    new_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(app, ["check", "--old", str(old_path), "--new", str(new_path)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.stderr


def test_cli_explain_known_rule() -> None:
    result = runner.invoke(app, ["explain", "REEMBED_REQUIRED"])

    assert result.exit_code == 0
    assert "REEMBED_REQUIRED" in result.output


def _write_manifests(tmp_path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_manifest = starter_manifest()
    new_manifest = starter_manifest()
    new_manifest["embedding"]["model"] = "text-embedding-3-large"
    old_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    new_path.write_text(json.dumps(new_manifest), encoding="utf-8")
    return old_path, new_path


def test_cli_check_renders_markdown(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)

    result = runner.invoke(
        app,
        ["check", "--old", str(old_path), "--new", str(new_path), "--format", "markdown"],
    )

    assert result.exit_code == 0
    assert "## RAG Blast Radius" in result.output
    assert "| Risk | <code>HIGH</code> |" in result.output


def test_cli_check_renders_html(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)

    result = runner.invoke(
        app,
        ["check", "--old", str(old_path), "--new", str(new_path), "--format", "html"],
    )

    assert result.exit_code == 0
    assert result.output.startswith("<!doctype html>")
    assert "<title>RAG Blast Radius Report</title>" in result.output


def test_cli_check_renders_github_output(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)

    result = runner.invoke(
        app,
        ["check", "--old", str(old_path), "--new", str(new_path), "--format", "github-output"],
    )

    assert result.exit_code == 0
    assert "risk=HIGH" in result.output
    assert "change_count=2" in result.output


def test_cli_check_writes_report_to_output_file(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)
    output = tmp_path / "nested" / "report.html"

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "html",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote report:" in result.stderr
    assert result.stdout == ""
    written = output.read_text(encoding="utf-8")
    assert written.startswith("<!doctype html>")
    assert written.endswith("</html>\n")


def test_cli_check_writes_output_file_before_failing_threshold(tmp_path) -> None:
    """A blocking run must still leave its report behind for CI to publish."""
    old_path, new_path = _write_manifests(tmp_path)
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(output),
            "--fail-on",
            "high",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["risk"] == "HIGH"


def test_cli_check_overwrites_an_existing_output_file(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("stale", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["risk"] == "HIGH"


def test_cli_report_rerenders_a_saved_json_report(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)
    report_path = tmp_path / "report.json"
    runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
    )

    result = runner.invoke(app, ["report", "--input", str(report_path), "--format", "markdown"])

    assert result.exit_code == 0
    assert "## RAG Blast Radius" in result.output
    assert "<code>REEMBED_REQUIRED</code>" in result.output


def test_cli_report_writes_to_an_output_file(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    old_path, new_path = _write_manifests(tmp_path)
    runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
    )
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", "--input", str(report_path), "--format", "html", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_report_rejects_malformed_json(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["report", "--input", str(report_path)])

    assert result.exit_code == 1
    assert "Invalid JSON in report" in result.stderr


def test_cli_report_rejects_a_payload_that_is_not_a_report(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"risk": "HIGH"}), encoding="utf-8")

    result = runner.invoke(app, ["report", "--input", str(report_path)])

    assert result.exit_code == 1
    assert "is not a rag-blast report" in result.stderr
    assert "changes: missing" in result.stderr


def test_cli_report_reports_a_malformed_payload_instead_of_crashing(tmp_path) -> None:
    """A structurally wrong report must produce a diagnostic, never a traceback."""
    report_path = tmp_path / "report.json"
    payload = {
        "risk": "HIGH",
        "change_count": 1,
        "categories": [],
        "changes": [None],
        "finding_count": 0,
        "findings": {},
        "unassessed_change_count": 0,
        "unassessed_change_paths": [],
        "recommended_rollout": [],
        "note": "x",
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["report", "--input", str(report_path)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "findings: expected an array" in result.stderr
    assert "changes[0]: expected an object" in result.stderr
    assert "Traceback" not in result.output


def test_cli_report_rejects_a_missing_file(tmp_path) -> None:
    result = runner.invoke(app, ["report", "--input", str(tmp_path / "nope.json")])

    assert result.exit_code == 1
    assert "Unable to read report" in result.stderr


def test_cli_report_rejects_an_unsupported_format(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["report", "--input", str(report_path), "--format", "xml"])

    assert result.exit_code == 1
    assert "Unsupported format" in result.stderr


def test_cli_validate_accepts_a_valid_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(manifest_path)])

    assert result.exit_code == 0
    assert "Valid manifest:" in result.output


def test_cli_validate_reports_field_level_errors(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = starter_manifest()
    manifest["chunking"]["chunk_overlap"] = 5000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(manifest_path)])

    assert result.exit_code == 1
    assert "chunk_overlap must be smaller than chunk_size" in result.output


def test_cli_validate_checks_every_path_before_failing(tmp_path) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps(starter_manifest()), encoding="utf-8")
    bad.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(bad), str(good)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output
    assert "Valid manifest:" in result.output


def test_cli_rules_lists_every_rule() -> None:
    result = runner.invoke(app, ["rules"])

    assert result.exit_code == 0
    for rule_id in RULE_ORDER:
        assert rule_id in result.output


def test_cli_rules_json_exposes_triggers() -> None:
    result = runner.invoke(app, ["rules", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [rule["rule_id"] for rule in payload] == list(RULE_ORDER)
    reembed = next(rule for rule in payload if rule["rule_id"] == "REEMBED_REQUIRED")
    assert reembed["severity"] == "HIGH"
    assert "embedding_model_changed" in reembed["triggered_by"]


def test_cli_rules_rejects_an_unsupported_format() -> None:
    result = runner.invoke(app, ["rules", "--format", "html"])

    assert result.exit_code == 1
    assert "Unsupported format" in result.stderr


def test_cli_schema_prints_a_usable_json_schema() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert "$schema" in schema["properties"]
    assert "embedding" in schema["properties"]


def test_cli_schema_writes_to_a_file(tmp_path) -> None:
    output = tmp_path / "schema.json"

    result = runner.invoke(app, ["schema", "--output", str(output)])

    assert result.exit_code == 0
    assert "Wrote manifest schema:" in result.output
    assert json.loads(output.read_text(encoding="utf-8"))["title"] == "RAG Manifest"


def test_cli_report_reads_stdin(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)
    report_path = tmp_path / "report.json"
    runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
    )

    result = runner.invoke(
        app,
        ["report", "--input", "-", "--format", "github-output"],
        input=report_path.read_text(encoding="utf-8"),
    )

    assert result.exit_code == 0
    assert "risk=HIGH" in result.output


def test_cli_check_keeps_stdout_machine_readable_when_writing_a_file(tmp_path) -> None:
    """Status text on stdout would corrupt --format json for anything parsing it."""
    old_path, new_path = _write_manifests(tmp_path)

    for output_format in ("json", "github-output"):
        result = runner.invoke(
            app,
            [
                "check",
                "--old",
                str(old_path),
                "--new",
                str(new_path),
                "--format",
                output_format,
                "--output",
                str(tmp_path / f"report.{output_format}"),
            ],
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert "Wrote report:" in result.stderr


def test_cli_report_keeps_stdout_machine_readable_when_writing_a_file(tmp_path) -> None:
    old_path, new_path = _write_manifests(tmp_path)
    report_path = tmp_path / "report.json"
    runner.invoke(
        app,
        [
            "check",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
    )

    result = runner.invoke(
        app,
        [
            "report",
            "--input",
            str(report_path),
            "--format",
            "html",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Wrote report:" in result.stderr


def test_data_commands_never_write_diagnostics_to_stdout(tmp_path) -> None:
    """stdout carries report data or nothing, so redirecting it can never capture errors."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(starter_manifest()), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    not_a_report = tmp_path / "not-a-report.json"
    not_a_report.write_text(json.dumps({"risk": "HIGH"}), encoding="utf-8")

    failing_invocations = [
        ["check", "--old", str(manifest), "--new", str(manifest), "--format", "xml"],
        ["check", "--old", str(manifest), "--new", str(manifest), "--fail-on", "critical"],
        ["check", "--old", str(broken), "--new", str(manifest), "--format", "json"],
        ["report", "--input", str(broken), "--format", "json"],
        ["report", "--input", str(not_a_report), "--format", "json"],
        ["report", "--input", str(tmp_path / "missing.json"), "--format", "json"],
        ["report", "--input", str(not_a_report), "--format", "xml"],
        ["rules", "--format", "xml"],
    ]

    for invocation in failing_invocations:
        result = runner.invoke(app, invocation)

        assert result.exit_code == 1, invocation
        assert result.stdout == "", f"{invocation} leaked to stdout: {result.stdout!r}"
        assert result.stderr.strip(), f"{invocation} reported nothing on stderr"
