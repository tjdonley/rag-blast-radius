from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_action_metadata_declares_pr_gate_inputs_and_outputs() -> None:
    action = _read("action.yml")

    assert "name: RAG Blast Radius" in action
    assert "description: Pre-deploy safety checks for RAG manifest changes" in action
    for input_name in ("old_manifest", "new_manifest", "fail_on", "format", "python_version"):
        assert f"  {input_name}:" in action
    assert "default: high" in action
    assert "default: text" in action
    assert 'default: "3.12"' in action

    for output_name in (
        "risk",
        "change_count",
        "finding_count",
        "unassessed_change_count",
    ):
        assert f"  {output_name}:" in action
        assert f"value: ${{{{ steps.run.outputs.{output_name} }}}}" in action


def test_action_runs_installed_cli_with_validated_inputs() -> None:
    action = _read("action.yml")

    assert "uses: actions/setup-python@v5" in action
    assert 'python -m pip install "$GITHUB_ACTION_PATH"' in action
    assert "rag-blast check \\" in action
    assert '--old "$OLD_MANIFEST"' in action
    assert '--new "$NEW_MANIFEST"' in action
    assert '--format "$REPORT_FORMAT"' in action
    assert '--fail-on "$FAIL_ON"' in action
    assert "text|json" in action
    assert "none|low|medium|high" in action
    assert '>> "$GITHUB_OUTPUT"' in action
    assert '>> "$GITHUB_STEP_SUMMARY"' in action


def test_action_uses_shared_markdown_renderer_for_step_summary() -> None:
    action = _read("action.yml")

    assert "from rag_blast.report import render_markdown_report" in action
    assert "print(render_markdown_report(report))" in action
    assert 'print("## RAG Blast Radius")' not in action
    assert "### Recommended Rollout" not in action


def test_action_emits_blocking_annotation_after_valid_report() -> None:
    action = _read("action.yml")

    assert 'if [ "$status" -ne 0 ]; then' in action
    assert "::error title=RAG Blast Radius blocked::" in action
    assert "risk={report['risk']}" in action
    assert "fail_on={sys.argv[2]}" in action
    assert "unassessed_changes={report['unassessed_change_count']}" in action


def test_action_validates_json_report_before_parsing_outputs() -> None:
    action = _read("action.yml")

    assert "if ! python - \"$JSON_REPORT\" >/dev/null 2>&1 <<'PY'" in action
    assert "json.load(report_file)" in action
    assert 'if [ "$status" -ne 0 ]; then' in action
    assert "rag-blast did not produce a valid JSON report" in action


def test_action_docs_include_workflows_inputs_and_json_mode() -> None:
    action_docs = _read("docs/github-action.md")
    readme = _read("README.md")

    for docs in (action_docs, readme):
        assert "uses: tjdonley/rag-blast-radius@v0" in docs
        assert "old_manifest:" in docs
        assert "new_manifest:" in docs
        assert "fail_on: high" in docs

    assert "format: json" in action_docs
    assert "risk" in action_docs
    assert "change_count" in action_docs
    assert "docs/github-action.md" in readme


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")
