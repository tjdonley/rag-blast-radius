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
