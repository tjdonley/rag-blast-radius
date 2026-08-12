import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "openai_ada_to_3_large"


def test_action_yaml_parses() -> None:
    """GitHub refuses to run an action whose metadata is not valid YAML."""
    action = _load_action()

    assert isinstance(action, dict)
    assert action["runs"]["using"] == "composite"


def test_action_metadata_declares_pr_gate_inputs_and_outputs() -> None:
    action = _load_action()

    assert action["name"] == "RAG Blast Radius"
    assert action["description"] == "Pre-deploy safety checks for RAG manifest changes"

    inputs = action["inputs"]
    assert set(inputs) == {
        "old_manifest",
        "new_manifest",
        "fail_on",
        "format",
        "python_version",
        "pr_comment",
        "github_token",
    }
    assert inputs["old_manifest"]["required"] is True
    assert inputs["new_manifest"]["required"] is True
    assert inputs["fail_on"]["default"] == "high"
    assert inputs["format"]["default"] == "text"
    assert inputs["python_version"]["default"] == "3.12"
    assert inputs["pr_comment"]["default"] == "false"

    outputs = action["outputs"]
    assert set(outputs) == {"risk", "change_count", "finding_count", "unassessed_change_count"}
    for output_name in outputs:
        assert outputs[output_name]["value"] == f"${{{{ steps.run.outputs.{output_name} }}}}"


def test_action_installs_the_cli_and_validates_inputs() -> None:
    action = _load_action()
    steps = action["runs"]["steps"]

    assert steps[0]["uses"] == "actions/setup-python@v5"
    assert 'python -m pip install "$GITHUB_ACTION_PATH"' in steps[1]["run"]

    script = _run_script()
    assert "text|json|markdown|html|github-output" in script
    assert "none|low|medium|high" in script
    assert "true|false" in script


def test_action_delegates_rendering_to_the_report_command() -> None:
    """The action should shell out to rag-blast, not embed its own Python."""
    script = _run_script()

    assert "<<'PY'" not in script
    assert "import json" not in script
    assert "from rag_blast.report import" not in script

    assert 'rag-blast report --input "$JSON_REPORT" --format "$REPORT_FORMAT"' in script
    assert 'rag-blast report --input "$JSON_REPORT" --format github-output' in script
    assert 'rag-blast report --input "$JSON_REPORT" --format markdown' in script


def test_action_runs_check_once_and_reuses_the_json_report() -> None:
    script = _run_script()

    assert script.count("rag-blast check") == 1
    assert '--output "$JSON_REPORT"' in script
    assert "PIPESTATUS" not in script
    assert "tee " not in script


def test_action_core_path_needs_no_gh_or_jq() -> None:
    """gh and jq are only acceptable inside the optional PR comment branch."""
    script = _run_script()
    core, _, pr_comment = script.partition('if [ "$PR_COMMENT" = "true" ]; then')

    assert pr_comment, "PR comment branch not found"
    assert "gh api" not in core
    assert "jq " not in core


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_writes_outputs_and_summary_for_a_high_risk_change(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="none", report_format="text")

    assert result.returncode == 0, result.stdout + result.stderr

    outputs = _parse_env_file(tmp_path / "github_output")
    assert outputs == {
        "risk": "HIGH",
        "change_count": "5",
        "finding_count": "5",
        "unassessed_change_count": "2",
    }

    summary = (tmp_path / "github_summary").read_text(encoding="utf-8")
    assert "## RAG Blast Radius" in summary
    assert "<code>HIGH</code>" in summary
    assert "### Recommended Rollout" in summary

    assert "RAG BLAST RADIUS REPORT" in result.stdout
    assert "REEMBED_REQUIRED" in result.stdout

    report = json.loads((tmp_path / "runner_temp" / "rag-blast-report.json").read_text())
    assert report["risk"] == "HIGH"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_blocks_and_annotates_when_fail_on_trips(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="high", report_format="text")

    assert result.returncode == 1
    assert (
        "::error title=RAG Blast Radius blocked::risk=HIGH; fail_on=high; "
        "findings=5; unassessed_changes=2" in result.stdout
    )

    outputs = _parse_env_file(tmp_path / "github_output")
    assert outputs["risk"] == "HIGH"

    summary = (tmp_path / "github_summary").read_text(encoding="utf-8")
    assert "## RAG Blast Radius" in summary


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_honours_the_requested_log_format(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="none", report_format="markdown")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "## RAG Blast Radius" in result.stdout
    assert "RAG BLAST RADIUS REPORT" not in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_rejects_an_unsupported_format(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="none", report_format="xml")

    assert result.returncode == 1
    assert "::error::Unsupported report format: xml" in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_fails_clearly_when_a_manifest_is_invalid(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    result = _execute_action(tmp_path, fail_on="high", report_format="text", new_manifest=broken)

    assert result.returncode == 1
    assert "::error::rag-blast did not produce a report." in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_skips_pr_comment_without_a_token(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="none", report_format="text", pr_comment="true")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr_comment requested but github_token was not provided" in result.stdout


def _execute_action(
    tmp_path: Path,
    *,
    fail_on: str,
    report_format: str,
    pr_comment: str = "false",
    new_manifest: Path | None = None,
    github_output: Path | None = None,
    github_summary: Path | None = None,
    github_token: str = "",
    event_path: Path | None = None,
    repository: str | None = None,
    extra_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir(exist_ok=True)
    github_output = github_output or tmp_path / "github_output"
    github_summary = github_summary or tmp_path / "github_summary"
    for env_file in (github_output, github_summary):
        if env_file.parent.is_dir():
            env_file.touch()

    env = {
        **os.environ,
        "OLD_MANIFEST": str(EXAMPLE / "old.json"),
        "NEW_MANIFEST": str(new_manifest or EXAMPLE / "new.json"),
        "FAIL_ON": fail_on,
        "REPORT_FORMAT": report_format,
        "PR_COMMENT": pr_comment,
        "GH_TOKEN": github_token,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(github_output),
        "GITHUB_STEP_SUMMARY": str(github_summary),
        "PATH": os.pathsep.join(
            part
            for part in (
                str(extra_path) if extra_path else "",
                str(Path(sys.executable).parent),
                os.environ.get("PATH", ""),
            )
            if part
        ),
    }
    env.pop("GITHUB_EVENT_PATH", None)
    if event_path is not None:
        env["GITHUB_EVENT_PATH"] = str(event_path)
    if repository is not None:
        env["GITHUB_REPOSITORY"] = repository

    return subprocess.run(
        ["bash", "-c", _run_script()],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=False,
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition("=")
        entries[key] = value
    return entries


def _run_script() -> str:
    for step in _load_action()["runs"]["steps"]:
        if step.get("id") == "run":
            return step["run"]
    raise AssertionError("action.yml has no step with id 'run'")


def _load_action() -> dict:
    return yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_does_not_reuse_a_previous_runs_report(tmp_path) -> None:
    """$RUNNER_TEMP is shared across a job, so a stale report must never be re-rendered."""
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    first = _execute_action(tmp_path, fail_on="none", report_format="text")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _parse_env_file(tmp_path / "github_output")["risk"] == "HIGH"

    (tmp_path / "github_output").write_text("", encoding="utf-8")
    (tmp_path / "github_summary").write_text("", encoding="utf-8")

    second = _execute_action(
        tmp_path, fail_on="none", report_format="text", new_manifest=broken
    )

    assert second.returncode == 1
    assert "::error::rag-blast did not produce a report." in second.stdout
    assert (tmp_path / "github_output").read_text(encoding="utf-8") == ""
    assert (tmp_path / "github_summary").read_text(encoding="utf-8") == ""
    assert not (tmp_path / "runner_temp" / "rag-blast-report.json").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_json_format_logs_a_parseable_document(tmp_path) -> None:
    """format: json is documented as machine-readable, so stdout must be pure JSON."""
    result = _execute_action(tmp_path, fail_on="none", report_format="json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["risk"] == "HIGH"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_github_output_format_logs_only_key_value_lines(tmp_path) -> None:
    result = _execute_action(tmp_path, fail_on="none", report_format="github-output")

    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        "risk=HIGH",
        "change_count=5",
        "finding_count=5",
        "unassessed_change_count=2",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_fails_when_step_outputs_cannot_be_written(tmp_path) -> None:
    """A passing check must not report success when its documented outputs are missing."""
    result = _execute_action(
        tmp_path,
        fail_on="none",
        report_format="text",
        github_output=tmp_path / "missing-dir" / "github_output",
    )

    assert result.returncode == 1
    assert "::error::Unable to write step outputs to GITHUB_OUTPUT." in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run the action")
def test_action_script_fails_when_the_job_summary_cannot_be_written(tmp_path) -> None:
    result = _execute_action(
        tmp_path,
        fail_on="none",
        report_format="text",
        github_summary=tmp_path / "missing-dir" / "github_summary",
    )

    assert result.returncode == 1
    assert "::error::Unable to write the job summary to GITHUB_STEP_SUMMARY." in result.stdout


GH_STUB = """#!/usr/bin/env bash
set -u
printf 'CALL %s\\n' "$*" >> "$GH_STUB_LOG"
previous=""
for argument in "$@"; do
  if [ "$previous" = "--input" ]; then
    cp "$argument" "$GH_STUB_PAYLOAD"
  fi
  previous="$argument"
done
case "$*" in
  *"-X PATCH"*|*"-X POST"*) exit 0 ;;
  *) cat "$GH_STUB_COMMENTS" ;;
esac
"""


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="bash and jq are required to run the action's PR comment path",
)
def test_action_creates_a_pr_comment_when_no_marker_exists(tmp_path) -> None:
    calls, body = _run_pr_comment_action(tmp_path, existing_comments=[])

    assert any("-X POST" in call and "issues/7/comments" in call for call in calls)
    assert not any("-X PATCH" in call for call in calls)
    assert "<!-- rag-blast-radius-report -->" in body
    assert "## RAG Blast Radius" in body
    assert "REEMBED_REQUIRED" in body


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="bash and jq are required to run the action's PR comment path",
)
def test_action_updates_the_existing_marked_pr_comment(tmp_path) -> None:
    existing = [
        {"id": 111, "body": "unrelated review chatter"},
        {"id": 222, "body": "<!-- rag-blast-radius-report -->\n\nstale report"},
        {"id": 333, "body": "<!-- rag-blast-radius-report -->\n\nolder duplicate"},
    ]

    calls, body = _run_pr_comment_action(tmp_path, existing_comments=existing)

    assert any("-X PATCH" in call and "issues/comments/222" in call for call in calls)
    assert not any("-X POST" in call for call in calls)
    assert not any("issues/comments/111" in call for call in calls)
    assert "## RAG Blast Radius" in body


def _run_pr_comment_action(tmp_path: Path, *, existing_comments: list) -> tuple[list[str], str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh_stub = stub_dir / "gh"
    gh_stub.write_text(GH_STUB, encoding="utf-8")
    gh_stub.chmod(0o755)

    comments_fixture = tmp_path / "comments.json"
    comments_fixture.write_text(json.dumps(existing_comments), encoding="utf-8")
    log = tmp_path / "gh.log"
    log.touch()

    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7}}), encoding="utf-8")

    payload = tmp_path / "sent-payload.json"
    os.environ["GH_STUB_LOG"] = str(log)
    os.environ["GH_STUB_COMMENTS"] = str(comments_fixture)
    os.environ["GH_STUB_PAYLOAD"] = str(payload)
    try:
        result = _execute_action(
            tmp_path,
            fail_on="none",
            report_format="text",
            pr_comment="true",
            github_token="fake-token",
            event_path=event,
            repository="tjdonley/rag-blast-radius",
            extra_path=stub_dir,
        )
    finally:
        os.environ.pop("GH_STUB_LOG", None)
        os.environ.pop("GH_STUB_COMMENTS", None)
        os.environ.pop("GH_STUB_PAYLOAD", None)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::" not in result.stdout, result.stdout

    entries = log.read_text(encoding="utf-8")
    calls = [line[5:] for line in entries.splitlines() if line.startswith("CALL ")]
    assert payload.exists(), "no comment payload was sent"
    return calls, json.loads(payload.read_text(encoding="utf-8"))["body"]
