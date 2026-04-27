import json

from typer.testing import CliRunner

from rag_blast.cli import app
from rag_blast.manifest import starter_manifest

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
    assert report["change_count"] == 1
    assert report["changes"][0]["path"] == "embedding.model"


def test_cli_check_rejects_invalid_format(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", "--old", str(manifest_path), "--new", str(manifest_path), "--format", "xml"],
    )

    assert result.exit_code == 1
    assert "Unsupported format" in result.output


def test_cli_check_rejects_malformed_json(tmp_path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text("{", encoding="utf-8")
    new_path.write_text(json.dumps(starter_manifest()), encoding="utf-8")

    result = runner.invoke(app, ["check", "--old", str(old_path), "--new", str(new_path)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_cli_explain_known_rule() -> None:
    result = runner.invoke(app, ["explain", "REEMBED_REQUIRED"])

    assert result.exit_code == 0
    assert "REEMBED_REQUIRED" in result.output
