from __future__ import annotations

import html
import json
from typing import Any

from rag_blast.diff import ManifestDiff
from rag_blast.rules import RuleFinding, SEVERITY_ORDER, evaluate_rules, highest_severity

FAIL_ON_VALUES = frozenset({"none", "low", "medium", "high"})
ROLLOUT_STEPS = {
    "REEMBED_REQUIRED": "Regenerate document embeddings for the proposed manifest.",
    "VECTOR_INDEX_INCOMPATIBLE": "Build a shadow vector index before serving new query embeddings.",
    "SEMANTIC_CACHE_UNSAFE": "Use a new semantic cache namespace before rollout.",
    "RETRIEVAL_BASELINE_STALE": "Replay representative retrieval evals and compare against baseline.",
    "CHUNKING_CHANGED": "Regenerate chunks and rebuild affected indexes.",
    "RERANKER_CHANGED": "Replay answer-quality evals with the proposed reranker behavior.",
    "RETRIEVER_BEHAVIOR_CHANGED": "Compare retrieval overlap and answer quality before rollout.",
    "SHADOW_INDEX_RECOMMENDED": "Canary or shadow traffic before switching production reads.",
    "ROLLBACK_REQUIRES_OLD_INDEX": "Keep the old index and cache namespace until the rollback window closes.",
}


def build_report(manifest_diff: ManifestDiff) -> dict[str, Any]:
    """Build the manifest diff report payload."""
    findings = evaluate_rules(manifest_diff)
    unassessed_change_paths = _unassessed_change_paths(manifest_diff, findings)
    return {
        "risk": _report_risk(manifest_diff, findings),
        "change_count": manifest_diff.change_count,
        "categories": list(manifest_diff.categories),
        "changes": [change.to_dict() for change in manifest_diff.changes],
        "finding_count": len(findings),
        "findings": [finding.to_dict() for finding in findings],
        "unassessed_change_count": len(unassessed_change_paths),
        "unassessed_change_paths": list(unassessed_change_paths),
        "recommended_rollout": _recommended_rollout(
            manifest_diff, findings, unassessed_change_paths
        ),
        "note": "Risk is based on deterministic local rules.",
    }


def render_text_report(report: dict[str, Any]) -> str:
    """Render a human-readable manifest diff report."""
    lines = [
        "RAG BLAST RADIUS REPORT",
        "",
        f"Risk: {report['risk']}",
        "",
        "Detected changes:",
    ]

    changes = report["changes"]
    if not changes:
        lines.append("  - none")
    else:
        for change in changes:
            lines.append(
                f"  - {change['path']} ({change['category']}): "
                f"{change['summary']}; {change['old']} -> {change['new']}"
            )

    lines.extend(["", "Invalidation rules triggered:"])
    findings = report["findings"]
    if not findings:
        lines.append("  - none")
    else:
        for finding in findings:
            lines.append(f"  - {finding['severity']}: {finding['rule_id']} - {finding['summary']}")

    unassessed_change_paths = report["unassessed_change_paths"]
    if unassessed_change_paths:
        lines.extend(["", "Unassessed changes:"])
        for path in unassessed_change_paths:
            lines.append(f"  - {path}")

    rollout_steps = report["recommended_rollout"]
    if rollout_steps:
        lines.extend(["", "Recommended rollout:"])
        for index, step in enumerate(rollout_steps, start=1):
            lines.append(f"  {index}. {step}")

    lines.extend(["", str(report["note"])])
    return "\n".join(lines)


def render_json_report(report: dict[str, Any]) -> str:
    """Render a machine-readable report."""
    return json.dumps(report, indent=2)


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a GitHub-friendly Markdown report."""
    lines = [
        "## RAG Blast Radius",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Risk | {_markdown_code(report['risk'])} |",
        f"| Changes | {_markdown_code(report['change_count'])} |",
        f"| Findings | {_markdown_code(report['finding_count'])} |",
        f"| Unassessed changes | {_markdown_code(report['unassessed_change_count'])} |",
        "",
        "### Detected Changes",
    ]

    changes = report["changes"]
    if not changes:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| Path | Category | Summary | Old | New |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for change in changes:
            lines.append(
                "| "
                f"{_markdown_code(change['path'])} | "
                f"{_markdown_code(change['category'])} | "
                f"{_markdown_table_cell(change['summary'])} | "
                f"{_markdown_table_cell(change['old'])} | "
                f"{_markdown_table_cell(change['new'])} |"
            )

    lines.extend(["", "### Findings"])
    findings = report["findings"]
    if not findings:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| Severity | Rule | Summary | Change Paths |",
                "| --- | --- | --- | --- |",
            ]
        )
        for finding in findings:
            change_paths = ", ".join(_markdown_code(path) for path in finding["change_paths"])
            lines.append(
                "| "
                f"{_markdown_code(finding['severity'])} | "
                f"{_markdown_code(finding['rule_id'])} | "
                f"{_markdown_table_cell(finding['summary'])} | "
                f"{change_paths or 'none'} |"
            )

    unassessed_change_paths = report["unassessed_change_paths"]
    if unassessed_change_paths:
        lines.extend(["", "### Unassessed Changes"])
        for path in unassessed_change_paths:
            lines.append(f"- {_markdown_code(path)}")

    rollout_steps = report["recommended_rollout"]
    if rollout_steps:
        lines.extend(["", "### Recommended Rollout"])
        for index, step in enumerate(rollout_steps, start=1):
            lines.append(f"{index}. {_markdown_text(step)}")

    lines.extend(["", str(report["note"])])
    return "\n".join(lines)


def normalize_fail_on(value: str) -> str | None:
    """Normalize a fail-on threshold, or return None when invalid."""
    normalized = value.lower()
    if normalized not in FAIL_ON_VALUES:
        return None
    return normalized


def should_fail_report(report: dict[str, Any], fail_on: str) -> bool:
    """Return whether a report should produce a failing exit code."""
    if fail_on == "none":
        return False

    if report["unassessed_change_count"]:
        return True

    risk = str(report["risk"])
    if risk == "UNASSESSED":
        return bool(report["change_count"])
    if risk == "NONE":
        return False

    return SEVERITY_ORDER[risk] >= SEVERITY_ORDER[fail_on.upper()]


def _report_risk(manifest_diff: ManifestDiff, findings: tuple[RuleFinding, ...]) -> str:
    if findings:
        return highest_severity(findings)
    if manifest_diff.change_count:
        return "UNASSESSED"
    return "NONE"


def _recommended_rollout(
    manifest_diff: ManifestDiff,
    findings: tuple[RuleFinding, ...],
    unassessed_change_paths: tuple[str, ...],
) -> list[str]:
    steps: list[str] = []
    for finding in findings:
        step = ROLLOUT_STEPS.get(finding.rule_id, finding.recommendation)
        if step not in steps:
            steps.append(step)

    if unassessed_change_paths:
        steps.append("Review unassessed manifest changes before deployment.")
    elif not steps and manifest_diff.change_count:
        steps.append("Review unassessed manifest changes before deployment.")

    return steps


def _unassessed_change_paths(
    manifest_diff: ManifestDiff, findings: tuple[RuleFinding, ...]
) -> tuple[str, ...]:
    assessed_paths = {path for finding in findings for path in finding.change_paths}
    return tuple(
        change.path for change in manifest_diff.changes if change.path not in assessed_paths
    )


def _markdown_table_cell(value: Any) -> str:
    return _markdown_text(value).replace("\n", "<br>").replace("|", r"\|")


def _markdown_code(value: Any) -> str:
    text = _markdown_raw_text(value)
    escaped = html.escape(text, quote=False)
    escaped = escaped.replace("\n", "<br>").replace("|", "&#124;").replace("`", "&#96;")
    return f"<code>{escaped}</code>"


def _markdown_text(value: Any) -> str:
    text = _markdown_raw_text(value)
    return html.escape(text.replace("`", r"\`"), quote=False)


def _markdown_raw_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    return str(value)
