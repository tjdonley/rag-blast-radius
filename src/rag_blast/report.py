from __future__ import annotations

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
    return {
        "risk": _report_risk(manifest_diff, findings),
        "change_count": manifest_diff.change_count,
        "categories": list(manifest_diff.categories),
        "changes": [change.to_dict() for change in manifest_diff.changes],
        "finding_count": len(findings),
        "findings": [finding.to_dict() for finding in findings],
        "recommended_rollout": _recommended_rollout(manifest_diff, findings),
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
    manifest_diff: ManifestDiff, findings: tuple[RuleFinding, ...]
) -> list[str]:
    steps: list[str] = []
    for finding in findings:
        step = ROLLOUT_STEPS.get(finding.rule_id, finding.recommendation)
        if step not in steps:
            steps.append(step)

    if not steps and manifest_diff.change_count:
        steps.append("Review unassessed manifest changes before deployment.")

    return steps
