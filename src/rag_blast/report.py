from __future__ import annotations

from typing import Any

from rag_blast.diff import ManifestDiff
from rag_blast.rules import RuleFinding, evaluate_rules, highest_severity


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

    if findings:
        lines.extend(["", "Recommended actions:"])
        for finding in findings:
            lines.append(f"  - {finding['rule_id']}: {finding['recommendation']}")

    lines.extend(["", str(report["note"])])
    return "\n".join(lines)


def _report_risk(manifest_diff: ManifestDiff, findings: tuple[RuleFinding, ...]) -> str:
    if findings:
        return highest_severity(findings)
    if manifest_diff.change_count:
        return "UNASSESSED"
    return "NONE"
