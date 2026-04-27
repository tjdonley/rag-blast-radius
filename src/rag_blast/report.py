from __future__ import annotations

from typing import Any

from rag_blast.diff import ManifestDiff


def build_report(manifest_diff: ManifestDiff) -> dict[str, Any]:
    """Build the manifest diff report payload."""
    return {
        "risk": "UNASSESSED" if manifest_diff.change_count else "NONE",
        "change_count": manifest_diff.change_count,
        "categories": list(manifest_diff.categories),
        "changes": [change.to_dict() for change in manifest_diff.changes],
        "note": "Current reports list categorized manifest changes. Risk rules are added in later phases.",
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

    lines.extend(["", str(report["note"])])
    return "\n".join(lines)
