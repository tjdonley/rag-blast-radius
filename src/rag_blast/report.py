from __future__ import annotations

from typing import Any

from rag_blast.diff import ManifestChange


def build_report(changes: list[ManifestChange]) -> dict[str, Any]:
    """Build the Phase 1 report payload."""
    return {
        "risk": "UNASSESSED" if changes else "NONE",
        "change_count": len(changes),
        "changes": [change.to_dict() for change in changes],
        "note": "Current reports list raw manifest changes. Risk rules are added in later phases.",
    }


def render_text_report(report: dict[str, Any]) -> str:
    """Render a human-readable Phase 1 report."""
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
            lines.append(f"  - {change['path']}: {change['old']} -> {change['new']}")

    lines.extend(["", str(report["note"])])
    return "\n".join(lines)
