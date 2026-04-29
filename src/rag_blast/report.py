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


def render_html_report(report: dict[str, Any]) -> str:
    """Render a standalone static HTML report."""
    risk = _html_text(report["risk"])
    changes = report["changes"]
    findings = report["findings"]
    unassessed_change_paths = report["unassessed_change_paths"]
    rollout_steps = report["recommended_rollout"]
    raw_json = html.escape(json.dumps(report, indent=2), quote=False)

    changes_html = _html_changes_table(changes)
    findings_html = _html_findings_table(findings)
    unassessed_html = _html_unassessed_changes(unassessed_change_paths)
    rollout_html = _html_rollout_steps(rollout_steps)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Blast Radius Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f4ef;
      --panel: #fffaf2;
      --ink: #201a14;
      --muted: #6f6257;
      --line: #ded4c8;
      --accent: #5d4bff;
      --danger: #b42318;
      --warn: #b76e00;
      --ok: #1f7a4d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 56px; }}
    header {{
      border: 1px solid var(--line);
      background: linear-gradient(135deg, #fffaf2 0%, #eeeaff 100%);
      border-radius: 24px;
      padding: 32px;
      box-shadow: 0 20px 60px rgba(32, 26, 20, 0.08);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 5vw, 4rem); line-height: 0.95; }}
    h2 {{ margin: 36px 0 14px; font-size: 1.35rem; }}
    p {{ margin: 0; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); background: var(--panel); }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    code, pre {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }}
    code {{ font-size: 0.92em; color: var(--accent); }}
    pre {{ overflow: auto; padding: 16px; border-radius: 14px; background: #15131a; color: #fffaf2; }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    .lede {{ color: var(--muted); max-width: 760px; }}
    .risk {{ display: inline-flex; gap: 10px; align-items: center; margin-bottom: 18px; }}
    .risk-badge {{
      border-radius: 999px;
      padding: 6px 12px;
      background: {_html_risk_color(report["risk"])};
      color: #fff;
      font-weight: 800;
      letter-spacing: 0.08em;
    }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 24px; }}
    .card {{ border: 1px solid var(--line); background: rgba(255, 250, 242, 0.78); border-radius: 18px; padding: 16px; }}
    .card span {{ display: block; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }}
    .card strong {{ display: block; margin-top: 8px; font-size: 1.8rem; }}
    .empty {{ border: 1px dashed var(--line); background: rgba(255, 250, 242, 0.58); border-radius: 16px; padding: 18px; color: var(--muted); }}
    .section {{ margin-top: 30px; }}
    .rollout {{ margin: 0; padding-left: 24px; }}
    .rollout li {{ margin: 10px 0; }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 12px 36px; }}
      header {{ padding: 24px; border-radius: 18px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="risk"><span class="risk-badge">{risk}</span><span>Deterministic local rules</span></div>
      <h1>RAG Blast Radius Report</h1>
      <p class="lede">{_html_text(report["note"])}</p>
      <div class="cards" aria-label="Report summary">
        <div class="card"><span>Risk</span><strong>{risk}</strong></div>
        <div class="card"><span>Changes</span><strong>{_html_text(report["change_count"])}</strong></div>
        <div class="card"><span>Findings</span><strong>{_html_text(report["finding_count"])}</strong></div>
        <div class="card"><span>Unassessed</span><strong>{_html_text(report["unassessed_change_count"])}</strong></div>
      </div>
    </header>
    <section class="section" aria-labelledby="changes-heading">
      <h2 id="changes-heading">Detected Changes</h2>
      {changes_html}
    </section>
    <section class="section" aria-labelledby="findings-heading">
      <h2 id="findings-heading">Findings</h2>
      {findings_html}
    </section>
    <section class="section" aria-labelledby="unassessed-heading">
      <h2 id="unassessed-heading">Unassessed Changes</h2>
      {unassessed_html}
    </section>
    <section class="section" aria-labelledby="rollout-heading">
      <h2 id="rollout-heading">Recommended Rollout</h2>
      {rollout_html}
    </section>
    <section class="section" aria-labelledby="raw-json-heading">
      <h2 id="raw-json-heading">Raw JSON</h2>
      <details>
        <summary>Show report payload</summary>
        <pre>{raw_json}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


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


def _html_changes_table(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return '<p class="empty">No manifest changes detected.</p>'

    rows = []
    for change in changes:
        rows.append(
            "<tr>"
            f"<td><code>{_html_text(change['path'])}</code></td>"
            f"<td><code>{_html_text(change['category'])}</code></td>"
            f"<td>{_html_text(change['summary'])}</td>"
            f"<td>{_html_value(change['old'])}</td>"
            f"<td>{_html_value(change['new'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Path</th><th>Category</th><th>Summary</th><th>Old</th><th>New</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _html_findings_table(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return '<p class="empty">No invalidation rules triggered.</p>'

    rows = []
    for finding in findings:
        change_paths = ", ".join(
            f"<code>{_html_text(path)}</code>" for path in finding["change_paths"]
        )
        rows.append(
            "<tr>"
            f"<td><code>{_html_text(finding['severity'])}</code></td>"
            f"<td><code>{_html_text(finding['rule_id'])}</code></td>"
            f"<td>{_html_text(finding['summary'])}</td>"
            f"<td>{change_paths or 'none'}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Severity</th><th>Rule</th><th>Summary</th><th>Change Paths</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _html_unassessed_changes(paths: list[str]) -> str:
    if not paths:
        return '<p class="empty">No unassessed changes.</p>'
    items = "".join(f"<li><code>{_html_text(path)}</code></li>" for path in paths)
    return f"<ul>{items}</ul>"


def _html_rollout_steps(steps: list[str]) -> str:
    if not steps:
        return '<p class="empty">No rollout steps required.</p>'
    items = "".join(f"<li>{_html_text(step)}</li>" for step in steps)
    return f'<ol class="rollout">{items}</ol>'


def _html_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return f"<code>{_html_text(json.dumps(value, sort_keys=True))}</code>"
    return f"<code>{_html_text(value)}</code>"


def _html_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        text = json.dumps(value)
    else:
        text = str(value)
    return html.escape(text, quote=True)


def _html_risk_color(risk: Any) -> str:
    return {
        "HIGH": "var(--danger)",
        "MEDIUM": "var(--warn)",
        "LOW": "var(--accent)",
        "NONE": "var(--ok)",
        "UNASSESSED": "var(--warn)",
    }.get(str(risk), "var(--muted)")
