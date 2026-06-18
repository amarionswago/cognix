"""Intelligence Brief generation for Cognix v2."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now


def generate_intelligence_brief(brief_date: date | None = None) -> tuple[int, Path]:
    """Generate a markdown Intelligence Brief from stored SQLite findings."""
    settings = get_settings()
    day = brief_date or date.today()
    brief_dir = settings.resolved_wiki_dir() / "_intelligence"
    brief_dir.mkdir(parents=True, exist_ok=True)
    path = brief_dir / f"intelligence-brief-{day.isoformat()}.md"

    with db_session() as conn:
        findings = conn.execute(
            """
            SELECT *
            FROM intelligence_findings
            WHERE status='open'
            ORDER BY
                CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                id DESC
            LIMIT 100
            """
        ).fetchall()
        totals = {
            "files": conn.execute("SELECT COUNT(*) AS count FROM raw_files").fetchone()["count"],
            "chunks": conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"],
            "claims": conn.execute("SELECT COUNT(*) AS count FROM claims").fetchone()["count"],
            "concepts": conn.execute("SELECT COUNT(DISTINCT normalized_concept) AS count FROM concept_mentions").fetchone()["count"],
        }

    counts = Counter(str(finding["finding_type"]) for finding in findings)
    content = render_brief(day, totals, findings, counts)
    path.write_text(content, encoding="utf-8")
    summary = brief_summary(counts)
    now = utc_now()
    with db_session() as conn:
        existing = conn.execute("SELECT id FROM briefings WHERE brief_date=?", (day.isoformat(),)).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE briefings
                SET title=?, path=?, summary=?, finding_counts_json=?, status='generated', updated_at=?
                WHERE id=?
                """,
                (
                    f"Cognix Intelligence Brief — {day.isoformat()}",
                    str(path),
                    summary,
                    json.dumps(dict(counts), sort_keys=True),
                    now,
                    existing["id"],
                ),
            )
            return int(existing["id"]), path
        cursor = conn.execute(
            """
            INSERT INTO briefings
            (brief_date, title, path, summary, finding_counts_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'generated', ?, ?)
            """,
            (
                day.isoformat(),
                f"Cognix Intelligence Brief — {day.isoformat()}",
                str(path),
                summary,
                json.dumps(dict(counts), sort_keys=True),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid), path


def render_brief(day: date, totals: dict, findings: list[dict], counts: Counter) -> str:
    """Render markdown for the Intelligence Brief."""
    top_findings = prioritized_findings(findings)[:5]
    lines = [
        f"# Cognix Intelligence Brief — {day.isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"- Open contradictions: {counts.get('contradiction', 0)}",
        f"- Contradiction candidates: {counts.get('contradiction_candidate', 0)}",
        f"- Knowledge gaps: {counts.get('gap', 0)}",
        f"- Stale claim candidates: {counts.get('staleness', 0)}",
        f"- Claims indexed: {totals.get('claims', 0)}",
        f"- Concepts tracked: {totals.get('concepts', 0)}",
        "",
        "## Priority Review Queue",
        "",
    ]
    if top_findings:
        for index, finding in enumerate(top_findings, start=1):
            metadata = parse_json_field(finding.get("metadata_json"))
            trigger = finding_trigger(finding, metadata)
            lines.append(f"{index}. **{finding['severity']}** {finding['title']}")
            lines.append(f"   - Trigger: {trigger}")
            lines.append(f"   - Action: {finding.get('suggested_action') or 'Review the cited evidence.'}")
        lines.append("")
    else:
        lines.extend(["- No open findings need review.", ""])
    lines.extend(
        [
        "## Library Pulse",
        "",
        f"- Files: {totals.get('files', 0)}",
        f"- Chunks: {totals.get('chunks', 0)}",
        "",
        ]
    )
    append_finding_section(lines, "Contradictions", findings, {"contradiction", "contradiction_candidate"})
    append_finding_section(lines, "Knowledge Gaps", findings, {"gap"})
    append_finding_section(lines, "Stale Claims", findings, {"staleness"})
    lines.extend(
        [
            "## Suggested Research Actions",
            "",
            "- Review high-severity contradictions first.",
            "- Compile the highest-frequency knowledge gaps into wiki concept pages.",
            "- Promote useful answers so Cognix can collect future fine-tuning data.",
            "",
        ]
    )
    return "\n".join(lines)


def append_finding_section(lines: list[str], title: str, findings: list[dict], finding_types: set[str]) -> None:
    """Append a markdown section for selected finding types."""
    selected = [finding for finding in findings if finding["finding_type"] in finding_types]
    lines.extend([f"## {title}", ""])
    if not selected:
        lines.extend(["- No open findings.", ""])
        return
    for finding in selected[:20]:
        metadata = parse_json_field(finding.get("metadata_json"))
        refs = parse_json_field(finding.get("source_refs_json"))
        lines.append(f"- **{finding['severity']}** {finding['title']}: {finding['description'].splitlines()[0]}")
        lines.append(f"  - Trigger: {finding_trigger(finding, metadata)}")
        if isinstance(refs, list) and refs:
            source_paths = []
            for ref in refs[:3]:
                if isinstance(ref, dict) and ref.get("source_path"):
                    source_paths.append(str(ref["source_path"]))
            if source_paths:
                lines.append(f"  - Sources: {', '.join(source_paths)}")
        if finding.get("suggested_action"):
            lines.append(f"  - Action: {finding['suggested_action']}")
    lines.append("")


def brief_summary(counts: Counter) -> str:
    """Return a compact stored summary for the briefings table."""
    return (
        f"{counts.get('gap', 0)} gaps, "
        f"{counts.get('contradiction', 0)} contradictions, "
        f"{counts.get('contradiction_candidate', 0)} candidates, "
        f"{counts.get('staleness', 0)} stale claim candidates"
    )


def prioritized_findings(findings: list[dict]) -> list[dict]:
    """Sort findings by product review priority."""
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    type_rank = {"contradiction": 0, "contradiction_candidate": 1, "staleness": 2, "gap": 3}
    return sorted(
        findings,
        key=lambda finding: (
            severity_rank.get(str(finding["severity"]), 3),
            type_rank.get(str(finding["finding_type"]), 4),
            -float(finding.get("confidence") or 0),
        ),
    )


def finding_trigger(finding: dict, metadata: dict) -> str:
    """Explain why a finding appeared."""
    finding_type = str(finding["finding_type"])
    if finding_type == "gap":
        return f"mentioned {metadata.get('mention_count', 'multiple')} times without a wiki concept page"
    if finding_type in {"contradiction", "contradiction_candidate"}:
        return f"claim similarity {metadata.get('similarity', finding.get('confidence', 0))}, verdict {metadata.get('verdict', 'candidate')}"
    if finding_type == "staleness":
        return "older claim has related newer evidence"
    return "open intelligence finding"


def parse_json_field(raw: object) -> dict | list:
    """Parse a JSON database field without breaking brief generation."""
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, (dict, list)) else {}
