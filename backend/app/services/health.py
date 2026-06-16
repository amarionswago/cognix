from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now


REQUIRED_WIKI_DIRS = [
    "personal",
    "work",
    "code",
    "health",
    "finance",
    "learning",
    "media",
    "people",
    "places",
    "concepts",
    "sources",
    "outputs/analysis",
    "outputs/reports",
    "outputs/slides",
    "outputs/charts",
    "outputs/diagrams",
    "outputs/tables",
    "contradictions",
    "decisions",
    "datasets",
    "timelines",
    "_indexes",
    "_health",
]


def run_health_check() -> dict:
    settings = get_settings()
    settings.ensure_directories()
    findings: list[dict] = []
    for relative in REQUIRED_WIKI_DIRS:
        path = settings.resolved_wiki_dir() / relative
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            findings.append(
                {
                    "severity": "info",
                    "category": "structure",
                    "title": "Created missing wiki directory",
                    "message": f"Created {relative}.",
                    "path": str(path),
                }
            )

    with db_session() as conn:
        findings.extend(_database_findings(conn))
        conn.execute("DELETE FROM health_findings WHERE status='open'")
        for finding in findings:
            conn.execute(
                """
                INSERT INTO health_findings
                (severity, category, title, message, path, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    finding["severity"],
                    finding["category"],
                    finding["title"],
                    finding["message"],
                    finding.get("path"),
                    utc_now(),
                ),
            )
        totals = {
            "files": conn.execute("SELECT COUNT(*) AS count FROM raw_files").fetchone()["count"],
            "chunks": conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"],
            "outputs": conn.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()["count"],
            "errors": conn.execute("SELECT COUNT(*) AS count FROM ingest_errors").fetchone()["count"],
            "findings": conn.execute("SELECT COUNT(*) AS count FROM health_findings WHERE status='open'").fetchone()["count"],
        }
        open_findings = conn.execute("SELECT * FROM health_findings WHERE status='open' ORDER BY id DESC LIMIT 50").fetchall()

    score = _score(totals)
    _write_health_report(settings.resolved_wiki_dir() / "_health" / "latest-health-report.md", score, totals, open_findings)
    return {"score": score, "totals": totals, "findings": open_findings}


def health_summary() -> dict:
    with db_session() as conn:
        live_findings = _database_findings(conn)
        if live_findings:
            conn.execute("DELETE FROM health_findings WHERE status='open'")
            for finding in live_findings:
                conn.execute(
                    """
                    INSERT INTO health_findings
                    (severity, category, title, message, path, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        finding["severity"],
                        finding["category"],
                        finding["title"],
                        finding["message"],
                        finding.get("path"),
                        utc_now(),
                    ),
                )
        totals = {
            "files": conn.execute("SELECT COUNT(*) AS count FROM raw_files").fetchone()["count"],
            "chunks": conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"],
            "outputs": conn.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()["count"],
            "errors": conn.execute("SELECT COUNT(*) AS count FROM ingest_errors").fetchone()["count"],
            "findings": conn.execute("SELECT COUNT(*) AS count FROM health_findings WHERE status='open'").fetchone()["count"],
        }
        findings = conn.execute("SELECT * FROM health_findings WHERE status='open' ORDER BY id DESC LIMIT 50").fetchall()
    return {"score": _score(totals), "totals": totals, "findings": findings}


def _database_findings(conn) -> list[dict]:
    findings: list[dict] = []
    failed_files = conn.execute(
        """
        SELECT relative_path, error_count
        FROM raw_files
        WHERE status='failed' OR error_count > 0
        ORDER BY error_count DESC, relative_path
        LIMIT 20
        """
    ).fetchall()
    for row in failed_files:
        findings.append(
            {
                "severity": "error",
                "category": "ingest",
                "title": "File has ingest failures",
                "message": f"{row['relative_path']} has status/error count requiring attention.",
                "path": row["relative_path"],
            }
        )

    recent_errors = conn.execute(
        """
        SELECT ie.path, ie.error_type, ie.message, ie.created_at,
               COALESCE(rf.status, 'missing') AS current_status
        FROM ingest_errors ie
        LEFT JOIN raw_files rf ON rf.path = ie.path
        ORDER BY ie.id DESC
        LIMIT 10
        """
    ).fetchall()
    for row in recent_errors:
        severity = "warning" if row["current_status"] == "processed" else "error"
        findings.append(
            {
                "severity": severity,
                "category": "ingest",
                "title": "Recent ingest error",
                "message": f"{row['error_type']}: {row['message']} Current status: {row['current_status']}.",
                "path": row["path"],
            }
        )

    unchunked = conn.execute(
        """
        SELECT rf.relative_path
        FROM raw_files rf
        LEFT JOIN chunks c ON c.file_id = rf.id
        WHERE rf.status='processed'
        GROUP BY rf.id
        HAVING COUNT(c.id)=0
        LIMIT 20
        """
    ).fetchall()
    for row in unchunked:
        findings.append(
            {
                "severity": "warning",
                "category": "retrieval",
                "title": "Processed file has no searchable chunks",
                "message": "The file is marked processed but will not be retrievable until chunks exist.",
                "path": row["relative_path"],
            }
        )

    large_sources = conn.execute(
        """
        SELECT rf.relative_path, ed.word_count, ed.extraction_method
        FROM raw_files rf
        JOIN extracted_documents ed ON ed.file_id = rf.id
        WHERE ed.word_count >= 30000
        ORDER BY ed.word_count DESC
        LIMIT 10
        """
    ).fetchall()
    for row in large_sources:
        findings.append(
            {
                "severity": "info",
                "category": "large_source",
                "title": "Large document processed",
                "message": f"{row['word_count']} words extracted with {row['extraction_method']}.",
                "path": row["relative_path"],
            }
        )

    return findings[:50]


def _score(totals: dict[str, int]) -> int:
    score = 100
    score -= min(30, totals.get("errors", 0) * 5)
    score -= min(25, totals.get("findings", 0) * 3)
    if totals.get("files", 0) and not totals.get("chunks", 0):
        score -= 20
    return max(0, score)


def _write_health_report(path: Path, score: int, totals: dict[str, int], findings: list[dict]) -> None:
    lines = ["# Cognix Health Report", "", f"Score: {score}", "", "## Totals", ""]
    for key, value in totals.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Open Findings", ""])
    if not findings:
        lines.append("- No open findings.")
    for finding in findings:
        suffix = f" `{finding['path']}`" if finding.get("path") else ""
        lines.append(f"- **{finding['severity']}** {finding['title']}: {finding['message']}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
