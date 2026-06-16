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
        totals = {
            "files": conn.execute("SELECT COUNT(*) AS count FROM raw_files").fetchone()["count"],
            "chunks": conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"],
            "outputs": conn.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()["count"],
            "errors": conn.execute("SELECT COUNT(*) AS count FROM ingest_errors").fetchone()["count"],
            "findings": conn.execute("SELECT COUNT(*) AS count FROM health_findings WHERE status='open'").fetchone()["count"],
        }
        findings = conn.execute("SELECT * FROM health_findings WHERE status='open' ORDER BY id DESC LIMIT 50").fetchall()
    return {"score": _score(totals), "totals": totals, "findings": findings}


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
        lines.append(f"- **{finding['severity']}** {finding['title']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

