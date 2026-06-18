"""Orchestrates Cognix v2 intelligence passes."""

from __future__ import annotations

import json
from datetime import datetime

from app.database import db_session, utc_now
from app.services.intelligence.briefing import generate_intelligence_brief
from app.services.intelligence.contradiction import ContradictionDetector
from app.services.intelligence.gaps import GapDetector
from app.services.intelligence.staleness import StalenessDetector
from app.services.intelligence.types import Finding, store_findings


def run_intelligence_pass(run_type: str = "manual", since: datetime | None = None, use_llm: bool = True) -> dict:
    """Run gap, contradiction, staleness, and briefing generation."""
    run_id = start_run(run_type)
    all_findings: list[Finding] = []
    try:
        detectors = [GapDetector(), ContradictionDetector(use_llm=use_llm), StalenessDetector()]
        for detector in detectors:
            all_findings.extend(detector.run(since))
        stored = store_findings(all_findings)
        briefing_id, briefing_path = generate_intelligence_brief()
        finish_run(
            run_id,
            status="completed",
            findings_created=stored,
            briefings_created=1,
            metadata={"briefing_id": briefing_id, "briefing_path": str(briefing_path)},
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "findings_created": stored,
            "briefing_id": briefing_id,
            "briefing_path": str(briefing_path),
        }
    except Exception as exc:
        finish_run(run_id, status="failed", error=str(exc), metadata={})
        raise


def start_run(run_type: str) -> int:
    """Create an intelligence run log row."""
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO intelligence_runs
            (run_type, status, started_at)
            VALUES (?, 'running', ?)
            """,
            (run_type, utc_now()),
        )
        return int(cursor.lastrowid)


def finish_run(
    run_id: int,
    status: str,
    findings_created: int = 0,
    briefings_created: int = 0,
    error: str = "",
    metadata: dict | None = None,
) -> None:
    """Mark an intelligence run as finished."""
    with db_session() as conn:
        conn.execute(
            """
            UPDATE intelligence_runs
            SET status=?, finished_at=?, findings_created=?, briefings_created=?,
                error=?, metadata_json=?
            WHERE id=?
            """,
            (
                status,
                utc_now(),
                findings_created,
                briefings_created,
                error,
                json.dumps(metadata or {}, sort_keys=True),
                run_id,
            ),
        )
