"""Shared types for Cognix proactive intelligence detectors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from app.database import db_session, utc_now

FindingType = Literal["gap", "contradiction", "contradiction_candidate", "staleness"]
Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class SourceRef:
    """A source pointer for an intelligence finding."""

    source_path: str
    chunk_id: int | None = None
    claim_id: int | None = None
    excerpt: str = ""


@dataclass(frozen=True)
class Finding:
    """A proactive knowledge-auditing finding."""

    finding_type: FindingType
    severity: Severity
    title: str
    description: str
    source_refs: list[SourceRef] = field(default_factory=list)
    suggested_action: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector:
    """Base protocol for intelligence detectors."""

    def run(self, since: datetime | None = None) -> list[Finding]:
        """Run the detector and return findings."""
        raise NotImplementedError


def store_findings(findings: list[Finding]) -> int:
    """Persist findings before any wiki/UI rendering occurs."""
    if not findings:
        return 0
    now = utc_now()
    with db_session() as conn:
        existing = {
            (row["finding_type"], row["title"], row["description"])
            for row in conn.execute(
                """
                SELECT finding_type, title, description
                FROM intelligence_findings
                WHERE status='open'
                """
            ).fetchall()
        }
        new_findings = [
            finding
            for finding in findings
            if (finding.finding_type, finding.title, finding.description) not in existing
        ]
        if not new_findings:
            return 0
        conn.executemany(
            """
            INSERT INTO intelligence_findings
            (finding_type, severity, title, description, source_refs_json,
             suggested_action, status, confidence, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
            """,
            [
                (
                    finding.finding_type,
                    finding.severity,
                    finding.title,
                    finding.description,
                    json.dumps([ref.__dict__ for ref in finding.source_refs], sort_keys=True),
                    finding.suggested_action,
                    finding.confidence,
                    json.dumps(finding.metadata, sort_keys=True),
                    now,
                    now,
                )
                for finding in new_findings
            ],
        )
    return len(new_findings)
