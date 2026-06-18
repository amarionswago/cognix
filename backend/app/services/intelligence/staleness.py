"""Staleness detection for Cognix v2."""

from __future__ import annotations

from datetime import datetime

from app.database import db_session
from app.services.intelligence.types import Detector, Finding, SourceRef

STALE_AFTER_DAYS = 180


class StalenessDetector(Detector):
    """Flag old claims that may need review against newer material."""

    def run(self, since: datetime | None = None) -> list[Finding]:
        """Return staleness candidates based on claim age and related newer claims."""
        with db_session() as conn:
            rows = conn.execute(
                """
                SELECT old.id AS old_claim_id, old.chunk_id AS old_chunk_id,
                       old.claim_text AS old_claim, old.ingest_date AS old_date,
                       old_chunks.source_path AS old_source,
                       newer.id AS newer_claim_id, newer.chunk_id AS newer_chunk_id,
                       newer.claim_text AS newer_claim, newer.ingest_date AS newer_date,
                       newer_chunks.source_path AS newer_source
                FROM claims old
                JOIN claims newer ON newer.id != old.id
                JOIN chunks old_chunks ON old_chunks.id = old.chunk_id
                JOIN chunks newer_chunks ON newer_chunks.id = newer.chunk_id
                WHERE old.status='active'
                  AND newer.status='active'
                  AND old.ingest_date < newer.ingest_date
                  AND julianday('now') - julianday(old.ingest_date) >= ?
                  AND substr(lower(old.claim_text), 1, 24) = substr(lower(newer.claim_text), 1, 24)
                ORDER BY old.ingest_date ASC
                LIMIT 20
                """,
                (STALE_AFTER_DAYS,),
            ).fetchall()

        findings: list[Finding] = []
        for row in rows:
            findings.append(
                Finding(
                    finding_type="staleness",
                    severity="info",
                    title="Stale claim candidate",
                    description=(
                        f"Older claim may need review: {row['old_claim']}\n\n"
                        f"Newer related claim: {row['newer_claim']}"
                    ),
                    source_refs=[
                        SourceRef(row["old_source"], int(row["old_chunk_id"]), int(row["old_claim_id"]), row["old_claim"]),
                        SourceRef(row["newer_source"], int(row["newer_chunk_id"]), int(row["newer_claim_id"]), row["newer_claim"]),
                    ],
                    suggested_action="Review whether the older claim should be updated, marked stale, or kept.",
                    confidence=0.55,
                    metadata={"old_date": row["old_date"], "newer_date": row["newer_date"]},
                )
            )
        return findings
