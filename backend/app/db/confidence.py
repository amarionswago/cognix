"""Evidence-confidence scoring for Cognix answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db.calibration import calibrated_probability
from app.database import db_session, utc_now

SOURCE_COUNT_WEIGHT = 0.30
SOURCE_DIVERSITY_WEIGHT = 0.25
RETRIEVAL_SCORE_WEIGHT = 0.25
RECENCY_WEIGHT = 0.15
CONTRADICTION_WEIGHT = 0.05


@dataclass(frozen=True)
class ConfidenceResult:
    """Computed evidence-confidence score with auditable components."""

    score: float
    label: str
    breakdown: dict[str, Any]


@dataclass(frozen=True)
class EvidenceSource:
    """Minimal evidence fields required for confidence scoring."""

    source_path: str
    score: float
    created_at: str | None = None


def compute_confidence(
    sources: list[EvidenceSource],
    contradiction_count: int = 0,
    now: datetime | None = None,
) -> ConfidenceResult:
    """Compute an evidence-confidence score from retrieved sources."""
    if not sources:
        return ConfidenceResult(
            score=0.0,
            label="low",
            breakdown={
                "source_count_score": 0.0,
                "source_diversity_score": 0.0,
                "retrieval_score": 0.0,
                "recency_score": 0.0,
                "contradiction_penalty": 0.0,
                "source_count": 0,
                "unique_files": 0,
            },
        )

    current = now or datetime.now(timezone.utc)
    unique_files = {Path(source.source_path).as_posix() for source in sources}
    source_count_score = min(1.0, len(sources) / 6)
    source_diversity_score = min(1.0, len(unique_files) / 4)
    retrieval_score = max(0.0, min(1.0, sum(source.score for source in sources) / len(sources)))
    recency_score = _mean_recency_score(sources, current)
    contradiction_penalty = min(1.0, contradiction_count / 3)

    raw_score = (
        source_count_score * SOURCE_COUNT_WEIGHT
        + source_diversity_score * SOURCE_DIVERSITY_WEIGHT
        + retrieval_score * RETRIEVAL_SCORE_WEIGHT
        + recency_score * RECENCY_WEIGHT
        - contradiction_penalty * CONTRADICTION_WEIGHT
    )
    raw_score = max(0.0, min(1.0, raw_score))
    calibrated = calibrated_probability("answer_confidence", raw_score)
    score = calibrated.calibrated_score
    breakdown = {
        "raw_score": round(raw_score, 4),
        "calibrated_score": round(score, 4),
        "calibration_applied": calibrated.applied,
        "calibration_examples": calibrated.examples,
        "calibration_method": calibrated.method,
        "source_count_score": round(source_count_score, 4),
        "source_diversity_score": round(source_diversity_score, 4),
        "retrieval_score": round(retrieval_score, 4),
        "recency_score": round(recency_score, 4),
        "contradiction_penalty": round(contradiction_penalty, 4),
        "source_count": len(sources),
        "unique_files": len(unique_files),
        "weights": {
            "source_count": SOURCE_COUNT_WEIGHT,
            "source_diversity": SOURCE_DIVERSITY_WEIGHT,
            "retrieval_score": RETRIEVAL_SCORE_WEIGHT,
            "recency": RECENCY_WEIGHT,
            "contradiction_penalty": CONTRADICTION_WEIGHT,
        },
    }
    return ConfidenceResult(score=round(score, 4), label=confidence_label(score), breakdown=breakdown)


def store_confidence_score(output_id: int | None, query: str, result: ConfidenceResult) -> int:
    """Persist a confidence score and return its database id."""
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO confidence_scores
            (output_id, query, score, label, breakdown_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                output_id,
                query,
                result.score,
                result.label,
                json.dumps(result.breakdown, sort_keys=True),
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def confidence_label(score: float) -> str:
    """Map a normalized confidence score to a user-facing label."""
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _mean_recency_score(sources: list[EvidenceSource], now: datetime) -> float:
    scores = [_recency_score(source.created_at, now) for source in sources]
    return sum(scores) / len(scores)


def _recency_score(value: str | None, now: datetime) -> float:
    if not value:
        return 0.5
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0, (now - parsed).days)
    if age_days <= 30:
        return 1.0
    if age_days >= 365:
        return 0.2
    return 1.0 - ((age_days - 30) / 335) * 0.8
