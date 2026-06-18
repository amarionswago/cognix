"""Model artifact registry for Cognix fine-tuning outputs."""

from __future__ import annotations

import json
from typing import Any

from app.database import db_session, utc_now


def record_model_artifact(
    name: str,
    base_model: str,
    artifact_type: str,
    path: str,
    status: str,
    metrics: dict[str, Any] | None = None,
    training_manifest: dict[str, Any] | None = None,
) -> int:
    """Store a model artifact or planned artifact record."""
    now = utc_now()
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_artifacts
            (name, base_model, artifact_type, path, status, metrics_json,
             training_manifest_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                base_model,
                artifact_type,
                path,
                status,
                json.dumps(metrics or {}, sort_keys=True),
                json.dumps(training_manifest or {}, sort_keys=True),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def list_model_artifacts(limit: int = 50) -> list[dict]:
    """Return recent model artifact records."""
    with db_session() as conn:
        return conn.execute(
            "SELECT * FROM model_artifacts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def update_model_artifact_status(artifact_id: int, status: str, metrics: dict[str, Any] | None = None) -> None:
    """Update model artifact status and optional metrics."""
    with db_session() as conn:
        if metrics is None:
            conn.execute(
                "UPDATE model_artifacts SET status=?, updated_at=? WHERE id=?",
                (status, utc_now(), artifact_id),
            )
        else:
            conn.execute(
                "UPDATE model_artifacts SET status=?, metrics_json=?, updated_at=? WHERE id=?",
                (status, json.dumps(metrics, sort_keys=True), utc_now(), artifact_id),
            )
