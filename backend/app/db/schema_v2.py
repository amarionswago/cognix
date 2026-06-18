"""Idempotent v2 schema migration for Cognix intelligence features.

The v1 schema stores files, chunks, embeddings, outputs, jobs, and basic health
findings. Cognix v2 adds the durable substrate for proactive knowledge auditing:
claims, concepts, confidence scores, graph edges, intelligence findings, run
logs, and generated intelligence briefings.
"""

from __future__ import annotations

import sqlite3

MIGRATION_ID = "20260617_0001_cognix_v2_foundation"
EMBEDDING_METADATA_MIGRATION_ID = "20260617_0002_embedding_metadata"
CALIBRATION_MIGRATION_ID = "20260617_0003_evaluation_and_calibration"
MODEL_ARTIFACTS_MIGRATION_ID = "20260617_0004_model_artifacts"
CALIBRATION_MODELS_MIGRATION_ID = "20260617_0005_calibration_models"
EXTRACTION_ARTIFACTS_MIGRATION_ID = "20260617_0006_extraction_artifacts"


def apply_v2_migration(conn: sqlite3.Connection) -> None:
    """Create v2 foundation tables and indexes without destroying v1 data."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    apply_sql_migration(conn, MIGRATION_ID, V2_SCHEMA_SQL)
    apply_embedding_metadata_migration(conn)
    apply_sql_migration(conn, CALIBRATION_MIGRATION_ID, CALIBRATION_SCHEMA_SQL)
    apply_sql_migration(conn, MODEL_ARTIFACTS_MIGRATION_ID, MODEL_ARTIFACTS_SCHEMA_SQL)
    apply_sql_migration(conn, CALIBRATION_MODELS_MIGRATION_ID, CALIBRATION_MODELS_SCHEMA_SQL)
    apply_sql_migration(conn, EXTRACTION_ARTIFACTS_MIGRATION_ID, EXTRACTION_ARTIFACTS_SCHEMA_SQL)


def apply_sql_migration(conn: sqlite3.Connection, migration_id: str, sql: str) -> None:
    """Apply a SQL migration once."""
    existing = conn.execute("SELECT id FROM schema_migrations WHERE id=?", (migration_id,)).fetchone()
    if existing:
        return
    conn.executescript(sql)
    record_migration(conn, migration_id)


def apply_embedding_metadata_migration(conn: sqlite3.Connection) -> None:
    """Add provider metadata columns to existing chunk embeddings."""
    existing = conn.execute(
        "SELECT id FROM schema_migrations WHERE id=?",
        (EMBEDDING_METADATA_MIGRATION_ID,),
    ).fetchone()
    if existing:
        return
    existing_columns = {column_name(row) for row in conn.execute("PRAGMA table_info(chunk_embeddings)").fetchall()}
    if "provider" not in existing_columns:
        conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'")
    if "embedding_source" not in existing_columns:
        conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN embedding_source TEXT NOT NULL DEFAULT 'chunk'")
    record_migration(conn, EMBEDDING_METADATA_MIGRATION_ID)


def record_migration(conn: sqlite3.Connection, migration_id: str) -> None:
    """Record an applied migration."""
    conn.execute(
        """
        INSERT INTO schema_migrations (id, applied_at)
        VALUES (?, datetime('now'))
        """,
        (migration_id,),
    )


def column_name(row: object) -> str:
    """Return a PRAGMA table_info column name for tuple or dict row factories."""
    if isinstance(row, dict):
        return str(row["name"])
    return str(row[1])


V2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    embedding_json TEXT NOT NULL DEFAULT '',
    embedding_provider TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dimensions INTEGER NOT NULL DEFAULT 0,
    source_date TEXT,
    ingest_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(file_id) REFERENCES raw_files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_claims_chunk_id ON claims(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claims_file_id ON claims(file_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);

CREATE TABLE IF NOT EXISTS concept_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    normalized_concept TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(file_id) REFERENCES raw_files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_concept_mentions_normalized ON concept_mentions(normalized_concept);
CREATE INDEX IF NOT EXISTS idx_concept_mentions_chunk_id ON concept_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_concept_mentions_file_id ON concept_mentions(file_id);

CREATE TABLE IF NOT EXISTS intelligence_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    suggested_action TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    confidence REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_intelligence_findings_type ON intelligence_findings(finding_type);
CREATE INDEX IF NOT EXISTS idx_intelligence_findings_status ON intelligence_findings(status);
CREATE INDEX IF NOT EXISTS idx_intelligence_findings_severity ON intelligence_findings(severity);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_concept TEXT NOT NULL,
    target_concept TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    source_file TEXT NOT NULL DEFAULT '',
    source_chunk_id INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_chunk_id) REFERENCES chunks(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_concept);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_concept);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relationship ON graph_edges(relationship);

CREATE TABLE IF NOT EXISTS confidence_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    output_id INTEGER,
    query TEXT NOT NULL,
    score REAL NOT NULL,
    label TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(output_id) REFERENCES outputs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_confidence_scores_output_id ON confidence_scores(output_id);
CREATE INDEX IF NOT EXISTS idx_confidence_scores_label ON confidence_scores(label);

CREATE TABLE IF NOT EXISTS briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_date TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    summary TEXT NOT NULL,
    finding_counts_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_briefings_date ON briefings(brief_date);
CREATE INDEX IF NOT EXISTS idx_briefings_status ON briefings(status);

CREATE TABLE IF NOT EXISTS intelligence_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    chunks_scanned INTEGER NOT NULL DEFAULT 0,
    claims_extracted INTEGER NOT NULL DEFAULT 0,
    concepts_extracted INTEGER NOT NULL DEFAULT 0,
    findings_created INTEGER NOT NULL DEFAULT 0,
    briefings_created INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_intelligence_runs_status ON intelligence_runs(status);
CREATE INDEX IF NOT EXISTS idx_intelligence_runs_type ON intelligence_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_intelligence_runs_started ON intelligence_runs(started_at);
"""


CALIBRATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    predicted_score REAL NOT NULL,
    model_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_task ON model_predictions(task);
CREATE INDEX IF NOT EXISTS idx_model_predictions_input_hash ON model_predictions(input_hash);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    actual_label TEXT NOT NULL,
    reviewer TEXT NOT NULL DEFAULT 'system',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES model_predictions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_prediction ON prediction_outcomes(prediction_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_actual ON prediction_outcomes(actual_label);

CREATE TABLE IF NOT EXISTS training_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    source TEXT NOT NULL,
    quality_label TEXT NOT NULL DEFAULT 'unreviewed',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_examples_task ON training_examples(task);
CREATE INDEX IF NOT EXISTS idx_training_examples_quality ON training_examples(quality_label);
"""


MODEL_ARTIFACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    training_manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_artifacts_name ON model_artifacts(name);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_status ON model_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_type ON model_artifacts(artifact_type);
"""


CALIBRATION_MODELS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calibration_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    method TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    examples INTEGER NOT NULL,
    positive_examples INTEGER NOT NULL,
    negative_examples INTEGER NOT NULL,
    brier_score REAL NOT NULL,
    log_loss REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calibration_models_task ON calibration_models(task);
CREATE INDEX IF NOT EXISTS idx_calibration_models_status ON calibration_models(status);
"""


EXTRACTION_ARTIFACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS extraction_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    document_id INTEGER,
    artifact_type TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence REAL NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    text_length INTEGER NOT NULL DEFAULT 0,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES raw_files(id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES extracted_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_extraction_artifacts_file ON extraction_artifacts(file_id);
CREATE INDEX IF NOT EXISTS idx_extraction_artifacts_method ON extraction_artifacts(method);
CREATE INDEX IF NOT EXISTS idx_extraction_artifacts_confidence ON extraction_artifacts(confidence);
"""
