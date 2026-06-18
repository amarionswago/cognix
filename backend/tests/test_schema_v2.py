import sqlite3

from app.db.schema_v2 import (
    CALIBRATION_MIGRATION_ID,
    CALIBRATION_MODELS_MIGRATION_ID,
    EMBEDDING_METADATA_MIGRATION_ID,
    EXTRACTION_ARTIFACTS_MIGRATION_ID,
    MIGRATION_ID,
    MODEL_ARTIFACTS_MIGRATION_ID,
    apply_v2_migration,
)


def test_v2_migration_creates_foundation_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE raw_files (id INTEGER PRIMARY KEY);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY);
        CREATE TABLE outputs (id INTEGER PRIMARY KEY);
        CREATE TABLE chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector_json TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    apply_v2_migration(conn)
    apply_v2_migration(conn)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "claims" in tables
    assert "concept_mentions" in tables
    assert "intelligence_findings" in tables
    assert "graph_edges" in tables
    assert "confidence_scores" in tables
    assert "briefings" in tables
    assert "intelligence_runs" in tables
    assert "model_predictions" in tables
    assert "prediction_outcomes" in tables
    assert "training_examples" in tables
    assert "model_artifacts" in tables
    assert "calibration_models" in tables
    assert "extraction_artifacts" in tables

    rows = conn.execute("SELECT id FROM schema_migrations ORDER BY id").fetchall()
    assert rows == [
        (MIGRATION_ID,),
        (EMBEDDING_METADATA_MIGRATION_ID,),
        (CALIBRATION_MIGRATION_ID,),
        (MODEL_ARTIFACTS_MIGRATION_ID,),
        (CALIBRATION_MODELS_MIGRATION_ID,),
        (EXTRACTION_ARTIFACTS_MIGRATION_ID,),
    ]

    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(chunk_embeddings)").fetchall()
    }
    assert "provider" in columns
    assert "embedding_source" in columns
