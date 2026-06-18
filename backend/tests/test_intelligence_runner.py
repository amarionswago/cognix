from pathlib import Path

from app.config import get_settings
from app.database import db_session, init_db, utc_now
from app.services.intelligence.runner import run_intelligence_pass


def test_offline_intelligence_pass_creates_findings_and_briefing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    get_settings.cache_clear()
    init_db()

    now = utc_now()
    text = (
        "Semantic Search is a retrieval technique that finds documents by meaning rather than exact words. "
        "Semantic Search helps knowledge systems connect related evidence. "
        "Semantic Search improves research workflows. "
    ) * 4
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO raw_files
            (path, relative_path, sha256, size_bytes, extension, source_type, sensitivity,
             status, parser_version, first_seen_at, last_seen_at, processed_at, error_count)
            VALUES (?, ?, 'hash', ?, '.md', 'document', 'research', 'processed', 'test', ?, ?, ?, 0)
            """,
            (str(tmp_path / "raw" / "semantic.md"), "documents/semantic.md", len(text), now, now, now),
        )
        file_id = int(cursor.lastrowid)
        doc_cursor = conn.execute(
            """
            INSERT INTO extracted_documents
            (file_id, title, text_path, text_preview, word_count, extraction_method, created_at)
            VALUES (?, 'semantic', 'processed/semantic.txt', ?, ?, 'test', ?)
            """,
            (file_id, text[:500], len(text.split()), now),
        )
        document_id = int(doc_cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO chunks
            (file_id, document_id, chunk_index, text, token_estimate, source_path, sensitivity, created_at)
            VALUES (?, ?, 0, ?, 120, 'documents/semantic.md', 'research', ?)
            """,
            (file_id, document_id, text, now),
        )

    result = run_intelligence_pass("test", use_llm=False)

    assert result["status"] == "completed"
    assert result["findings_created"] >= 1
    assert Path(result["briefing_path"]).exists()

    with db_session() as conn:
        assert conn.execute("SELECT COUNT(*) AS count FROM claims").fetchone()["count"] >= 1
        assert conn.execute("SELECT COUNT(*) AS count FROM intelligence_findings").fetchone()["count"] >= 1
        assert conn.execute("SELECT COUNT(*) AS count FROM briefings").fetchone()["count"] == 1

    get_settings.cache_clear()
