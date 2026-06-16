import hashlib
from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.chunking import chunk_text
from app.services.chroma_store import upsert_chunk
from app.services.embeddings import store_chunk_embedding
from app.services.jobs import create_job, finish_job, log_error, start_job
from app.services.parsers import PARSER_VERSION, parse_file
from app.services.security import classify_sensitivity, classify_source_type


SKIP_NAMES = {".DS_Store", ".gitkeep"}


def discover_raw_files() -> list[Path]:
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    files = [path for path in raw_dir.rglob("*") if path.is_file() and path.name not in SKIP_NAMES]
    return sorted(files, key=lambda path: str(path).lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ingest(source: str = "manual") -> dict:
    settings = get_settings()
    settings.ensure_directories()
    cleanup_ignored_files()
    files = discover_raw_files()
    job_id = create_job("ingest", f"Ingest started from {source}", total=len(files))
    start_job(job_id, len(files))

    processed = 0
    skipped = 0
    failed = 0

    for path in files:
        try:
            result = ingest_one_file(path, job_id)
            if result == "processed":
                processed += 1
            else:
                skipped += 1
        except Exception as exc:  # One failed file should not halt the run.
            failed += 1
            log_error(job_id, str(path), type(exc).__name__, str(exc))
            with db_session() as conn:
                conn.execute(
                    "UPDATE raw_files SET status='failed', error_count=error_count + 1 WHERE path=? AND status='processing'",
                    (str(path),),
                )

    finish_job(job_id, processed, failed, f"Ingest complete: {processed} processed, {skipped} skipped, {failed} failed")
    return {"job_id": job_id, "discovered": len(files), "processed": processed, "skipped": skipped, "failed": failed}


def ingest_one_file(path: Path, job_id: int | None = None) -> str:
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    relative_path = str(path.relative_to(raw_dir))
    digest = sha256_file(path)
    stat = path.stat()
    extension = path.suffix.lower()

    with db_session() as conn:
        existing = conn.execute(
            "SELECT id, status, parser_version FROM raw_files WHERE path=? AND sha256=?",
            (str(path), digest),
        ).fetchone()
        if existing and existing["status"] == "processed" and existing["parser_version"] == PARSER_VERSION:
            conn.execute("UPDATE raw_files SET last_seen_at=? WHERE id=?", (utc_now(), existing["id"]))
            return "skipped"

    parsed = parse_file(path)
    sensitivity = classify_sensitivity(path, parsed.text[:4000])
    source_type = classify_source_type(path)

    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO raw_files
            (path, relative_path, sha256, size_bytes, extension, source_type, sensitivity,
             status, parser_version, first_seen_at, last_seen_at, processed_at, error_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?, COALESCE(
                (SELECT first_seen_at FROM raw_files WHERE path=? AND sha256=?), ?
            ), ?, NULL, 0)
            """,
            (
                str(path),
                relative_path,
                digest,
                stat.st_size,
                extension,
                source_type,
                sensitivity,
                PARSER_VERSION,
                str(path),
                digest,
                utc_now(),
                utc_now(),
            ),
        )
        file_id = int(cursor.lastrowid)

    text_path = _write_processed_text(path, parsed.text)
    chunks = chunk_text(parsed.text)

    chunk_records: list[tuple[int, str]] = []
    with db_session() as conn:
        doc_cursor = conn.execute(
            """
            INSERT INTO extracted_documents
            (file_id, title, text_path, text_preview, word_count, extraction_method, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                parsed.title,
                str(text_path),
                parsed.text[:500],
                len(parsed.text.split()),
                parsed.method,
                utc_now(),
            ),
        )
        document_id = int(doc_cursor.lastrowid)
        conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        for chunk in chunks:
            chunk_cursor = conn.execute(
                """
                INSERT INTO chunks
                (file_id, document_id, chunk_index, text, token_estimate, source_path, sensitivity, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, document_id, chunk.index, chunk.text, chunk.token_estimate, relative_path, sensitivity, utc_now()),
            )
            chunk_records.append((int(chunk_cursor.lastrowid), chunk.text))
        conn.execute("UPDATE raw_files SET status='processed', processed_at=? WHERE id=?", (utc_now(), file_id))

    for chunk_id, text in chunk_records:
        store_chunk_embedding(chunk_id, text)
        upsert_chunk(chunk_id, text, {"source_path": relative_path, "sensitivity": sensitivity, "file_id": file_id})

    return "processed"


def _write_processed_text(source_path: Path, text: str) -> Path:
    settings = get_settings()
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
    output = settings.resolved_processed_dir() / f"{source_path.stem}-{digest}.txt"
    output.write_text(text, encoding="utf-8")
    return output


def cleanup_ignored_files() -> None:
    with db_session() as conn:
        conn.execute(
            """
            DELETE FROM raw_files
            WHERE relative_path = '.gitkeep'
               OR relative_path LIKE '%.gitkeep'
               OR path LIKE '%.gitkeep'
            """
        )
