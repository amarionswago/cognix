import hashlib
import json
import threading
from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.chunking import chunk_text
from app.services.chroma_store import upsert_chunks
from app.services.embeddings import active_embedding_model, active_embedding_provider, embed_texts
from app.services.jobs import create_job, finish_job, log_error, start_job
from app.services.parsers import PARSER_VERSION, parse_file
from app.services.security import classify_sensitivity, classify_source_type


SKIP_NAMES = {".DS_Store", ".gitkeep"}
SKIP_DIR_NAMES = {
    ".cache",
    ".git",
    ".gradle",
    ".idea",
    ".mypy_cache",
    ".next",
    ".nox",
    ".nuxt",
    ".parcel-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".turbo",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "env",
    "htmlcov",
    "node_modules",
    "site-packages",
    "target",
    "venv",
}
SKIP_DIR_SUFFIXES = (".dist-info", ".egg-info", "_files")
SKIP_EXTENSIONS = {
    ".a",
    ".bin",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".jar",
    ".lib",
    ".map",
    ".o",
    ".obj",
    ".pyo",
    ".pyc",
    ".so",
    ".wasm",
    ".whl",
}
INGEST_LOCK = threading.Lock()


def should_skip_path(path: Path, raw_dir: Path | None = None) -> bool:
    if path.name in SKIP_NAMES:
        return True
    lowered_name = path.name.lower()
    if lowered_name.endswith((".min.css", ".min.js")):
        return True
    try:
        parts = path.relative_to(raw_dir).parts if raw_dir else path.parts
    except ValueError:
        parts = path.parts
    for part in parts:
        lowered = part.lower()
        if lowered in SKIP_DIR_NAMES or lowered.endswith(SKIP_DIR_SUFFIXES):
            return True
    return path.suffix.lower() in SKIP_EXTENSIONS


def discover_raw_files() -> list[Path]:
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    files = [path for path in raw_dir.rglob("*") if path.is_file() and not should_skip_path(path, raw_dir)]
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
    if not INGEST_LOCK.acquire(blocking=False):
        job_id = create_job("ingest", f"Ingest skipped from {source}: another ingest is already running", total=0)
        start_job(job_id, 0)
        finish_job(job_id, 0, 0, "Ingest skipped: another ingest is already running")
        return {"job_id": job_id, "discovered": 0, "processed": 0, "skipped": 0, "failed": 0, "ignored_removed": 0}
    try:
        return _run_ingest_locked(source)
    finally:
        INGEST_LOCK.release()


def _run_ingest_locked(source: str = "manual") -> dict:
    ignored_removed = cleanup_unindexable_files()
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

    ignored_note = f", {ignored_removed} ignored index records removed" if ignored_removed else ""
    finish_job(
        job_id,
        processed,
        failed,
        f"Ingest complete: {processed} processed, {skipped} unchanged, {failed} failed{ignored_note}",
    )
    return {
        "job_id": job_id,
        "discovered": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "ignored_removed": ignored_removed,
    }


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
        existing = conn.execute("SELECT id FROM raw_files WHERE path=? AND sha256=?", (str(path), digest)).fetchone()
        if existing:
            file_id = int(existing["id"])
            conn.execute(
                """
                UPDATE raw_files
                SET relative_path=?, size_bytes=?, extension=?, source_type=?, sensitivity=?,
                    status='processing', parser_version=?, last_seen_at=?, processed_at=NULL, error_count=0
                WHERE id=?
                """,
                (relative_path, stat.st_size, extension, source_type, sensitivity, PARSER_VERSION, utc_now(), file_id),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO raw_files
                (path, relative_path, sha256, size_bytes, extension, source_type, sensitivity,
                 status, parser_version, first_seen_at, last_seen_at, processed_at, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?, ?, NULL, 0)
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
                    utc_now(),
                    utc_now(),
                ),
            )
            file_id = int(cursor.lastrowid)

    text_path = _write_processed_text(path, parsed.text)
    chunks = chunk_text(parsed.text)
    chunk_texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(chunk_texts)
    if len(vectors) != len(chunk_texts):
        raise RuntimeError("Embedding backend returned a different number of vectors than chunks.")

    chunk_records: list[tuple[int, str]] = []
    with db_session() as conn:
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id=?)",
            (file_id,),
        )
        conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        conn.execute("DELETE FROM extracted_documents WHERE file_id=?", (file_id,))
        conn.execute("DELETE FROM extraction_artifacts WHERE file_id=?", (file_id,))
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
        conn.execute(
            """
            INSERT INTO extraction_artifacts
            (file_id, document_id, artifact_type, method, confidence, page_count,
             text_length, warnings_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                document_id,
                "document_extraction",
                parsed.method,
                parsed.extraction_confidence,
                parsed.page_count,
                len(parsed.text),
                json.dumps(list(parsed.warnings), sort_keys=True),
                parsed.metadata_json,
                utc_now(),
            ),
        )
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
        embedding_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunk_embeddings)").fetchall()}
        has_provider_metadata = {"provider", "embedding_source"} <= embedding_columns
        now = utc_now()
        if has_provider_metadata:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                (chunk_id, vector_json, model, dimensions, created_at, provider, embedding_source)
                VALUES (?, ?, ?, ?, ?, ?, 'chunk')
                """,
                [
                    (
                        chunk_id,
                        json.dumps(vector),
                        active_embedding_model(),
                        len(vector),
                        now,
                        active_embedding_provider(),
                    )
                    for (chunk_id, _text), vector in zip(chunk_records, vectors)
                ],
            )
        else:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                (chunk_id, vector_json, model, dimensions, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (chunk_id, json.dumps(vector), active_embedding_model(), len(vector), now)
                    for (chunk_id, _text), vector in zip(chunk_records, vectors)
                ],
            )
        conn.execute("UPDATE raw_files SET status='processed', processed_at=? WHERE id=?", (utc_now(), file_id))
        conn.execute("DELETE FROM ingest_errors WHERE path=?", (str(path),))

    vector_by_chunk_id = {chunk_id: vector for (chunk_id, _text), vector in zip(chunk_records, vectors)}
    for batch in _batches(chunk_records, 128):
        upsert_chunks(
            [
                (
                    chunk_id,
                    text,
                    {"source_path": relative_path, "sensitivity": sensitivity, "file_id": file_id},
                    vector_by_chunk_id[chunk_id],
                )
                for chunk_id, text in batch
            ]
        )

    return "processed"


def _write_processed_text(source_path: Path, text: str) -> Path:
    settings = get_settings()
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
    output = settings.resolved_processed_dir() / f"{source_path.stem}-{digest}.txt"
    output.write_text(text, encoding="utf-8")
    return output


def cleanup_unindexable_files() -> int:
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    removed = 0
    with db_session() as conn:
        rows = conn.execute("SELECT id, relative_path, path FROM raw_files").fetchall()
        ignored_rows = [
            (int(row["id"]), str(row["path"]))
            for row in rows
            if not Path(str(row["path"])).exists() or should_skip_path(raw_dir / row["relative_path"], raw_dir)
        ]
        for batch in _batches(ignored_rows, 500):
            removed += len(batch)
            delete_indexed_files(conn, batch)
    return removed


def cleanup_ignored_files() -> int:
    return cleanup_unindexable_files()


def delete_indexed_files(conn, indexed_files: list[tuple[int, str]]) -> None:
    file_ids = [file_id for file_id, _path in indexed_files]
    paths = [path for _file_id, path in indexed_files]
    placeholders = ",".join("?" for _ in file_ids)
    path_placeholders = ",".join("?" for _ in paths)
    conn.execute(
        f"DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id IN ({placeholders}))",
        file_ids,
    )
    conn.execute(f"DELETE FROM chunks WHERE file_id IN ({placeholders})", file_ids)
    conn.execute(f"DELETE FROM extraction_artifacts WHERE file_id IN ({placeholders})", file_ids)
    conn.execute(f"DELETE FROM extracted_documents WHERE file_id IN ({placeholders})", file_ids)
    conn.execute(f"DELETE FROM raw_files WHERE id IN ({placeholders})", file_ids)
    conn.execute(f"DELETE FROM ingest_errors WHERE path IN ({path_placeholders})", paths)


def _batches(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]
