from fastapi import APIRouter

from app.database import db_session

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("")
def files() -> list[dict]:
    with db_session() as conn:
        return conn.execute(
            """
            SELECT id, path, relative_path, sha256, size_bytes, extension, source_type,
                   status, parser_version, first_seen_at, last_seen_at, processed_at, error_count
            FROM raw_files
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()
