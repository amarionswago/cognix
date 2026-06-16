from app.database import db_session, utc_now


def create_job(kind: str, message: str, total: int = 0) -> int:
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO jobs (kind, status, message, total, completed, failed, created_at)
            VALUES (?, 'queued', ?, ?, 0, 0, ?)
            """,
            (kind, message, total, utc_now()),
        )
        return int(cursor.lastrowid)


def start_job(job_id: int, total: int | None = None) -> None:
    with db_session() as conn:
        if total is None:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (utc_now(), job_id))
        else:
            conn.execute("UPDATE jobs SET status='running', started_at=?, total=? WHERE id=?", (utc_now(), total, job_id))


def finish_job(job_id: int, completed: int, failed: int, message: str) -> None:
    status = "failed" if completed == 0 and failed > 0 else "completed"
    with db_session() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status=?, completed=?, failed=?, message=?, finished_at=?
            WHERE id=?
            """,
            (status, completed, failed, message, utc_now(), job_id),
        )


def log_error(job_id: int | None, path: str, error_type: str, message: str) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO ingest_errors (job_id, path, error_type, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, path, error_type, message, utc_now()),
        )


def list_jobs(limit: int = 50) -> list[dict]:
    with db_session() as conn:
        return conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def list_errors(limit: int = 100) -> list[dict]:
    with db_session() as conn:
        return conn.execute("SELECT * FROM ingest_errors ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

