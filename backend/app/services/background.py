import threading
import time

from app.database import db_session, utc_now
from app.services.compiler import compile_source_summaries
from app.services.health import run_health_check
from app.services.ingest import run_ingest


_watcher_thread: threading.Thread | None = None
_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def ensure_background_rows() -> None:
    now = utc_now()
    with db_session() as conn:
        for name, interval in {"watcher": 20, "scheduler": 300}.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO background_services
                (name, enabled, interval_seconds, last_message, updated_at)
                VALUES (?, 0, ?, '', ?)
                """,
                (name, interval, now),
            )


def list_background_services() -> list[dict]:
    ensure_background_rows()
    with db_session() as conn:
        return conn.execute("SELECT * FROM background_services ORDER BY name").fetchall()


def set_background_service(name: str, enabled: bool, interval_seconds: int | None = None) -> dict:
    ensure_background_rows()
    if name not in {"watcher", "scheduler"}:
        raise ValueError(f"Unknown background service: {name}")
    if interval_seconds is None:
        with db_session() as conn:
            current = conn.execute("SELECT interval_seconds FROM background_services WHERE name=?", (name,)).fetchone()
            interval_seconds = current["interval_seconds"]
    interval_seconds = max(5, int(interval_seconds))
    with db_session() as conn:
        conn.execute(
            """
            UPDATE background_services
            SET enabled=?, interval_seconds=?, updated_at=?
            WHERE name=?
            """,
            (1 if enabled else 0, interval_seconds, utc_now(), name),
        )
        row = conn.execute("SELECT * FROM background_services WHERE name=?", (name,)).fetchone()
    start_background_threads()
    return row


def start_background_threads() -> None:
    global _watcher_thread, _scheduler_thread
    ensure_background_rows()
    if _watcher_thread is None or not _watcher_thread.is_alive():
        _watcher_thread = threading.Thread(target=_polling_loop, args=("watcher",), daemon=True)
        _watcher_thread.start()
    if _scheduler_thread is None or not _scheduler_thread.is_alive():
        _scheduler_thread = threading.Thread(target=_polling_loop, args=("scheduler",), daemon=True)
        _scheduler_thread.start()


def _polling_loop(name: str) -> None:
    while not _stop_event.is_set():
        service = _get_service(name)
        if service and service["enabled"]:
            try:
                if name == "watcher":
                    result = run_ingest("watcher")
                    message = f"Watcher ingest: {result['processed']} processed, {result['skipped']} skipped, {result['failed']} failed"
                else:
                    result = run_ingest("scheduler")
                    compile_source_summaries()
                    run_health_check()
                    message = f"Scheduled run: {result['processed']} processed, {result['skipped']} skipped, {result['failed']} failed"
                _mark_run(name, message)
            except Exception as exc:
                _mark_run(name, f"{type(exc).__name__}: {exc}")
        time.sleep(max(5, int((service or {}).get("interval_seconds", 30))))


def _get_service(name: str) -> dict | None:
    with db_session() as conn:
        return conn.execute("SELECT * FROM background_services WHERE name=?", (name,)).fetchone()


def _mark_run(name: str, message: str) -> None:
    with db_session() as conn:
        conn.execute(
            "UPDATE background_services SET last_run_at=?, last_message=?, updated_at=? WHERE name=?",
            (utc_now(), message, utc_now(), name),
        )
