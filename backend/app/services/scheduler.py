"""Nightly intelligence scheduler for Cognix v2."""

from __future__ import annotations

import threading
import time
from datetime import timezone

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.background import ensure_background_rows
from app.services.intelligence.runner import run_intelligence_pass

_fallback_thread: threading.Thread | None = None
_fallback_stop = threading.Event()
_apscheduler_started = False
_apscheduler = None


def start_intelligence_scheduler() -> None:
    """Start the configurable nightly intelligence scheduler if enabled."""
    ensure_background_rows()
    if start_apscheduler_if_available():
        return
    start_fallback_scheduler()


def start_apscheduler_if_available() -> bool:
    """Start APScheduler when installed; return False when unavailable."""
    global _apscheduler_started, _apscheduler
    if _apscheduler_started:
        return True
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        mark_intelligence_scheduler("APScheduler is not installed; using fallback polling scheduler.")
        return False

    try:
        _apscheduler = BackgroundScheduler(timezone=get_scheduler_timezone())
        _apscheduler.add_job(
            scheduled_intelligence_run,
            trigger="cron",
            hour=max(0, min(23, get_settings().intelligence_run_hour)),
            id="cognix-nightly-intelligence",
            replace_existing=True,
            max_instances=1,
        )
        _apscheduler.start()
    except Exception as exc:
        _apscheduler = None
        mark_intelligence_scheduler(
            f"APScheduler startup failed ({type(exc).__name__}: {exc}); using fallback polling scheduler."
        )
        return False

    _apscheduler_started = True
    mark_intelligence_scheduler("Nightly intelligence scheduler started with APScheduler.")
    return True


def get_scheduler_timezone():
    """Return a concrete local timezone object for APScheduler."""
    try:
        from tzlocal import get_localzone

        return get_localzone()
    except Exception:
        return timezone.utc


def start_fallback_scheduler() -> None:
    """Start a lightweight fallback loop for environments without APScheduler."""
    global _fallback_thread
    if _fallback_thread is not None and _fallback_thread.is_alive():
        return
    _fallback_thread = threading.Thread(target=fallback_loop, daemon=True)
    _fallback_thread.start()


def fallback_loop() -> None:
    """Poll the background service row and run intelligence when enabled."""
    while not _fallback_stop.is_set():
        service = get_intelligence_service()
        interval = max(300, int((service or {}).get("interval_seconds", 86400)))
        if service and service["enabled"]:
            scheduled_intelligence_run()
        time.sleep(interval)


def scheduled_intelligence_run() -> None:
    """Run the intelligence pass from a scheduler context."""
    service = get_intelligence_service()
    if not service or not service["enabled"]:
        return
    try:
        result = run_intelligence_pass("nightly", use_llm=False)
        mark_intelligence_scheduler(
            f"Nightly intelligence: {result['findings_created']} findings, brief {result['briefing_id']}."
        )
    except Exception as exc:
        mark_intelligence_scheduler(f"{type(exc).__name__}: {exc}")


def get_intelligence_service() -> dict | None:
    """Return the background service row for intelligence scheduling."""
    ensure_background_rows()
    with db_session() as conn:
        return conn.execute("SELECT * FROM background_services WHERE name='intelligence'").fetchone()


def mark_intelligence_scheduler(message: str) -> None:
    """Write scheduler status into the existing background service table."""
    ensure_background_rows()
    with db_session() as conn:
        conn.execute(
            """
            UPDATE background_services
            SET last_message=?, updated_at=?
            WHERE name='intelligence'
            """,
            (message, utc_now()),
        )
