"""FastAPI routes for Cognix Intelligence Briefs."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.database import db_session
from app.models.schemas import BriefingResponse

router = APIRouter(prefix="/api/briefings", tags=["briefings"])


@router.get("/latest", response_model=BriefingResponse)
async def latest_briefing() -> BriefingResponse:
    """Return the latest generated Intelligence Brief."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM briefings ORDER BY brief_date DESC LIMIT 1").fetchone()
    if not row:
        raise HTTPException(status_code=404, detail={"error": "No briefing found", "code": "BRIEFING_NOT_FOUND", "detail": "Run intelligence first."})
    return briefing_response(dict(row))


@router.get("/{brief_date}", response_model=BriefingResponse)
async def briefing_by_date(brief_date: str) -> BriefingResponse:
    """Return an Intelligence Brief by ISO date."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM briefings WHERE brief_date=?", (brief_date,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail={"error": "Briefing not found", "code": "BRIEFING_NOT_FOUND", "detail": brief_date})
    return briefing_response(dict(row))


def briefing_response(row: dict) -> BriefingResponse:
    """Attach markdown content to a briefing row."""
    path = Path(row["path"])
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return BriefingResponse(**row, content=content)

