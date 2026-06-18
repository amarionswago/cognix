"""FastAPI routes for Cognix v2 proactive intelligence."""

import re

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.database import db_session, utc_now
from app.models.schemas import IntelligenceFindingResponse, IntelligenceRunRequest, IntelligenceRunResponse
from app.services.llm import synthesize_answer
from app.services.intelligence.runner import run_intelligence_pass
from app.services.retrieval import retrieve

router = APIRouter(tags=["intelligence"])


@router.post("/api/intelligence/run", response_model=IntelligenceRunResponse)
async def run_intelligence(request: IntelligenceRunRequest) -> IntelligenceRunResponse:
    """Run the proactive intelligence pipeline and generate a brief."""
    try:
        result = run_intelligence_pass("manual", use_llm=request.use_llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Intelligence run failed", "code": "INTELLIGENCE_RUN_FAILED", "detail": str(exc)}) from exc
    return IntelligenceRunResponse(**result)


@router.get("/api/intelligence/findings", response_model=list[IntelligenceFindingResponse])
async def list_intelligence_findings() -> list[IntelligenceFindingResponse]:
    """List open proactive intelligence findings."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM intelligence_findings
            WHERE status='open'
            ORDER BY
                CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                id DESC
            LIMIT 100
            """
        ).fetchall()
    return [IntelligenceFindingResponse(**dict(row)) for row in rows]


@router.get("/api/contradictions", response_model=list[IntelligenceFindingResponse])
async def list_contradictions() -> list[IntelligenceFindingResponse]:
    """List confirmed and candidate contradictions."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM intelligence_findings
            WHERE status='open'
              AND finding_type IN ('contradiction', 'contradiction_candidate')
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()
    return [IntelligenceFindingResponse(**dict(row)) for row in rows]


@router.post("/api/contradictions/{finding_id}/resolve", response_model=IntelligenceFindingResponse)
async def resolve_contradiction(finding_id: int) -> IntelligenceFindingResponse:
    """Mark a contradiction finding resolved."""
    return update_finding_status(finding_id, "resolved")


@router.get("/api/gaps", response_model=list[IntelligenceFindingResponse])
async def list_gaps() -> list[IntelligenceFindingResponse]:
    """List open knowledge-gap findings."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM intelligence_findings
            WHERE status='open' AND finding_type='gap'
            ORDER BY confidence DESC, id DESC
            LIMIT 100
            """
        ).fetchall()
    return [IntelligenceFindingResponse(**dict(row)) for row in rows]


@router.post("/api/gaps/{concept}/compile")
async def compile_gap(concept: str) -> dict:
    """Compile a knowledge gap into a wiki concept page."""
    normalized = concept.replace("-", " ").strip()
    chunks = retrieve(f"define and explain {normalized}", limit=8)
    if not chunks:
        raise HTTPException(status_code=404, detail={"error": "No evidence found", "code": "GAP_EVIDENCE_NOT_FOUND", "detail": normalized})
    answer = synthesize_answer(f"What does my library say about {normalized}?", chunks, "memo")
    settings = get_settings()
    concepts_dir = settings.resolved_wiki_dir() / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    path = concepts_dir / f"{slugify(normalized)}.md"
    lines = [
        "---",
        f"title: {normalized}",
        "type: concept",
        "status: compiled-from-gap",
        f"created: {utc_now()}",
        "---",
        "",
        f"# {normalized}",
        "",
        answer.strip(),
        "",
        "## Sources",
        "",
    ]
    for chunk in chunks:
        lines.append(f"- `{chunk.source_path}`, chunk {chunk.chunk_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with db_session() as conn:
        conn.execute(
            """
            UPDATE intelligence_findings
            SET status='resolved', resolved_at=?, updated_at=?
            WHERE finding_type='gap'
              AND lower(title)=lower(?)
            """,
            (utc_now(), utc_now(), f"Knowledge gap: {normalized}"),
        )
    return {"status": "compiled", "concept": normalized, "path": str(path)}


def update_finding_status(finding_id: int, status: str) -> IntelligenceFindingResponse:
    """Update a finding status and return the row."""
    with db_session() as conn:
        conn.execute(
            """
            UPDATE intelligence_findings
            SET status=?, resolved_at=?, updated_at=?
            WHERE id=?
            """,
            (status, utc_now(), utc_now(), finding_id),
        )
        row = conn.execute("SELECT * FROM intelligence_findings WHERE id=?", (finding_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail={"error": "Finding not found", "code": "FINDING_NOT_FOUND", "detail": str(finding_id)})
    return IntelligenceFindingResponse(**dict(row))


def slugify(value: str) -> str:
    """Create a filesystem-safe concept slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "concept"
