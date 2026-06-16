from fastapi import APIRouter

from app.models.schemas import IngestRequest, IngestResponse
from app.services.compiler import compile_source_summaries
from app.services.ingest import run_ingest

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest(request: IngestRequest) -> dict:
    result = run_ingest(request.source)
    compile_source_summaries()
    return result

