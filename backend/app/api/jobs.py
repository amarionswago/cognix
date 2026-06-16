from fastapi import APIRouter

from app.services.jobs import list_errors, list_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def jobs() -> dict:
    return {"jobs": list_jobs(), "errors": list_errors()}

