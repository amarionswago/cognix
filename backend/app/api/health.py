from fastapi import APIRouter

from app.models.schemas import HealthSummary
from app.services.health import health_summary, run_health_check

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthSummary)
def summary() -> dict:
    return health_summary()


@router.post("/run", response_model=HealthSummary)
def run() -> dict:
    return run_health_check()

