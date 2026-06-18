"""FastAPI routes for advanced ML capability readiness."""

from fastapi import APIRouter

from app.models.schemas import MlReadinessResponse
from app.services.ml_readiness import advanced_ml_readiness

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/readiness", response_model=MlReadinessResponse)
async def ml_readiness() -> MlReadinessResponse:
    """Return machine-verifiable readiness for advanced ML features."""
    return MlReadinessResponse(**advanced_ml_readiness())
