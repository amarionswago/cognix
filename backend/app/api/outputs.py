from fastapi import APIRouter, HTTPException

from app.models.schemas import OutputUpdate
from app.services.outputs import list_outputs, update_output_status

router = APIRouter(prefix="/api/outputs", tags=["outputs"])


@router.get("")
def outputs() -> list[dict]:
    return list_outputs()


@router.patch("/{output_id}")
def update_output(output_id: int, update: OutputUpdate) -> dict:
    row = update_output_status(output_id, update.status)
    if not row:
        raise HTTPException(status_code=404, detail="Output not found")
    return row

