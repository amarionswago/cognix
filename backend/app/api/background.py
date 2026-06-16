from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.background import list_background_services, set_background_service

router = APIRouter(prefix="/api/background", tags=["background"])


class BackgroundUpdate(BaseModel):
    enabled: bool
    interval_seconds: int | None = None


@router.get("")
def background_services() -> list[dict]:
    return list_background_services()


@router.put("/{name}")
def update_background_service(name: str, update: BackgroundUpdate) -> dict:
    try:
        return set_background_service(name, update.enabled, update.interval_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
