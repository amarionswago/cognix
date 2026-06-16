from fastapi import APIRouter, HTTPException

from app.models.schemas import ProviderUpdate
from app.services.providers import (
    get_provider_settings,
    list_provider_settings,
    save_provider_settings,
    test_provider_connection,
)

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("")
def providers() -> list[dict]:
    return list_provider_settings()


@router.get("/{provider}")
def provider_status(provider: str) -> dict:
    try:
        return get_provider_settings(provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{provider}")
def provider_save(provider: str, update: ProviderUpdate) -> dict:
    try:
        return save_provider_settings(provider, update.enabled, update.api_key, update.model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{provider}/test")
def provider_test(provider: str) -> dict:
    try:
        return test_provider_connection(provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
