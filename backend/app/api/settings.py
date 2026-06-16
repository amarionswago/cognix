from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import ProfileUpdate
from app.services.preferences import get_profile, update_profile

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def settings() -> dict:
    settings_obj = get_settings()
    return {
        "app_name": settings_obj.app_name,
        "app_version": settings_obj.app_version,
        "raw_dir": str(settings_obj.resolved_raw_dir()),
        "wiki_dir": str(settings_obj.resolved_wiki_dir()),
        "database_path": str(settings_obj.resolved_database_path()),
        "embedding_dimensions": settings_obj.embedding_dimensions,
        "llm_strategy": "hybrid-ready; deterministic local synthesis active until providers are configured",
    }


@router.get("/profile")
def profile() -> dict:
    return get_profile()


@router.put("/profile")
def save_profile(update: ProfileUpdate) -> dict:
    return update_profile(update.model_dump())
