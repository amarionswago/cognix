from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, background, files, health, ingest, jobs, outputs, providers, settings
from app.config import get_settings
from app.database import init_db
from app.services.background import start_background_threads


def create_app() -> FastAPI:
    settings_obj = get_settings()
    init_db()
    app = FastAPI(title=settings_obj.app_name, version=settings_obj.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ingest.router)
    app.include_router(ask.router)
    app.include_router(outputs.router)
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(settings.router)
    app.include_router(files.router)
    app.include_router(providers.router)
    app.include_router(background.router)

    @app.on_event("startup")
    def startup_background() -> None:
        start_background_threads()

    return app


app = create_app()


@app.get("/api/status")
def status() -> dict:
    return {"status": "ok", "name": "Cognix"}
