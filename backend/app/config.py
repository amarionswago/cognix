from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime paths and defaults for the local Cognix instance."""

    model_config = SettingsConfigDict(env_prefix="COGNIX_", env_file=".env")

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path | None = None
    wiki_dir: Path | None = None
    raw_dir: Path | None = None
    processed_dir: Path | None = None
    chroma_dir: Path | None = None
    logs_dir: Path | None = None
    database_path: Path | None = None
    app_name: str = "Cognix"
    app_version: str = "0.1.0"
    embedding_dimensions: int = 128

    def resolved_data_dir(self) -> Path:
        return self.data_dir or self.project_root / "data"

    def resolved_wiki_dir(self) -> Path:
        return self.wiki_dir or self.project_root / "wiki"

    def resolved_raw_dir(self) -> Path:
        return self.raw_dir or self.resolved_data_dir() / "raw"

    def resolved_processed_dir(self) -> Path:
        return self.processed_dir or self.resolved_data_dir() / "processed"

    def resolved_chroma_dir(self) -> Path:
        return self.chroma_dir or self.resolved_data_dir() / "chroma"

    def resolved_logs_dir(self) -> Path:
        return self.logs_dir or self.resolved_data_dir() / "logs"

    def resolved_database_path(self) -> Path:
        return self.database_path or self.resolved_data_dir() / "library.sqlite"

    def ensure_directories(self) -> None:
        paths = [
            self.resolved_data_dir(),
            self.resolved_raw_dir(),
            self.resolved_processed_dir(),
            self.resolved_chroma_dir(),
            self.resolved_logs_dir(),
            self.resolved_wiki_dir(),
            self.resolved_wiki_dir() / "sources",
            self.resolved_wiki_dir() / "outputs" / "analysis",
            self.resolved_wiki_dir() / "_health",
            self.resolved_wiki_dir() / "_indexes",
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

