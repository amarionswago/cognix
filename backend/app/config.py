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
    chroma_enabled: bool = True
    logs_dir: Path | None = None
    database_path: Path | None = None
    app_name: str = "Cognix"
    app_version: str = "0.1.0"
    embedding_dimensions: int = 128
    local_embedding_backend: str = "hash"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_backend: str = "deterministic"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_cross_encoder_reranker_model_path: Path | None = None
    pair_reranker_model_path: Path | None = None
    nli_backend: str = "heuristic"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    local_cross_encoder_nli_model_path: Path | None = None
    transformer_lora_nli_model_path: Path | None = None
    pair_nli_model_path: Path | None = None
    synthesis_backend: str = "provider"
    cognix_micro_model_path: Path | None = None
    cognix_sft_adapter_path: Path | None = None
    vision_backend: str = "local-ocr"
    openai_vision_model: str = "gpt-4.1-mini"
    cloud_embeddings_enabled: bool = False
    openai_embedding_model: str = "text-embedding-3-small"
    intelligence_run_hour: int = 2

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

    def resolved_pair_reranker_model_path(self) -> Path:
        return self.pair_reranker_model_path or self.resolved_data_dir() / "models" / "cognix-reranker-pair.json"

    def resolved_local_cross_encoder_reranker_model_path(self) -> Path:
        return self.local_cross_encoder_reranker_model_path or self.resolved_data_dir() / "models" / "cognix-reranker-cross-encoder.json"

    def resolved_pair_nli_model_path(self) -> Path:
        return self.pair_nli_model_path or self.resolved_data_dir() / "models" / "cognix-nli-pair.json"

    def resolved_local_cross_encoder_nli_model_path(self) -> Path:
        return self.local_cross_encoder_nli_model_path or self.resolved_data_dir() / "models" / "cognix-nli-cross-encoder.json"

    def resolved_transformer_lora_nli_model_path(self) -> Path:
        return self.transformer_lora_nli_model_path or self.resolved_data_dir() / "models" / "cognix-nli-transformer-lora.json"

    def resolved_cognix_micro_model_path(self) -> Path:
        return self.cognix_micro_model_path or self.resolved_data_dir() / "models" / "cognix-micro-synthesis.json"

    def resolved_cognix_sft_adapter_path(self) -> Path:
        return self.cognix_sft_adapter_path or self.resolved_data_dir() / "models" / "cognix-sft-adapter.json"

    def ensure_directories(self) -> None:
        paths = [
            self.resolved_data_dir(),
            self.resolved_raw_dir(),
            self.resolved_processed_dir(),
            self.resolved_chroma_dir(),
            self.resolved_logs_dir(),
            self.resolved_data_dir() / "models",
            self.resolved_wiki_dir(),
            self.resolved_wiki_dir() / "sources",
            self.resolved_wiki_dir() / "outputs" / "analysis",
            self.resolved_wiki_dir() / "_health",
            self.resolved_wiki_dir() / "_indexes",
            self.resolved_wiki_dir() / "_intelligence",
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
