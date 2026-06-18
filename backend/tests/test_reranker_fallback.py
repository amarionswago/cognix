from app.config import get_settings
import app.services.reranker as reranker
from app.services.retrieval_types import RetrievedChunk


def test_cross_encoder_reranker_falls_back_without_optional_dependency(monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_RERANKER_BACKEND", "cross-encoder")
    monkeypatch.setattr(reranker, "load_cross_encoder", lambda: None)
    get_settings.cache_clear()
    chunks = [
        RetrievedChunk(1, "a.md", "alpha", 0.1, "research"),
        RetrievedChunk(2, "b.md", "beta", 0.2, "research"),
    ]

    reranked = reranker.rerank_with_optional_model("alpha", chunks)

    assert reranked == chunks
    get_settings.cache_clear()
