from app.config import get_settings
from app.services.embeddings import EMBEDDING_MODEL, active_embedding_model, embed_text


def test_hash_embedding_fallback_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_LOCAL_EMBEDDING_BACKEND", "hash")
    get_settings.cache_clear()

    vector = embed_text("semantic search retrieval")

    assert len(vector) == get_settings().embedding_dimensions
    assert active_embedding_model() == EMBEDDING_MODEL
    assert abs(sum(value * value for value in vector) - 1.0) < 0.001
    get_settings.cache_clear()
