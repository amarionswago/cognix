"""Optional trained reranking backends for Cognix retrieval."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.cross_encoder_model import TinyCrossEncoder, load_cross_encoder_model
from app.services.pair_model import PairTextModel, load_pair_model
from app.services.retrieval_types import RetrievedChunk

LOCAL_CROSS_ENCODER_BACKENDS = {"cognix-cross-encoder", "local-cross-encoder", "tiny-cross-encoder"}


def rerank_with_optional_model(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank chunks with a trained cross-encoder when configured and available."""
    settings = get_settings()
    backend = settings.reranker_backend.strip().lower()
    if not chunks or backend not in {"cross-encoder", "cross_encoder", "neural", "cognix-pair", "pair-mlp", "local-pair", *LOCAL_CROSS_ENCODER_BACKENDS}:
        return chunks
    if backend in LOCAL_CROSS_ENCODER_BACKENDS:
        return rerank_with_local_cross_encoder(question, chunks)
    if backend in {"cognix-pair", "pair-mlp", "local-pair"}:
        return rerank_with_pair_model(question, chunks)
    model = load_cross_encoder()
    if model is None:
        return chunks
    pairs = [(question, f"{chunk.source_path}\n{chunk.excerpt}") for chunk in chunks]
    try:
        scores = model.predict(pairs)
    except Exception:
        return chunks
    reranked = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            excerpt=chunk.excerpt,
            score=round(float(score), 4),
            sensitivity=chunk.sensitivity,
        )
        for chunk, score in zip(chunks, scores)
    ]
    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


def rerank_with_local_cross_encoder(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank chunks with Cognix's local neural cross-encoder artifact."""
    model = load_local_cross_encoder_reranker()
    if model is None:
        return chunks
    reranked = []
    for chunk in chunks:
        probability = model.predict_proba(question, f"{chunk.source_path}\n{chunk.excerpt}").get("relevant", chunk.score)
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                excerpt=chunk.excerpt,
                score=round(float(probability), 4),
                sensitivity=chunk.sensitivity,
            )
        )
    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


def rerank_with_pair_model(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank chunks with a local trained Cognix pair model."""
    model = load_pair_reranker()
    if model is None:
        return chunks
    reranked = []
    for chunk in chunks:
        probability = model.predict_proba(question, f"{chunk.source_path}\n{chunk.excerpt}").get("relevant", chunk.score)
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                excerpt=chunk.excerpt,
                score=round(float(probability), 4),
                sensitivity=chunk.sensitivity,
            )
        )
    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


@lru_cache(maxsize=1)
def load_cross_encoder():
    """Load the configured cross-encoder lazily."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        return CrossEncoder(get_settings().cross_encoder_model)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_local_cross_encoder_reranker() -> TinyCrossEncoder | None:
    """Load the local trained reranker cross-encoder if present."""
    path = get_settings().resolved_local_cross_encoder_reranker_model_path()
    if not path.exists():
        return None
    try:
        return load_cross_encoder_model(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_pair_reranker() -> PairTextModel | None:
    """Load the local trained pair reranker if present."""
    path = get_settings().resolved_pair_reranker_model_path()
    if not path.exists():
        return None
    try:
        return load_pair_model(path)
    except Exception:
        return None
