"""Hybrid embedding router for Cognix v2.

The router preserves Cognix as a customer-owned system: local embeddings are the
default and guaranteed path. Cloud embeddings are opt-in and only used when the
owner enables them and a working provider key exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.services.embeddings import active_embedding_model, active_embedding_provider, embed_texts as embed_local_texts
from app.services.providers import get_provider_row, resolve_provider_key

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
LOCAL_PROVIDER = "local"
OPENAI_PROVIDER = "openai"
NEVER_CLOUD_SENSITIVITY = {"secret", "sensitive", "private"}


@dataclass(frozen=True)
class EmbeddingRecord:
    """A single embedded text with provider/model metadata."""

    text: str
    vector: list[float]
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class EmbeddingPolicy:
    """Controls whether a batch may use cloud embeddings."""

    sensitivity: str = "research"
    allow_cloud: bool = False


def embed_with_routing(texts: list[str], policy: EmbeddingPolicy | None = None) -> list[EmbeddingRecord]:
    """Embed text using the best allowed backend for this local Cognix instance."""
    if not texts:
        return []
    active_policy = policy or EmbeddingPolicy()
    settings = get_settings()
    if should_use_openai(active_policy):
        try:
            return embed_with_openai(texts, settings.openai_embedding_model)
        except Exception:
            # Embedding should never make ingest or intelligence passes unusable.
            pass
    vectors = embed_local_texts(texts, settings.embedding_dimensions)
    return [
            EmbeddingRecord(
                text=text,
                vector=vector,
                provider=active_embedding_provider(),
                model=active_embedding_model(),
                dimensions=len(vector),
            )
        for text, vector in zip(texts, vectors)
    ]


def should_use_openai(policy: EmbeddingPolicy) -> bool:
    """Return True only when cloud embeddings are explicitly allowed."""
    settings = get_settings()
    if not settings.cloud_embeddings_enabled or not policy.allow_cloud:
        return False
    if policy.sensitivity.lower() in NEVER_CLOUD_SENSITIVITY:
        return False
    key = resolve_provider_key(OPENAI_PROVIDER, get_provider_row(OPENAI_PROVIDER))
    return bool(key.value)


def embed_with_openai(texts: list[str], model: str) -> list[EmbeddingRecord]:
    """Embed text with OpenAI's embeddings endpoint."""
    key = resolve_provider_key(OPENAI_PROVIDER, get_provider_row(OPENAI_PROVIDER))
    if not key.value:
        raise RuntimeError("OpenAI embeddings requested but no API key is configured.")
    response = httpx.post(
        OPENAI_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {key.value}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": texts,
            "encoding_format": "float",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    by_index = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [item["embedding"] for item in by_index]
    if len(vectors) != len(texts):
        raise RuntimeError("OpenAI embeddings response did not match input batch length.")
    return [
        EmbeddingRecord(
            text=text,
            vector=[float(value) for value in vector],
            provider=OPENAI_PROVIDER,
            model=str(data.get("model") or model),
            dimensions=len(vector),
        )
        for text, vector in zip(texts, vectors)
    ]
