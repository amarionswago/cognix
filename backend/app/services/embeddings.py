import hashlib
import json
import math
import re
from functools import lru_cache

from app.config import get_settings
from app.database import db_session, utc_now


EMBEDDING_MODEL = "cognix-hash-embedding-v1"
SENTENCE_TRANSFORMER_PROVIDER = "sentence-transformers"
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    neural_vector = embed_text_with_sentence_transformer(text)
    if neural_vector:
        return neural_vector
    return embed_text_with_hash(text, dimensions)


def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    neural_vectors = embed_texts_with_sentence_transformer(texts)
    if neural_vectors:
        return neural_vectors
    return [embed_text_with_hash(text, dimensions) for text in texts]


def active_embedding_model() -> str:
    settings = get_settings()
    if should_use_sentence_transformer() and sentence_transformer_available():
        return settings.sentence_transformer_model
    return EMBEDDING_MODEL


def active_embedding_provider() -> str:
    if should_use_sentence_transformer() and sentence_transformer_available():
        return SENTENCE_TRANSFORMER_PROVIDER
    return "local"


def embed_text_with_hash(text: str, dimensions: int | None = None) -> list[float]:
    settings = get_settings()
    dims = dimensions or settings.embedding_dimensions
    vector = [0.0] * dims
    words = WORD_RE.findall(text.lower())
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = -1.0 if digest[4] % 2 else 1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def should_use_sentence_transformer() -> bool:
    backend = get_settings().local_embedding_backend.strip().lower()
    return backend in {"sentence-transformer", "sentence-transformers", "neural", "local-neural"}


def embed_text_with_sentence_transformer(text: str) -> list[float]:
    vectors = embed_texts_with_sentence_transformer([text])
    return vectors[0] if vectors else []


def embed_texts_with_sentence_transformer(texts: list[str]) -> list[list[float]]:
    if not texts or not should_use_sentence_transformer():
        return []
    model = load_sentence_transformer()
    if model is None:
        return []
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[round(float(value), 6) for value in vector] for vector in vectors]


@lru_cache(maxsize=1)
def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(get_settings().sentence_transformer_model)
    except Exception:
        return None


def sentence_transformer_available() -> bool:
    return load_sentence_transformer() is not None


def store_chunk_embedding(chunk_id: int, text: str) -> None:
    vector = embed_text(text)
    store_chunk_embeddings([(chunk_id, vector)])


def store_chunk_embeddings(
    records: list[tuple[int, list[float]]],
    provider: str | None = None,
    model: str | None = None,
) -> None:
    if not records:
        return
    resolved_provider = provider or active_embedding_provider()
    resolved_model = model or active_embedding_model()
    now = utc_now()
    with db_session() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunk_embeddings)").fetchall()}
        has_provider_metadata = {"provider", "embedding_source"} <= columns
        if has_provider_metadata:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                (chunk_id, vector_json, model, dimensions, created_at, provider, embedding_source)
                VALUES (?, ?, ?, ?, ?, ?, 'chunk')
                """,
                [
                    (chunk_id, json.dumps(vector), resolved_model, len(vector), now, resolved_provider)
                    for chunk_id, vector in records
                ],
            )
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_embeddings
            (chunk_id, vector_json, model, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (chunk_id, json.dumps(vector), resolved_model, len(vector), now)
                for chunk_id, vector in records
            ],
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
