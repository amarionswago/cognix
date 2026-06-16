import hashlib
import json
import math
import re

from app.config import get_settings
from app.database import db_session, utc_now


EMBEDDING_MODEL = "cognix-hash-embedding-v1"
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
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


def store_chunk_embedding(chunk_id: int, text: str) -> None:
    vector = embed_text(text)
    with db_session() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings
            (chunk_id, vector_json, model, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chunk_id, json.dumps(vector), EMBEDDING_MODEL, len(vector), utc_now()),
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))

