from app.config import get_settings
from app.services.embeddings import EMBEDDING_MODEL, embed_text


def upsert_chunk(chunk_id: int, text: str, metadata: dict) -> None:
    upsert_chunks([(chunk_id, text, metadata, embed_text(text))])


def upsert_chunks(records: list[tuple[int, str, dict, list[float]]], batch_size: int = 128) -> None:
    if not records:
        return
    if not get_settings().chroma_enabled:
        return
    collection = get_collection()
    for index in range(0, len(records), batch_size):
        batch = records[index : index + batch_size]
        collection.upsert(
            ids=[str(chunk_id) for chunk_id, _text, _metadata, _vector in batch],
            embeddings=[vector for _chunk_id, _text, _metadata, vector in batch],
            documents=[text for _chunk_id, text, _metadata, _vector in batch],
            metadatas=[metadata for _chunk_id, _text, metadata, _vector in batch],
        )


def query_chunks(question: str, limit: int) -> list[dict]:
    if not get_settings().chroma_enabled:
        return []
    collection = get_collection()
    result = collection.query(query_embeddings=[embed_text(question)], n_results=limit)
    rows: list[dict] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        rows.append(
            {
                "chunk_id": int(chunk_id),
                "text": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "score": round(1.0 / (1.0 + max(0.0, distance)), 4),
            }
        )
    return rows


def get_collection():
    import chromadb

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.resolved_chroma_dir()))
    return client.get_or_create_collection(
        name="cognix_chunks",
        metadata={"description": "Cognix chunk embeddings", "model": EMBEDDING_MODEL},
    )
