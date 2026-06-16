from app.config import get_settings
from app.services.embeddings import EMBEDDING_MODEL, embed_text


def upsert_chunk(chunk_id: int, text: str, metadata: dict) -> None:
    collection = get_collection()
    vector = embed_text(text)
    collection.upsert(
        ids=[str(chunk_id)],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata],
    )


def query_chunks(question: str, limit: int) -> list[dict]:
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
