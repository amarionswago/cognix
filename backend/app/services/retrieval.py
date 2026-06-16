import json
import re
from dataclasses import dataclass

from app.database import db_session
from app.services.chroma_store import query_chunks
from app.services.embeddings import cosine_similarity, embed_text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source_path: str
    excerpt: str
    score: float
    sensitivity: str


def retrieve(question: str, limit: int = 8) -> list[RetrievedChunk]:
    candidates = retrieve_from_chroma(question, limit * 3) + retrieve_from_sqlite(question, limit * 3)
    if not candidates:
        return []
    return rerank_and_filter(question, candidates, limit)


def retrieve_from_chroma(question: str, limit: int) -> list[RetrievedChunk]:
    try:
        rows = query_chunks(question, limit * 3)
    except Exception:
        return []
    valid_ids = valid_chunk_ids([int(row["chunk_id"]) for row in rows])
    chunks: list[RetrievedChunk] = []
    for row in rows:
        if int(row["chunk_id"]) not in valid_ids:
            continue
        metadata = row["metadata"] or {}
        source_path = str(metadata.get("source_path", ""))
        if source_path.endswith(".gitkeep"):
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                source_path=source_path,
                excerpt=_excerpt(row["text"], 700),
                score=float(row["score"]),
                sensitivity=str(metadata.get("sensitivity", "research")),
            )
        )
        if len(chunks) >= limit:
            break
    return chunks


def valid_chunk_ids(chunk_ids: list[int]) -> set[int]:
    if not chunk_ids:
        return set()
    placeholders = ",".join("?" for _ in chunk_ids)
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id
            FROM chunks c
            JOIN raw_files rf ON rf.id = c.file_id
            WHERE c.id IN ({placeholders})
              AND rf.status='processed'
              AND rf.relative_path NOT LIKE '%.gitkeep'
            """,
            tuple(chunk_ids),
        ).fetchall()
    return {int(row["id"]) for row in rows}


def retrieve_from_sqlite(question: str, limit: int = 8) -> list[RetrievedChunk]:
    query_vector = embed_text(question)
    keywords = query_keywords(question)
    keyword_groups = query_keyword_groups(question)
    scored: list[RetrievedChunk] = []

    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.text, c.source_path, c.sensitivity, e.vector_json
            FROM chunks c
            JOIN chunk_embeddings e ON e.chunk_id = c.id
            """
        ).fetchall()

    for row in rows:
        vector = json.loads(row["vector_json"])
        semantic_score = cosine_similarity(query_vector, vector)
        text_lower = f"{row['source_path']} {row['text']}".lower()
        keyword_hits = keyword_group_hits(keyword_groups, text_lower)
        keyword_score = min(0.35, keyword_hits * 0.05)
        if keywords and keyword_hits == 0:
            continue
        score = semantic_score + keyword_score
        scored.append(
            RetrievedChunk(
                chunk_id=row["id"],
                source_path=row["source_path"],
                excerpt=_excerpt(row["text"], 700),
                score=round(float(score), 4),
                sensitivity=row["sensitivity"],
            )
        )

    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def rerank_and_filter(question: str, chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    keywords = query_keywords(question)
    keyword_groups = query_keyword_groups(question)
    best_by_id: dict[int, RetrievedChunk] = {}
    scored: list[tuple[float, int, RetrievedChunk]] = []

    for chunk in chunks:
        current = best_by_id.get(chunk.chunk_id)
        if not current or chunk.score > current.score:
            best_by_id[chunk.chunk_id] = chunk

    for chunk in best_by_id.values():
        text = f"{chunk.source_path} {chunk.excerpt}".lower()
        hits = keyword_group_hits(keyword_groups, text)
        required_hits = required_keyword_hits(question, keyword_groups)
        if keyword_groups and hits == 0:
            continue
        if is_keyword_search(question, keywords) and hits < required_hits:
            continue
        coverage = hits / max(1, len(keyword_groups))
        source_bonus = keyword_group_hits(keyword_groups, chunk.source_path.lower()) * 0.15
        reranked_score = float(chunk.score) + coverage + source_bonus
        scored.append((reranked_score, hits, RetrievedChunk(chunk.chunk_id, chunk.source_path, chunk.excerpt, round(reranked_score, 4), chunk.sensitivity)))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = scored[0][0]
    filtered = [item[2] for item in scored if item[0] >= max(0.12, best_score * 0.45)]
    return filtered[:limit]


def query_keywords(question: str) -> set[str]:
    normalized_question = normalize_search_text(question)
    explicit = explicit_keyword_query(normalized_question)
    text = explicit or normalized_question
    stopwords = {
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "the",
        "and",
        "for",
        "with",
        "about",
        "from",
        "this",
        "that",
        "does",
        "into",
        "define",
        "explain",
        "search",
        "searched",
        "find",
        "show",
        "keyword",
        "keywords",
        "particular",
        "specific",
        "related",
        "data",
        "please",
        "give",
        "tell",
        "me",
        "look",
        "up",
        "any",
        "all",
        "some",
    }
    keywords = set()
    for word in re.findall(r"[A-Za-z0-9_']+", text):
        lowered = word.lower()
        if len(lowered) <= 2 or lowered in stopwords or is_noise_token(lowered):
            continue
        keywords.add(normalize_keyword(lowered))
    return keywords


def query_keyword_groups(question: str) -> list[set[str]]:
    return [keyword_variants(keyword) for keyword in sorted(query_keywords(question))]


def keyword_group_hits(keyword_groups: list[set[str]], text: str) -> int:
    return sum(1 for variants in keyword_groups if any(variant in text for variant in variants))


def required_keyword_hits(question: str, keyword_groups: list[set[str]]) -> int:
    if not keyword_groups:
        return 0
    if len(keyword_groups) <= 2:
        return len(keyword_groups)
    if is_keyword_search(question, {next(iter(group)) for group in keyword_groups}):
        return max(1, len(keyword_groups) - 1)
    return max(1, min(len(keyword_groups), 2))


def normalize_keyword(word: str) -> str:
    if word in {"pdfs", "pdf"}:
        return "pdf"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ers") and len(word) > 5:
        return word[:-1]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def keyword_variants(keyword: str) -> set[str]:
    variants = {keyword}
    if keyword == "pdf":
        variants.update({".pdf", "pdfs"})
    if keyword == "hack":
        variants.update({"hacking", "hacker", "hackers", "hacked"})
    if keyword == "hacker":
        variants.update({"hack", "hacking", "hackers"})
    if keyword.endswith("k"):
        variants.add(keyword + "ing")
        variants.add(keyword + "er")
        variants.add(keyword + "ers")
    return variants


def is_keyword_search(question: str, keywords: set[str]) -> bool:
    lowered = normalize_search_text(question)
    if len(keywords) <= 3:
        return True
    return any(term in lowered for term in ("keyword", "search", "find", "show me", "mentions"))


def explicit_keyword_query(question: str) -> str:
    match = re.search(r"\b(?:keyword|keywords|search|find|mentions)\b[:\s-]+(.+)$", question, flags=re.IGNORECASE)
    if not match:
        return ""
    trailing = match.group(1).strip()
    trailing = re.sub(r"^(?:for|about|a|an|the|particular|specific)\s+", "", trailing, flags=re.IGNORECASE)
    return trailing


def normalize_search_text(text: str) -> str:
    # Users often stretch letters for emphasis. Collapse very long runs so
    # "keywwwwword" behaves like "keyword", without changing normal doubles.
    return re.sub(r"([A-Za-z])\1{2,}", r"\1", text.lower())


def is_noise_token(token: str) -> bool:
    if len(set(token)) <= 2 and len(token) >= 6:
        return True
    return False


def _excerpt(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."
