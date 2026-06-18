import json
import re
from dataclasses import dataclass

from app.database import db_session
from app.services.chroma_store import query_chunks
from app.services.embeddings import cosine_similarity, embed_text
from app.services.providers import call_configured_model
from app.services.retrieval_types import RetrievedChunk


FILE_TYPE_ALIASES = {
    "csv": {".csv"},
    "eml": {".eml"},
    "email": {".eml", ".mbox"},
    "emails": {".eml", ".mbox"},
    "htm": {".htm"},
    "html": {".html", ".htm"},
    "json": {".json", ".jsonl"},
    "jsonl": {".jsonl"},
    "log": {".log"},
    "markdown": {".md", ".markdown"},
    "mbox": {".mbox"},
    "md": {".md", ".markdown"},
    "pdf": {".pdf"},
    "pdfs": {".pdf"},
    "py": {".py"},
    "python": {".py"},
    "text": {".txt", ".text"},
    "txt": {".txt", ".text"},
    "xml": {".xml"},
}


@dataclass(frozen=True)
class EvidencePack:
    """Typed retrieval result passed through answer synthesis and intelligence code."""

    question: str
    chunks: list[RetrievedChunk]
    subqueries: list[str]
    retrieval_method: str
    extension_filter: set[str] | None
    diagnostics: dict[str, object]


def retrieve(question: str, limit: int = 8) -> list[RetrievedChunk]:
    return retrieve_evidence(question, limit).chunks


def retrieve_evidence(question: str, limit: int = 8) -> EvidencePack:
    extension_filter = detect_file_type_filter(question)
    file_matches = retrieve_from_file_matches(question, limit)
    if file_matches:
        filtered_matches = filter_chunks_by_extension(file_matches, extension_filter)
        if filtered_matches:
            chunks = filtered_matches[:limit]
            return EvidencePack(question, chunks, [question], "filename-match", extension_filter, retrieval_diagnostics(question, chunks, extension_filter, [question]))
        if not extension_filter:
            chunks = file_matches[:limit]
            return EvidencePack(question, chunks, [question], "filename-match", extension_filter, retrieval_diagnostics(question, chunks, extension_filter, [question]))
    subqueries = decompose_query(question)
    ranked_lists: list[list[RetrievedChunk]] = []
    for subquery in subqueries:
        ranked_lists.append(retrieve_from_chroma(subquery, limit * 4, extension_filter))
        keyword_results = retrieve_by_keyword(subquery, limit * 4, extension_filter)
        ranked_lists.append(keyword_results)
        if not has_strong_keyword_coverage(keyword_results, limit, query_keyword_groups(subquery, extension_filter)):
            ranked_lists.append(retrieve_from_sqlite(subquery, limit * 4, extension_filter))
    candidates = reciprocal_rank_fusion(ranked_lists)
    if not candidates:
        return EvidencePack(question, [], subqueries, "rrf-hybrid", extension_filter, retrieval_diagnostics(question, [], extension_filter, subqueries))
    chunks = rerank_and_filter(question, candidates, limit, extension_filter)
    return EvidencePack(question, chunks, subqueries, "rrf-hybrid", extension_filter, retrieval_diagnostics(question, chunks, extension_filter, subqueries))


def has_strong_keyword_coverage(chunks: list[RetrievedChunk], limit: int, keyword_groups: list[set[str]]) -> bool:
    """Return True when exact retrieval is already strong enough to skip vector scan."""
    if len(keyword_groups) < 4:
        return False
    if len(chunks) < max(3, min(limit, 6)):
        return False
    return sum(1 for chunk in chunks[:limit] if chunk.score >= 0.75) >= max(3, min(limit, 6))


def retrieval_diagnostics(
    question: str,
    chunks: list[RetrievedChunk],
    extension_filter: set[str] | None,
    subqueries: list[str],
) -> dict[str, object]:
    """Return user-facing retrieval evidence diagnostics."""
    unique_sources = sorted({chunk.source_path for chunk in chunks})
    scores = [chunk.score for chunk in chunks]
    keyword_groups = query_keyword_groups(question, extension_filter)
    keyword_hits = sum(
        1
        for group in keyword_groups
        if any(any(variant in f"{chunk.source_path} {chunk.excerpt}".lower() for variant in group) for chunk in chunks)
    )
    notes: list[str] = []
    if not chunks:
        notes.append("No indexed chunks passed retrieval filters.")
    if extension_filter:
        notes.append(f"Restricted retrieval to file types: {', '.join(sorted(extension_filter))}.")
    if chunks and len(unique_sources) == 1:
        notes.append("All evidence came from one source; verify if the question needs broader coverage.")
    if keyword_groups and keyword_hits < max(1, min(2, len(keyword_groups))):
        notes.append("Retrieved evidence has weak keyword coverage for the question.")
    if scores and max(scores) < 0.25:
        notes.append("Top retrieval score is low; answer confidence should be treated cautiously.")
    return {
        "chunk_count": len(chunks),
        "unique_source_count": len(unique_sources),
        "unique_sources": unique_sources[:10],
        "max_score": round(max(scores), 4) if scores else 0.0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "extension_filter": sorted(extension_filter) if extension_filter else [],
        "subquery_count": len(subqueries),
        "keyword_group_count": len(keyword_groups),
        "keyword_group_hits": keyword_hits,
        "notes": notes,
    }


def retrieve_from_file_matches(question: str, limit: int = 8) -> list[RetrievedChunk]:
    file_ids = matching_file_ids(question, max_files=4)
    if not file_ids:
        return []
    placeholders = ",".join("?" for _ in file_ids)
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.source_path, c.sensitivity, c.chunk_index, rf.id AS file_id
            FROM chunks c
            JOIN raw_files rf ON rf.id = c.file_id
            WHERE rf.id IN ({placeholders})
              AND rf.status='processed'
            ORDER BY CASE rf.id
                {' '.join(f'WHEN ? THEN {index}' for index, _file_id in enumerate(file_ids))}
                ELSE {len(file_ids)}
              END,
              c.chunk_index
            LIMIT ?
            """,
            tuple(file_ids + file_ids + [limit]),
        ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=int(row["id"]),
            source_path=str(row["source_path"]),
            excerpt=_excerpt(str(row["text"]), 900),
            score=2.5,
            sensitivity=str(row["sensitivity"]),
        )
        for row in rows
    ]


def matching_file_ids(question: str, max_files: int = 4) -> list[int]:
    normalized_question = normalize_path_text(question)
    query_tokens = file_query_tokens(question)
    if not normalized_question or not query_tokens:
        return []

    scored: list[tuple[float, int]] = []
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT id, relative_path
            FROM raw_files
            WHERE status='processed'
            """
        ).fetchall()

    for row in rows:
        relative_path = str(row["relative_path"])
        normalized_path = normalize_path_text(relative_path)
        filename = relative_path.rsplit("/", 1)[-1]
        normalized_filename = normalize_path_text(filename)
        stem = filename.rsplit(".", 1)[0]
        normalized_stem = normalize_path_text(stem)
        hits = sum(1 for token in query_tokens if token in normalized_path.split())
        filename_hits = sum(1 for token in query_tokens if token in normalized_filename.split())
        score = 0.0
        if normalized_filename and normalized_filename in normalized_question:
            score += 6.0
        if normalized_stem and normalized_stem in normalized_question:
            score += 5.0
        if hits:
            score += hits
        if filename_hits:
            score += filename_hits * 1.5
        required_hits = min(2, len([token for token in query_tokens if not token.isdigit()]))
        if len(query_tokens) > 1 and hits < required_hits:
            continue
        if len(query_tokens) == 1 and not filename_hits:
            continue
        if score > 0:
            scored.append((score, int(row["id"])))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [file_id for _score, file_id in scored[:max_files]]


def retrieve_from_chroma(question: str, limit: int, extension_filter: set[str] | None = None) -> list[RetrievedChunk]:
    try:
        rows = query_chunks(question, limit * 3)
    except Exception:
        return []
    valid_ids = valid_chunk_ids([int(row["chunk_id"]) for row in rows], extension_filter)
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


def valid_chunk_ids(chunk_ids: list[int], extension_filter: set[str] | None = None) -> set[int]:
    if not chunk_ids:
        return set()
    placeholders = ",".join("?" for _ in chunk_ids)
    extension_clause = ""
    params: list[str | int] = list(chunk_ids)
    if extension_filter:
        extension_placeholders = ",".join("?" for _ in extension_filter)
        extension_clause = f" AND rf.extension IN ({extension_placeholders})"
        params.extend(sorted(extension_filter))
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id
            FROM chunks c
            JOIN raw_files rf ON rf.id = c.file_id
            WHERE c.id IN ({placeholders})
              AND rf.status='processed'
              AND rf.relative_path NOT LIKE '%.gitkeep'
              {extension_clause}
            """,
            tuple(params),
        ).fetchall()
    return {int(row["id"]) for row in rows}


def retrieve_from_sqlite(question: str, limit: int = 8, extension_filter: set[str] | None = None) -> list[RetrievedChunk]:
    query_vector = embed_text(question)
    keywords = query_keywords(question, extension_filter)
    keyword_groups = query_keyword_groups(question, extension_filter)
    scored: list[RetrievedChunk] = []

    extension_clause = ""
    params: list[str] = []
    if extension_filter:
        extension_placeholders = ",".join("?" for _ in extension_filter)
        extension_clause = f" WHERE rf.extension IN ({extension_placeholders})"
        params.extend(sorted(extension_filter))

    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.source_path, c.sensitivity, e.vector_json
            FROM chunks c
            JOIN chunk_embeddings e ON e.chunk_id = c.id
            JOIN raw_files rf ON rf.id = c.file_id
            {extension_clause}
            """,
            tuple(params),
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


def retrieve_by_keyword(question: str, limit: int = 8, extension_filter: set[str] | None = None) -> list[RetrievedChunk]:
    """Retrieve chunks by exact keyword/concept overlap."""
    keyword_groups = query_keyword_groups(question, extension_filter)
    if not keyword_groups:
        return []
    extension_clause = ""
    params: list[str] = []
    if extension_filter:
        extension_placeholders = ",".join("?" for _ in extension_filter)
        extension_clause = f" AND rf.extension IN ({extension_placeholders})"
        params.extend(sorted(extension_filter))
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.source_path, c.sensitivity
            FROM chunks c
            JOIN raw_files rf ON rf.id = c.file_id
            WHERE rf.status='processed'
              {extension_clause}
            """,
            tuple(params),
        ).fetchall()

    scored: list[RetrievedChunk] = []
    for row in rows:
        text = f"{row['source_path']} {row['text']}".lower()
        hits = keyword_group_hits(keyword_groups, text)
        if hits == 0:
            continue
        score = hits / max(1, len(keyword_groups))
        scored.append(
            RetrievedChunk(
                chunk_id=int(row["id"]),
                source_path=str(row["source_path"]),
                excerpt=_excerpt(str(row["text"]), 700),
                score=round(score, 4),
                sensitivity=str(row["sensitivity"]),
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Merge ranked retrieval lists with Reciprocal Rank Fusion."""
    fused_scores: dict[int, float] = {}
    best_chunk: dict[int, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            current = best_chunk.get(chunk.chunk_id)
            if not current or chunk.score > current.score:
                best_chunk[chunk.chunk_id] = chunk
    fused = [
        RetrievedChunk(
            chunk_id=chunk_id,
            source_path=chunk.source_path,
            excerpt=chunk.excerpt,
            score=round(fused_scores[chunk_id], 4),
            sensitivity=chunk.sensitivity,
        )
        for chunk_id, chunk in best_chunk.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)


def rerank_and_filter(
    question: str,
    chunks: list[RetrievedChunk],
    limit: int,
    extension_filter: set[str] | None = None,
) -> list[RetrievedChunk]:
    chunks = filter_chunks_by_extension(chunks, extension_filter)
    keywords = query_keywords(question, extension_filter)
    keyword_groups = query_keyword_groups(question, extension_filter)
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
    return apply_optional_reranker(question, filtered[: max(limit * 3, limit)])[:limit]


def apply_optional_reranker(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Apply trained reranking when enabled, without making retrieval depend on it."""
    try:
        from app.services.reranker import rerank_with_optional_model
    except Exception:
        return chunks
    return rerank_with_optional_model(question, chunks)


def decompose_query(question: str) -> list[str]:
    """Split complex questions into retrieval subqueries with deterministic fallback."""
    if not is_complex_query(question):
        return [question]
    prompt = (
        "Break this research question into 2-4 concise search queries. "
        "Return only a JSON array of strings.\n\n"
        f"Question: {question}"
    )
    response = call_configured_model(prompt)
    subqueries = parse_subqueries(response or "")
    if subqueries:
        return [question, *subqueries[:4]]
    parts = re.split(r"\b(?:and|versus|vs|between|relationship|difference|compare)\b", question, flags=re.IGNORECASE)
    cleaned = [part.strip(" ?.,;:") for part in parts if len(part.strip()) >= 4]
    return list(dict.fromkeys([question, *cleaned]))[:5]


def parse_subqueries(raw: str) -> list[str]:
    """Parse JSON subqueries returned by a model."""
    if not raw.strip():
        return []
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if isinstance(item, str) and len(item.strip()) >= 3]


def is_complex_query(question: str) -> bool:
    """Return True when a question likely benefits from decomposition."""
    lowered = normalize_search_text(question)
    if len(lowered.split()) >= 14:
        return True
    return any(term in lowered for term in ("compare", "relationship", "difference", "contradict", "changed", "between", "versus", " vs "))


def query_keywords(question: str, extension_filter: set[str] | None = None) -> set[str]:
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
        "document",
        "documents",
        "evidence",
        "define",
        "explain",
        "file",
        "files",
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
        "library",
        "source",
        "sources",
        "say",
        "says",
        "talk",
        "talking",
        "up",
        "any",
        "all",
        "some",
    }
    extension_keywords = file_type_keywords_for_extensions(extension_filter)
    keywords = set()
    for word in re.findall(r"[A-Za-z0-9_']+", text):
        lowered = word.lower()
        normalized = normalize_keyword(lowered)
        if (len(lowered) <= 2 and lowered not in {"ai", "ml"}) or lowered in stopwords or normalized in extension_keywords or is_noise_token(lowered):
            continue
        keywords.add(normalized)
    return keywords


def query_keyword_groups(question: str, extension_filter: set[str] | None = None) -> list[set[str]]:
    return [keyword_variants(keyword) for keyword in sorted(query_keywords(question, extension_filter))]


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
    if keyword in {"ai", "artificial", "intelligence"}:
        variants.update({"ai", "artificial intelligence", "machine intelligence"})
    if keyword in {"ml", "machine", "learn"}:
        variants.update({"ml", "machine learning"})
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


def detect_file_type_filter(question: str) -> set[str] | None:
    lowered = normalize_search_text(question)
    normalized = normalize_path_text(question)
    extensions: set[str] = set()

    for raw_extension in re.findall(r"\.[a-z0-9]+", lowered):
        extensions.update(FILE_TYPE_ALIASES.get(raw_extension[1:], set()))

    words = normalized.split()
    file_context_words = {"document", "documents", "file", "files", "source", "sources"}
    for index, word in enumerate(words):
        if word not in FILE_TYPE_ALIASES:
            continue
        previous_word = words[index - 1] if index else ""
        next_word = words[index + 1] if index + 1 < len(words) else ""
        plural_type = word.endswith("s") and word[:-1] in FILE_TYPE_ALIASES
        strongly_typed = word in {"pdf", "pdfs", "csv", "json", "jsonl", "xml", "eml", "mbox"}
        if previous_word in file_context_words or next_word in file_context_words or plural_type or strongly_typed:
            extensions.update(FILE_TYPE_ALIASES[word])

    return extensions or None


def file_type_keywords_for_extensions(extension_filter: set[str] | None) -> set[str]:
    if not extension_filter:
        return set()
    keywords = set()
    for keyword, extensions in FILE_TYPE_ALIASES.items():
        if extensions & extension_filter:
            keywords.add(normalize_keyword(keyword))
    return keywords


def filter_chunks_by_extension(
    chunks: list[RetrievedChunk],
    extension_filter: set[str] | None,
) -> list[RetrievedChunk]:
    if not extension_filter:
        return chunks
    return [chunk for chunk in chunks if source_extension(chunk.source_path) in extension_filter]


def source_extension(source_path: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", source_path)
    return match.group(1).lower() if match else ""


def file_query_tokens(question: str) -> list[str]:
    stopwords = {
        "about",
        "answer",
        "content",
        "contents",
        "document",
        "file",
        "find",
        "inside",
        "look",
        "open",
        "read",
        "search",
        "library",
        "say",
        "says",
        "show",
        "summarize",
        "summary",
        "tell",
        "the",
        "this",
        "what",
        "where",
        "which",
    }
    tokens = []
    for token in normalize_path_text(question).split():
        if (len(token) <= 2 and not token.isdigit()) or token in stopwords or is_noise_token(token):
            continue
        tokens.append(normalize_keyword(token))
    return list(dict.fromkeys(tokens))


def normalize_path_text(text: str) -> str:
    text = normalize_search_text(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
