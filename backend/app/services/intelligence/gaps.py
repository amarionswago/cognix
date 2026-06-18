"""Knowledge gap detection for Cognix v2."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.intelligence.types import Detector, Finding, SourceRef

MIN_CONCEPT_MENTIONS = 5
CONCEPT_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}|[a-z]+(?:\s+[a-z]+){1,2})\b")
STOP_CONCEPTS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "your",
    "what",
    "when",
    "where",
    "which",
    "there",
    "these",
    "those",
    "page",
    "pages",
    "chapter",
    "section",
    "figure",
    "source",
    "sources",
    "suck",
    "chunk",
    "chunks",
    "file",
    "files",
    "http",
    "https",
    "post",
    "get",
    "if",
    "clone",
    "next",
}
REJECT_WORDS = {
    "and",
    "are",
    "going",
    "more",
    "need",
    "should",
    "the",
    "that",
    "they",
    "this",
    "use",
    "used",
    "using",
    "we",
    "you",
    "your",
}
DOMAIN_WORDS = {
    "agent",
    "agents",
    "ai",
    "api",
    "database",
    "embedding",
    "embeddings",
    "graph",
    "hacking",
    "intelligence",
    "kali",
    "knowledge",
    "learning",
    "linux",
    "llm",
    "machine",
    "memory",
    "model",
    "models",
    "network",
    "neural",
    "python",
    "quantization",
    "rag",
    "retrieval",
    "search",
    "security",
    "semantic",
    "transformer",
}
CANONICAL_CONCEPTS = {
    "ai": "artificial intelligence",
    "a i": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "machine intelligence": "artificial intelligence",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "semantic search": "semantic search",
    "vector search": "vector search",
    "retrieval augmented generation": "rag",
    "retrieval augmented": "rag",
    "large language model": "llm",
    "large language models": "llm",
    "language model": "llm",
    "language models": "llm",
}


class GapDetector(Detector):
    """Detect concepts that appear often but lack structured wiki pages."""

    def run(self, since: datetime | None = None) -> list[Finding]:
        """Extract concept mentions and return knowledge-gap findings."""
        rebuild_concept_mentions(limit=2000)
        rebuild_graph_edges()
        concepts = frequent_concepts(MIN_CONCEPT_MENTIONS)
        findings: list[Finding] = []
        for concept, mention_count, examples in concepts:
            if wiki_page_exists(concept):
                continue
            refs = [
                SourceRef(
                    source_path=example["source_path"],
                    chunk_id=int(example["chunk_id"]),
                    excerpt=example["excerpt"],
                )
                for example in examples[:3]
            ]
            findings.append(
                Finding(
                    finding_type="gap",
                    severity="warning",
                    title=f"Knowledge gap: {concept}",
                    description=(
                        f"`{concept}` is mentioned {mention_count} times across indexed evidence, "
                        "but Cognix has no structured wiki page for it. This is a concept-linking gap, "
                        "not proof that the source material is missing."
                    ),
                    source_refs=refs,
                    suggested_action=f"Compile a concept page for `{concept}` from the cited sources.",
                    confidence=min(0.95, 0.45 + mention_count / 20),
                    metadata={"concept": concept, "mention_count": mention_count},
                )
            )
        return findings[:25]


def rebuild_concept_mentions(limit: int = 2000) -> int:
    """Rebuild deterministic concept mentions from current chunks."""
    now = utc_now()
    inserted = 0
    with db_session() as conn:
        conn.execute("DELETE FROM concept_mentions")
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.file_id, c.text, c.source_path
            FROM chunks c
            JOIN raw_files rf ON rf.id = c.file_id
            WHERE rf.status='processed'
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        records = []
        for row in rows:
            counts = Counter(extract_concepts(str(row["text"])))
            for concept, count in counts.items():
                records.append(
                    (
                        concept,
                        normalize_concept(concept),
                        int(row["chunk_id"]),
                        int(row["file_id"]),
                        str(row["source_path"]),
                        count,
                        now,
                        now,
                    )
                )
        if records:
            conn.executemany(
                """
                INSERT INTO concept_mentions
                (concept, normalized_concept, chunk_id, file_id, source_path,
                 mention_count, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            inserted = len(records)
    return inserted


def frequent_concepts(threshold: int) -> list[tuple[str, int, list[dict]]]:
    """Return frequent concepts with example source refs."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT normalized_concept, MIN(concept) AS concept, SUM(mention_count) AS mentions
            FROM concept_mentions
            GROUP BY normalized_concept
            HAVING mentions >= ?
            ORDER BY mentions DESC, concept
            LIMIT 50
            """,
            (threshold,),
        ).fetchall()
        results = []
        for row in rows:
            examples = conn.execute(
                """
                SELECT cm.chunk_id, cm.source_path, substr(c.text, 1, 260) AS excerpt
                FROM concept_mentions cm
                JOIN chunks c ON c.id = cm.chunk_id
                WHERE cm.normalized_concept=?
                ORDER BY cm.mention_count DESC
                LIMIT 3
                """,
                (row["normalized_concept"],),
            ).fetchall()
            results.append((str(row["concept"]), int(row["mentions"]), [dict(example) for example in examples]))
    return results


def rebuild_graph_edges() -> int:
    """Build lightweight RELATED_TO graph edges from concept co-occurrence."""
    now = utc_now()
    records = []
    with db_session() as conn:
        conn.execute("DELETE FROM graph_edges WHERE relationship='RELATED_TO'")
        rows = conn.execute(
            """
            SELECT chunk_id, source_path, group_concat(DISTINCT normalized_concept) AS concepts
            FROM concept_mentions
            GROUP BY chunk_id, source_path
            HAVING COUNT(DISTINCT normalized_concept) >= 2
            LIMIT 1000
            """
        ).fetchall()
        for row in rows:
            concepts = sorted(set(str(row["concepts"]).split(",")))[:8]
            for index, source in enumerate(concepts):
                for target in concepts[index + 1 :]:
                    if source == target:
                        continue
                    records.append(
                        (
                            source,
                            target,
                            "RELATED_TO",
                            1.0,
                            str(row["source_path"]),
                            int(row["chunk_id"]),
                            "{}",
                            now,
                            now,
                        )
                    )
        if records:
            conn.executemany(
                """
                INSERT INTO graph_edges
                (source_concept, target_concept, relationship, weight, source_file,
                 source_chunk_id, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records[:5000],
            )
    return len(records)


def extract_concepts(text: str) -> list[str]:
    """Extract simple concept candidates from text."""
    concepts: list[str] = []
    for match in CONCEPT_RE.finditer(text[:8000]):
        concept = " ".join(match.group(0).split())
        normalized = normalize_concept(concept)
        if not is_plausible_concept(concept, normalized):
            continue
        concepts.append(concept)
    return concepts


def is_plausible_concept(concept: str, normalized: str) -> bool:
    """Return True for concept candidates precise enough for user-facing gaps."""
    words = normalized.split()
    if len(normalized) < 4 or normalized in STOP_CONCEPTS or normalized.isdigit():
        return False
    if any(word in REJECT_WORDS for word in words):
        return False
    if normalized.startswith(("http ", "https ", "git ")):
        return False
    if len(words) == 1:
        return concept.isupper() and normalized not in STOP_CONCEPTS
    return bool(DOMAIN_WORDS & set(words)) or any(word.isupper() for word in concept.split())


def wiki_page_exists(concept: str) -> bool:
    """Check whether a concept already has a wiki page."""
    slug = normalize_concept(concept).replace(" ", "-")
    concepts_dir = get_settings().resolved_wiki_dir() / "concepts"
    return any(path.exists() for path in [concepts_dir / f"{slug}.md", concepts_dir / f"{concept}.md"])


def normalize_concept(concept: str) -> str:
    """Normalize a concept for grouping."""
    normalized = re.sub(r"[^a-z0-9]+", " ", concept.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.endswith("ies") and len(normalized) > 5:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and len(normalized) > 5:
        normalized = normalized[:-1]
    return CANONICAL_CONCEPTS.get(normalized, normalized)
