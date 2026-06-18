"""Claim extraction and storage for Cognix v2.

Claims are the atomic facts Cognix audits. They power contradiction detection,
staleness checks, confidence penalties, and future fine-tuning data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.embedding_router import EmbeddingPolicy, embed_with_routing
from app.database import db_session, utc_now
from app.services.chunking import estimate_tokens
from app.services.providers import call_configured_model

MIN_CLAIM_TOKENS = 50
MAX_CLAIMS_PER_CHUNK = 8
CLAIM_TYPES = {"factual", "statistical", "temporal", "causal", "definition", "recommendation", "opinion"}
DEFINITION_RE = re.compile(r"\b([A-Z][A-Za-z0-9 _-]{2,80})\s+(?:is|are|refers to|means)\s+([^.!?]{12,220})[.!?]")
NUMBERED_FACT_RE = re.compile(r"([^.!?]*(?:\d{4}|\d+(?:\.\d+)?\s?%|\$\d+)[^.!?]{12,220})[.!?]")


@dataclass(frozen=True)
class ExtractedClaim:
    """A factual assertion extracted from a chunk."""

    claim: str
    confidence: float
    claim_type: str


def extract_claims_from_text(text: str, use_llm: bool = True) -> list[ExtractedClaim]:
    """Extract claims from a chunk using provider routing with deterministic fallback."""
    if should_skip_claim_extraction(text):
        return []
    if use_llm:
        prompt = build_claim_prompt(text)
        response = call_configured_model(prompt)
        claims = parse_claim_json(response or "")
        if claims:
            return claims[:MAX_CLAIMS_PER_CHUNK]
    return deterministic_claims(text)[:MAX_CLAIMS_PER_CHUNK]


def store_claims_for_chunk(chunk_id: int, file_id: int, claims: list[ExtractedClaim], sensitivity: str = "research") -> int:
    """Store extracted claims for a chunk and return the inserted count."""
    if not claims:
        return 0
    texts = [claim.claim for claim in claims]
    embeddings = embed_with_routing(texts, EmbeddingPolicy(sensitivity=sensitivity, allow_cloud=False))
    now = utc_now()
    with db_session() as conn:
        conn.execute("DELETE FROM claims WHERE chunk_id=?", (chunk_id,))
        conn.executemany(
            """
            INSERT INTO claims
            (chunk_id, file_id, claim_text, claim_type, confidence, embedding_json,
             embedding_provider, embedding_model, embedding_dimensions, source_date,
             ingest_date, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'active', ?, ?)
            """,
            [
                (
                    chunk_id,
                    file_id,
                    claim.claim,
                    claim.claim_type,
                    claim.confidence,
                    json.dumps(embedding.vector),
                    embedding.provider,
                    embedding.model,
                    embedding.dimensions,
                    now,
                    now,
                    now,
                )
                for claim, embedding in zip(claims, embeddings)
            ],
        )
    return len(claims)


def extract_claims_for_recent_chunks(limit: int = 100, use_llm: bool = True) -> int:
    """Extract claims for chunks that do not yet have claim records."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.file_id, c.text, c.sensitivity
            FROM chunks c
            LEFT JOIN claims cl ON cl.chunk_id = c.id
            WHERE cl.id IS NULL
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    inserted = 0
    for row in rows:
        claims = extract_claims_from_text(str(row["text"]), use_llm=use_llm)
        inserted += store_claims_for_chunk(
            int(row["chunk_id"]),
            int(row["file_id"]),
            claims,
            str(row["sensitivity"]),
        )
    return inserted


def should_skip_claim_extraction(text: str) -> bool:
    """Return True for chunks unlikely to contain useful factual claims."""
    if estimate_tokens(text) < MIN_CLAIM_TOKENS:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    code_markers = sum(stripped.count(marker) for marker in ("def ", "class ", "import ", "{", "}", "=>", "const "))
    return code_markers >= 8


def build_claim_prompt(text: str) -> str:
    """Build the structured claim extraction prompt."""
    return (
        "You are a claim extractor. Given the following text chunk, extract factual claims. "
        "A claim is a statement that asserts something is true and could in principle be verified or contradicted. "
        "Return a JSON array of objects with keys: claim, confidence, type. "
        "Allowed type values: factual, statistical, temporal, causal, definition, recommendation, opinion. "
        "Return only JSON. No markdown fences.\n\n"
        f"Text chunk:\n{text[:5000]}"
    )


def parse_claim_json(raw: str) -> list[ExtractedClaim]:
    """Parse and validate LLM claim JSON."""
    if not raw.strip():
        return []
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    claims: list[ExtractedClaim] = []
    for item in data:
        parsed = claim_from_mapping(item)
        if parsed:
            claims.append(parsed)
    return claims


def claim_from_mapping(item: Any) -> ExtractedClaim | None:
    """Validate a single claim dictionary from model output."""
    if not isinstance(item, dict):
        return None
    claim = str(item.get("claim") or "").strip()
    claim_type = str(item.get("type") or item.get("claim_type") or "factual").strip().lower()
    if claim_type not in CLAIM_TYPES:
        claim_type = "factual"
    try:
        confidence = float(item.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))
    if len(claim.split()) < 5:
        return None
    return ExtractedClaim(claim=claim, confidence=confidence, claim_type=claim_type)


def deterministic_claims(text: str) -> list[ExtractedClaim]:
    """Extract obvious claims without an LLM."""
    claims: list[ExtractedClaim] = []
    compact = " ".join(text.split())
    for match in DEFINITION_RE.finditer(compact):
        subject = match.group(1).strip()
        definition = match.group(2).strip()
        claims.append(
            ExtractedClaim(
                claim=f"{subject} is {definition}.",
                confidence=0.55,
                claim_type="definition",
            )
        )
    for match in NUMBERED_FACT_RE.finditer(compact):
        claim = match.group(1).strip()
        if len(claim.split()) >= 5:
            claims.append(ExtractedClaim(claim=f"{claim}.", confidence=0.5, claim_type="factual"))
    return dedupe_claims(claims)


def dedupe_claims(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Remove duplicate claim text while preserving order."""
    seen: set[str] = set()
    unique: list[ExtractedClaim] = []
    for claim in claims:
        key = claim.claim.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(claim)
    return unique
