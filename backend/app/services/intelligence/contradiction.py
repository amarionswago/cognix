"""Contradiction candidate detection for Cognix v2."""

from __future__ import annotations

import json
import re
from datetime import datetime

from app.database import db_session
from app.services.claims import extract_claims_for_recent_chunks
from app.services.embeddings import cosine_similarity
from app.services.intelligence.types import Detector, Finding, SourceRef
from app.services.nli import classify_with_optional_nli
from app.services.providers import call_configured_model

SIMILARITY_THRESHOLD = 0.78
MAX_CANDIDATES = 30
NEGATION_TERMS = {"not", "never", "no", "cannot", "can't", "doesn't", "do not", "isn't", "false"}
STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
OPPOSING_TERMS = {
    "increase": {"decrease", "reduce", "reduces", "lower", "less", "harm", "harms"},
    "increases": {"decreases", "reduces", "lowers", "harms"},
    "higher": {"lower", "less"},
    "more": {"less", "fewer"},
    "improves": {"harms", "reduces", "worsens"},
    "supports": {"contradicts", "refutes"},
    "true": {"false"},
    "safe": {"unsafe"},
}


class ContradictionDetector(Detector):
    """Find claim pairs that may contradict each other."""

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    def run(self, since: datetime | None = None) -> list[Finding]:
        """Extract missing claims and evaluate likely contradiction pairs."""
        extract_claims_for_recent_chunks(limit=150, use_llm=self.use_llm)
        pairs = candidate_claim_pairs()
        findings: list[Finding] = []
        for left, right, similarity in pairs:
            verdict = judge_contradiction(left["claim_text"], right["claim_text"], use_llm=self.use_llm)
            verdict_label, verdict_confidence, verdict_backend = verdict
            if verdict_label == "unrelated":
                continue
            confirmed = verdict_label == "contradiction"
            findings.append(
                Finding(
                    finding_type="contradiction" if confirmed else "contradiction_candidate",
                    severity="error" if confirmed else "warning",
                    title="Contradiction found" if confirmed else "Contradiction candidate",
                    description=(
                        f"Claim A: {left['claim_text']}\n\n"
                        f"Claim B: {right['claim_text']}"
                    ),
                    source_refs=[
                        SourceRef(left["source_path"], int(left["chunk_id"]), int(left["id"]), left["claim_text"]),
                        SourceRef(right["source_path"], int(right["chunk_id"]), int(right["id"]), right["claim_text"]),
                    ],
                    suggested_action="Review both claims and resolve, dismiss, or update the wiki.",
                    confidence=round(max(similarity, verdict_confidence), 4),
                    metadata={
                        "similarity": round(similarity, 4),
                        "verdict": verdict_label,
                        "verdict_confidence": round(verdict_confidence, 4),
                        "verdict_backend": verdict_backend,
                    },
                )
            )
        return findings[:20]


def candidate_claim_pairs() -> list[tuple[dict, dict, float]]:
    """Return semantically similar claim pairs worth judging."""
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT cl.id, cl.chunk_id, cl.file_id, cl.claim_text, cl.embedding_json,
                   c.source_path
            FROM claims cl
            JOIN chunks c ON c.id = cl.chunk_id
            WHERE cl.status='active' AND cl.embedding_json != ''
            ORDER BY cl.id DESC
            LIMIT 400
            """
        ).fetchall()
    pairs: list[tuple[dict, dict, float]] = []
    for left_index, left in enumerate(rows):
        left_vector = json.loads(left["embedding_json"])
        for right in rows[left_index + 1 :]:
            if int(left["file_id"]) == int(right["file_id"]):
                continue
            right_vector = json.loads(right["embedding_json"])
            similarity = cosine_similarity(left_vector, right_vector)
            if similarity < SIMILARITY_THRESHOLD:
                continue
            if not has_possible_tension(left["claim_text"], right["claim_text"]):
                continue
            pairs.append((dict(left), dict(right), similarity))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:MAX_CANDIDATES]


def has_possible_tension(left: str, right: str) -> bool:
    """Cheap pre-filter for claim pairs with opposing wording."""
    left_lower = left.lower()
    right_lower = right.lower()
    left_negated = has_negation(left_lower)
    right_negated = has_negation(right_lower)
    if left_negated != right_negated:
        return True
    return bool(re.search(r"\b(increase|higher|more|improves|supports)\b", left_lower)) and bool(
        re.search(r"\b(decrease|lower|less|reduces|harms|contradicts)\b", right_lower)
    )


def judge_contradiction(left: str, right: str, use_llm: bool = True) -> tuple[str, float, str]:
    """Ask the configured model to classify a candidate pair when available."""
    nli_result = classify_with_optional_nli(left, right)
    if nli_result:
        return nli_result
    if not use_llm:
        return classify_claim_pair(left, right), 0.65, "heuristic-nli"
    prompt = (
        "Classify these two claims as CONTRADICTION, RELATED, or UNRELATED. "
        "Return exactly one word.\n\n"
        f"Claim A: {left}\n"
        f"Claim B: {right}"
    )
    response = (call_configured_model(prompt) or "").strip().lower()
    if "contradiction" in response:
        return "contradiction", 0.75, "llm-judge"
    if "related" in response:
        return "related", 0.6, "llm-judge"
    if response:
        return "unrelated", 0.6, "llm-judge"
    return classify_claim_pair(left, right), 0.65, "heuristic-nli"


def classify_claim_pair(left: str, right: str) -> str:
    """Local NLI-style fallback for contradiction candidate classification."""
    left_terms = content_terms(left)
    right_terms = content_terms(right)
    if not left_terms or not right_terms:
        return "unrelated"
    overlap = left_terms & right_terms
    overlap_ratio = len(overlap) / max(1, min(len(left_terms), len(right_terms)))
    if overlap_ratio < 0.25:
        return "unrelated"
    left_negated = has_negation(left.lower())
    right_negated = has_negation(right.lower())
    if left_negated != right_negated and overlap_ratio >= 0.4:
        return "contradiction"
    if has_opposing_terms(left_terms, right_terms) and overlap_ratio >= 0.25:
        return "contradiction"
    return "related"


def content_terms(text: str) -> set[str]:
    """Return normalized content terms for lightweight NLI heuristics."""
    terms = set()
    for raw in re.findall(r"[A-Za-z0-9_']+", text.lower()):
        if len(raw) <= 2 or raw in STOP_TERMS:
            continue
        if raw.endswith("ing") and len(raw) > 5:
            raw = raw[:-3]
        elif raw.endswith("s") and len(raw) > 4:
            raw = raw[:-1]
        terms.add(raw)
    return terms


def has_opposing_terms(left_terms: set[str], right_terms: set[str]) -> bool:
    """Return True when two term sets contain an explicit opposition pair."""
    for term, opposites in OPPOSING_TERMS.items():
        if term in left_terms and opposites & right_terms:
            return True
        if term in right_terms and opposites & left_terms:
            return True
    return False


def has_negation(text: str) -> bool:
    """Detect negation terms without matching substrings such as `no` in `noon`."""
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text) for term in NEGATION_TERMS)
