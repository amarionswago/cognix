from app.services.intelligence.contradiction import classify_claim_pair


def test_local_nli_fallback_detects_negated_contradiction() -> None:
    verdict = classify_claim_pair(
        "Coffee is safe for sleep research participants when consumed before noon.",
        "Coffee is not safe for sleep research participants when consumed before noon.",
    )

    assert verdict == "contradiction"


def test_local_nli_fallback_rejects_unrelated_claims() -> None:
    verdict = classify_claim_pair(
        "Semantic search retrieves documents by meaning.",
        "Bank statements list debit card transactions.",
    )

    assert verdict == "unrelated"
