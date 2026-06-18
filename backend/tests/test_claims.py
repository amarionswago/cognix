from app.services.claims import deterministic_claims, parse_claim_json


def test_parse_claim_json_validates_model_output() -> None:
    claims = parse_claim_json(
        '[{"claim":"Python was created by Guido van Rossum.", "confidence":0.9, "type":"factual"}]'
    )

    assert len(claims) == 1
    assert claims[0].claim_type == "factual"
    assert claims[0].confidence == 0.9


def test_deterministic_claims_extracts_definitions() -> None:
    claims = deterministic_claims(
        "Semantic Search is a retrieval technique that finds documents by meaning rather than exact words. "
        "It is useful in knowledge systems."
    )

    assert claims
    assert claims[0].claim_type == "definition"
    assert "Semantic Search is" in claims[0].claim

