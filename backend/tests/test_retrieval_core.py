from app.services.retrieval import RetrievedChunk, parse_subqueries, reciprocal_rank_fusion


def chunk(chunk_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path=f"doc{chunk_id}.md",
        excerpt="evidence",
        score=score,
        sensitivity="research",
    )


def test_reciprocal_rank_fusion_rewards_repeated_high_rank_results() -> None:
    results = reciprocal_rank_fusion(
        [
            [chunk(1, 0.9), chunk(2, 0.8)],
            [chunk(2, 0.95), chunk(3, 0.7)],
            [chunk(2, 0.7), chunk(1, 0.6)],
        ]
    )

    assert [item.chunk_id for item in results[:2]] == [2, 1]


def test_parse_subqueries_accepts_json_arrays_only() -> None:
    assert parse_subqueries('["sleep effects", "productivity factors"]') == [
        "sleep effects",
        "productivity factors",
    ]
    assert parse_subqueries("not json") == []
