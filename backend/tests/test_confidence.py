from datetime import datetime, timezone

from app.config import get_settings
from app.database import init_db
from app.db.calibration import record_outcome, record_prediction
from app.db.confidence import EvidenceSource, compute_confidence


def test_confidence_uses_source_count_diversity_and_scores() -> None:
    result = compute_confidence(
        [
            EvidenceSource("a.md", 0.9, "2026-06-01T00:00:00+00:00"),
            EvidenceSource("b.md", 0.8, "2026-06-01T00:00:00+00:00"),
            EvidenceSource("c.md", 0.7, "2026-06-01T00:00:00+00:00"),
            EvidenceSource("d.md", 0.6, "2026-06-01T00:00:00+00:00"),
        ],
        now=datetime(2026, 6, 17, tzinfo=timezone.utc),
    )

    assert result.score > 0.7
    assert result.label in {"medium", "high"}
    assert result.breakdown["source_count"] == 4
    assert result.breakdown["unique_files"] == 4


def test_confidence_is_low_without_sources() -> None:
    result = compute_confidence([])

    assert result.score == 0.0
    assert result.label == "low"
    assert result.breakdown["source_count"] == 0


def test_confidence_uses_empirical_calibration_when_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    get_settings.cache_clear()
    init_db()
    for index in range(5):
        prediction_id = record_prediction(
            "answer_confidence",
            {"question": f"q{index}"},
            "high",
            0.9,
            "test",
        )
        record_outcome(prediction_id, "low", reviewer="test")

    result = compute_confidence(
        [EvidenceSource("a.md", 0.9, "2026-06-01T00:00:00+00:00")],
        now=datetime(2026, 6, 17, tzinfo=timezone.utc),
    )

    assert result.breakdown["calibration_applied"] is True
    assert result.breakdown["calibration_examples"] == 5
    assert result.score < result.breakdown["raw_score"]
    get_settings.cache_clear()
