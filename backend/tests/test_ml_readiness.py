from pathlib import Path

from app.config import get_settings
from app.database import init_db
from app.db.calibration import record_outcome, record_prediction
from app.services.ml_readiness import advanced_ml_readiness
from app.services.pair_model import PairExample, train_pair_model


def test_ml_readiness_reports_calibration_and_large_eval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    monkeypatch.setenv("COGNIX_RERANKER_BACKEND", "deterministic")
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "heuristic")
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
        record_outcome(prediction_id, "high", reviewer="test")

    readiness = advanced_ml_readiness()
    capabilities = {item["name"]: item for item in readiness["capabilities"]}

    assert capabilities["calibrated_probability_model"]["state"] == "ready"
    assert capabilities["large_file_benchmark_suite"]["state"] == "ready"
    assert capabilities["neural_cross_encoder_reranker"]["state"] == "fallback"
    assert capabilities["trained_nli_contradiction_model"]["state"] == "fallback"

    get_settings.cache_clear()


def test_ml_readiness_reports_local_pair_artifacts(tmp_path: Path, monkeypatch) -> None:
    reranker_path = tmp_path / "reranker.json"
    nli_path = tmp_path / "nli.json"
    train_pair_model(
        [
            PairExample("semantic search", "Semantic search retrieves by meaning.", "relevant"),
            PairExample("semantic search", "Bank exports list transactions.", "irrelevant"),
        ],
        ["irrelevant", "relevant"],
        reranker_path,
        "reranker",
        epochs=30,
    )
    train_pair_model(
        [
            PairExample("Coffee is safe.", "Coffee is not safe.", "contradiction"),
            PairExample("OCR extracts text.", "OCR extracts visible words.", "related"),
            PairExample("OCR extracts text.", "Bank exports list transactions.", "unrelated"),
        ],
        ["contradiction", "related", "unrelated"],
        nli_path,
        "nli",
        epochs=30,
    )
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    monkeypatch.setenv("COGNIX_RERANKER_BACKEND", "cognix-pair")
    monkeypatch.setenv("COGNIX_PAIR_RERANKER_MODEL_PATH", str(reranker_path))
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "cognix-pair")
    monkeypatch.setenv("COGNIX_PAIR_NLI_MODEL_PATH", str(nli_path))
    get_settings.cache_clear()
    init_db()

    readiness = advanced_ml_readiness()
    capabilities = {item["name"]: item for item in readiness["capabilities"]}

    assert capabilities["neural_cross_encoder_reranker"]["state"] == "ready"
    assert capabilities["trained_nli_contradiction_model"]["state"] == "ready"

    get_settings.cache_clear()
