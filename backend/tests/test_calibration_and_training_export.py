from pathlib import Path

from app.config import get_settings
from app.database import init_db
from app.db.calibration import (
    calibrated_probability,
    calibration_summary,
    load_calibration_model,
    record_outcome,
    record_prediction,
    store_training_example,
    train_calibration_model,
)
from app.db.model_artifacts import list_model_artifacts
from app.services.finetuning import (
    build_model_package_manifest,
    build_training_manifest,
    register_planned_lora_artifact,
    validate_sft_jsonl,
    write_ollama_modelfile,
    write_training_manifest,
)
from app.services.training_export import export_sft_jsonl, export_training_jsonl


def test_prediction_calibration_and_training_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    get_settings.cache_clear()
    init_db()

    prediction_id = record_prediction(
        "answer_confidence",
        {"question": "What is semantic search?"},
        "high",
        0.9,
        "test-model",
    )
    record_outcome(prediction_id, "high", reviewer="test")
    store_training_example(
        "qa_citation",
        {
            "question": "What is semantic search?",
            "sources": [{"source_path": "semantic.md", "chunk_id": 1, "excerpt": "Semantic search retrieves by meaning."}],
        },
        {"answer": "Semantic search retrieves by meaning."},
        source="test",
        quality_label="reviewed",
    )

    summary = calibration_summary("answer_confidence")
    export_path = export_training_jsonl("qa_citation", "reviewed")
    sft_path = export_sft_jsonl("qa_citation", "reviewed")
    stats = validate_sft_jsonl(sft_path)
    manifest = build_training_manifest(sft_path, tmp_path / "adapter", "base-model", "cognix-test", 1.0, 0.0002, 1)
    manifest_path = write_training_manifest(manifest, tmp_path / "adapter")
    artifact_id = register_planned_lora_artifact(manifest)
    modelfile_path = write_ollama_modelfile("llama3.2", tmp_path / "adapter", tmp_path / "adapter" / "Modelfile")
    package_manifest = build_model_package_manifest("cognix-test", "llama3.2", tmp_path / "adapter", modelfile_path)

    assert summary.examples == 1
    assert summary.accuracy == 1.0
    assert export_path.exists()
    assert sft_path.exists()
    assert stats.examples == 1
    assert manifest_path.exists()
    assert "ADAPTER" in modelfile_path.read_text(encoding="utf-8")
    assert package_manifest["load_command"].startswith("ollama create cognix-test")
    assert artifact_id >= 1
    assert list_model_artifacts()[0]["name"] == "cognix-test"
    assert "Semantic search retrieves by meaning" in export_path.read_text(encoding="utf-8")

    get_settings.cache_clear()


def test_logistic_calibration_model_is_trained_persisted_and_applied(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    get_settings.cache_clear()
    init_db()

    for index in range(30):
        score = 0.82 + (index % 4) * 0.03
        prediction_id = record_prediction(
            "answer_confidence",
            {"question": f"reliable {index}"},
            "supported",
            score,
            "heuristic-confidence-v1",
        )
        record_outcome(prediction_id, "supported", reviewer="eval")

    for index in range(30):
        score = 0.12 + (index % 4) * 0.04
        prediction_id = record_prediction(
            "answer_confidence",
            {"question": f"weak {index}"},
            "supported",
            score,
            "heuristic-confidence-v1",
        )
        record_outcome(prediction_id, "unsupported", reviewer="eval")

    model = train_calibration_model("answer_confidence", min_examples=20)
    loaded = load_calibration_model("answer_confidence")
    high = calibrated_probability("answer_confidence", 0.9)
    low = calibrated_probability("answer_confidence", 0.15)

    assert loaded is not None
    assert loaded.method == "logistic_platt_score_v1"
    assert model.examples == 60
    assert model.positive_examples == 30
    assert model.negative_examples == 30
    assert model.brier_score < 0.15
    assert high.applied is True
    assert high.method == "logistic_platt_score_v1"
    assert high.calibrated_score > 0.75
    assert low.calibrated_score < 0.25

    get_settings.cache_clear()
