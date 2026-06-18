import subprocess
import sys
from pathlib import Path

from app.config import get_settings
from app.services.cross_encoder_model import CrossEncoderExample, load_cross_encoder_model, train_cross_encoder_model
from app.services.nli import classify_with_optional_nli, load_local_cross_encoder_nli
from app.services.reranker import load_local_cross_encoder_reranker, rerank_with_optional_model
from app.services.retrieval_types import RetrievedChunk


def test_cross_encoder_trains_saves_and_scores_relevance(tmp_path: Path) -> None:
    output = tmp_path / "reranker-cross.json"
    examples = [
        CrossEncoderExample("semantic search", "Semantic search retrieves by meaning.", "relevant"),
        CrossEncoderExample("semantic search", "Bank transactions and card statements.", "irrelevant"),
        CrossEncoderExample("ocr receipt", "OCR extracts receipt totals from images.", "relevant"),
        CrossEncoderExample("ocr receipt", "Knowledge graphs connect concepts.", "irrelevant"),
    ]

    _model, metrics = train_cross_encoder_model(examples, ["irrelevant", "relevant"], output, "reranker", epochs=160)
    loaded = load_cross_encoder_model(output)
    relevant = loaded.predict_proba("semantic search", "Semantic search retrieves documents by meaning.")["relevant"]
    irrelevant = loaded.predict_proba("semantic search", "Debit card bank transaction export.")["relevant"]

    assert output.exists()
    assert metrics["accuracy"] >= 0.75
    assert relevant > irrelevant


def test_cross_encoder_reranker_runtime_uses_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "reranker-cross.json"
    train_cross_encoder_model(
        [
            CrossEncoderExample("semantic search", "Semantic search retrieves by meaning.", "relevant"),
            CrossEncoderExample("semantic search", "Bank transactions and card statements.", "irrelevant"),
            CrossEncoderExample("receipt ocr", "OCR extracts receipt totals from images.", "relevant"),
            CrossEncoderExample("receipt ocr", "Knowledge graphs connect concepts.", "irrelevant"),
        ],
        ["irrelevant", "relevant"],
        output,
        "reranker",
        epochs=180,
    )
    monkeypatch.setenv("COGNIX_RERANKER_BACKEND", "cognix-cross-encoder")
    monkeypatch.setenv("COGNIX_LOCAL_CROSS_ENCODER_RERANKER_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_local_cross_encoder_reranker.cache_clear()
    chunks = [
        RetrievedChunk(1, "bank.md", "Bank transactions and card statements.", 0.9, "research"),
        RetrievedChunk(2, "semantic.md", "Semantic search retrieves by meaning.", 0.1, "research"),
    ]

    reranked = rerank_with_optional_model("semantic search", chunks)

    assert reranked[0].source_path == "semantic.md"
    get_settings.cache_clear()
    load_local_cross_encoder_reranker.cache_clear()


def test_cross_encoder_nli_runtime_uses_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nli-cross.json"
    train_cross_encoder_model(
        [
            CrossEncoderExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
            CrossEncoderExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
            CrossEncoderExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
            CrossEncoderExample("The file processed successfully.", "The file failed to process.", "contradiction"),
            CrossEncoderExample("OCR extracts text.", "OCR converts visible words into searchable text.", "related"),
            CrossEncoderExample("OCR extracts text.", "LoRA trains adapter weights.", "unrelated"),
        ],
        ["contradiction", "related", "unrelated"],
        output,
        "nli",
        epochs=220,
    )
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "cognix-cross-encoder")
    monkeypatch.setenv("COGNIX_LOCAL_CROSS_ENCODER_NLI_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_local_cross_encoder_nli.cache_clear()

    verdict = classify_with_optional_nli("Coffee is safe before noon.", "Coffee is not safe before noon.")

    assert verdict is not None
    assert verdict[0] == "contradiction"
    assert verdict[1] > 0.4
    get_settings.cache_clear()
    load_local_cross_encoder_nli.cache_clear()


def test_cross_encoder_training_cli_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "cli-reranker-cross.json"
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_cross_encoder_model.py",
            "--task",
            "reranker",
            "--output",
            str(output),
            "--epochs",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert '"examples"' in result.stdout
