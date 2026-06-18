from pathlib import Path
import subprocess
import sys

from app.config import get_settings
from app.services.nli import classify_with_optional_nli, load_pair_nli
from app.services.pair_model import PairExample, load_pair_model, train_pair_model
from app.services.reranker import load_pair_reranker, rerank_with_optional_model
from app.services.retrieval_types import RetrievedChunk


def test_pair_model_trains_saves_and_predicts_relevance(tmp_path: Path) -> None:
    output = tmp_path / "reranker.json"
    examples = [
        PairExample("semantic search", "Semantic search retrieves by meaning.", "relevant"),
        PairExample("semantic search", "Bank transactions and card statements.", "irrelevant"),
        PairExample("ocr receipt", "OCR extracts receipt totals from images.", "relevant"),
        PairExample("ocr receipt", "Knowledge graphs connect concepts.", "irrelevant"),
    ]

    _model, metrics = train_pair_model(examples, ["irrelevant", "relevant"], output, "reranker", epochs=100)
    loaded = load_pair_model(output)
    relevant = loaded.predict_proba("semantic search", "Semantic search retrieves documents by meaning.")["relevant"]
    irrelevant = loaded.predict_proba("semantic search", "Debit card bank transaction export.")["relevant"]

    assert output.exists()
    assert metrics["accuracy"] >= 0.75
    assert relevant > irrelevant


def test_pair_reranker_runtime_uses_trained_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "reranker.json"
    train_pair_model(
        [
            PairExample("semantic search", "Semantic search retrieves by meaning.", "relevant"),
            PairExample("semantic search", "Bank transactions and card statements.", "irrelevant"),
            PairExample("receipt ocr", "OCR extracts receipt totals from images.", "relevant"),
            PairExample("receipt ocr", "Knowledge graphs connect concepts.", "irrelevant"),
        ],
        ["irrelevant", "relevant"],
        output,
        "reranker",
        epochs=120,
    )
    monkeypatch.setenv("COGNIX_RERANKER_BACKEND", "cognix-pair")
    monkeypatch.setenv("COGNIX_PAIR_RERANKER_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_pair_reranker.cache_clear()
    chunks = [
        RetrievedChunk(1, "bank.md", "Bank transactions and card statements.", 0.9, "research"),
        RetrievedChunk(2, "semantic.md", "Semantic search retrieves by meaning.", 0.1, "research"),
    ]

    reranked = rerank_with_optional_model("semantic search", chunks)

    assert reranked[0].source_path == "semantic.md"
    get_settings.cache_clear()
    load_pair_reranker.cache_clear()


def test_pair_nli_runtime_uses_trained_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nli.json"
    train_pair_model(
        [
            PairExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
            PairExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
            PairExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
            PairExample("The file processed successfully.", "The file failed to process.", "contradiction"),
            PairExample("OCR extracts text.", "OCR converts visible words into searchable text.", "related"),
            PairExample("OCR extracts text.", "LoRA trains adapter weights.", "unrelated"),
        ],
        ["contradiction", "related", "unrelated"],
        output,
        "nli",
        epochs=140,
    )
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "cognix-pair")
    monkeypatch.setenv("COGNIX_PAIR_NLI_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_pair_nli.cache_clear()

    verdict = classify_with_optional_nli("Coffee is safe before noon.", "Coffee is not safe before noon.")

    assert verdict is not None
    assert verdict[0] == "contradiction"
    assert verdict[1] > 0.4
    get_settings.cache_clear()
    load_pair_nli.cache_clear()


def test_pair_model_training_cli_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "cli-reranker.json"
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_pair_model.py",
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
