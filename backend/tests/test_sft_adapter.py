import json
import subprocess
import sys
from pathlib import Path

from app.config import get_settings
from app.services.llm import load_optional_sft_adapter, synthesize_answer
from app.services.ml_readiness import sft_adapter_status
from app.services.retrieval_types import RetrievedChunk
from app.services.sft_adapter import load_sft_adapter, train_sft_adapter


def write_sft_dataset(path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: What is semantic search?"},
                {
                    "role": "assistant",
                    "content": "# Research Memo\n\n**Question:** What is semantic search?\n\n## Answer\n- Semantic search retrieves by meaning.\n\n## Evidence\n1. semantic.md",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: Why does OCR matter?"},
                {
                    "role": "assistant",
                    "content": "# Research Memo\n\n**Question:** Why does OCR matter?\n\n## Answer\n- OCR turns visible text into searchable evidence.\n\n## Evidence\n1. ocr.md",
                },
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_sft_adapter_trains_saves_and_selects_prototype(tmp_path: Path) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "adapter.json"
    write_sft_dataset(dataset)

    _adapter, metrics = train_sft_adapter(dataset, output, epochs=120)
    loaded = load_sft_adapter(output)
    prototype, confidence = loaded.select_prototype("What is semantic search?", [])

    assert output.exists()
    assert metrics["training_accuracy"] == 1.0
    assert "semantic search" in prototype.question.lower()
    assert confidence > 0.5


def test_sft_adapter_runtime_synthesizes_when_selected(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "adapter.json"
    write_sft_dataset(dataset)
    train_sft_adapter(dataset, output, epochs=120)
    monkeypatch.setenv("COGNIX_SYNTHESIS_BACKEND", "cognix-sft-adapter")
    monkeypatch.setenv("COGNIX_COGNIX_SFT_ADAPTER_PATH", str(output))
    get_settings.cache_clear()
    load_optional_sft_adapter.cache_clear()

    answer = synthesize_answer(
        "What is OCR?",
        [RetrievedChunk(2, "ocr.md", "OCR turns visible text into searchable evidence for scanned documents.", 0.8, "research")],
    )

    assert "## Answer" in answer
    assert "Adapter Notes" in answer
    assert "ocr.md" in answer
    get_settings.cache_clear()
    load_optional_sft_adapter.cache_clear()


def test_sft_adapter_cli_writes_artifact(tmp_path: Path) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "adapter.json"
    write_sft_dataset(dataset)

    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_sft_adapter.py",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--epochs",
            "30",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert '"training_accuracy"' in result.stdout


def test_sft_adapter_readiness_reports_ready_when_selected(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "adapter.json"
    output.write_text(json.dumps({"model_type": "cognix_sft_adapter_v1", "prototypes": [], "weights": [], "bias": []}), encoding="utf-8")
    monkeypatch.setenv("COGNIX_SYNTHESIS_BACKEND", "cognix-sft-adapter")
    monkeypatch.setenv("COGNIX_COGNIX_SFT_ADAPTER_PATH", str(output))
    get_settings.cache_clear()

    status = sft_adapter_status(get_settings())

    assert status["state"] == "ready"
    assert status["name"] == "custom_cognix_sft_adapter"
    get_settings.cache_clear()
