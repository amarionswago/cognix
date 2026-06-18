import json
import subprocess
import sys
from pathlib import Path

from app.config import get_settings
from app.services.cognix_micro_model import load_cognix_micro_model, train_cognix_micro_model
from app.services.llm import load_optional_cognix_micro_model, synthesize_answer
from app.services.ml_readiness import micro_synthesis_status
from app.services.retrieval_types import RetrievedChunk


def write_sft_dataset(path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: What is semantic search?\n\nEvidence:\n- semantic.md#chunk-1"},
                {
                    "role": "assistant",
                    "content": "# Learned Cognix Memo\n\n**Question:** What is semantic search?\n\n## Learned Answer\n- Semantic search retrieves by meaning.\n\n## Learned Evidence\n1. semantic.md",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: What is OCR?\n\nEvidence:\n- ocr.md#chunk-2"},
                {
                    "role": "assistant",
                    "content": "# Learned Cognix Memo\n\n**Question:** What is OCR?\n\n## Learned Answer\n- OCR converts visible text into searchable text.\n\n## Learned Evidence\n1. ocr.md",
                },
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_cognix_micro_model_trains_saves_and_uses_learned_structure(tmp_path: Path) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "micro.json"
    write_sft_dataset(dataset)

    _model, metrics = train_cognix_micro_model(dataset, output)
    loaded = load_cognix_micro_model(output)
    answer = loaded.synthesize(
        "What is semantic search?",
        [RetrievedChunk(1, "semantic.md", "Semantic search retrieves documents by meaning for retrieval systems.", 0.9, "research")],
    )

    assert output.exists()
    assert metrics["examples"] == 2
    assert "# Learned Cognix Memo" in answer
    assert "## Learned Answer" in answer
    assert "semantic.md" in answer


def test_cognix_micro_model_cli_writes_artifact(tmp_path: Path) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "micro.json"
    write_sft_dataset(dataset)

    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_cognix_micro_model.py",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert '"examples": 2' in result.stdout


def test_synthesize_answer_uses_configured_micro_model(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "sft.jsonl"
    output = tmp_path / "micro.json"
    write_sft_dataset(dataset)
    train_cognix_micro_model(dataset, output)
    monkeypatch.setenv("COGNIX_SYNTHESIS_BACKEND", "cognix-micro")
    monkeypatch.setenv("COGNIX_COGNIX_MICRO_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_optional_cognix_micro_model.cache_clear()

    answer = synthesize_answer(
        "What is OCR?",
        [RetrievedChunk(2, "ocr.md", "OCR converts visible text into searchable text for scanned documents.", 0.8, "research")],
    )

    assert "# Learned Cognix Memo" in answer
    assert "local trained Cognix micro-synthesis policy" in answer
    get_settings.cache_clear()
    load_optional_cognix_micro_model.cache_clear()


def test_micro_synthesis_readiness_reports_ready_when_selected(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "micro.json"
    output.write_text(json.dumps({"model_type": "cognix_micro_synthesis_v1"}), encoding="utf-8")
    monkeypatch.setenv("COGNIX_SYNTHESIS_BACKEND", "cognix-micro")
    monkeypatch.setenv("COGNIX_COGNIX_MICRO_MODEL_PATH", str(output))
    get_settings.cache_clear()

    status = micro_synthesis_status(get_settings())

    assert status["state"] == "ready"
    assert status["name"] == "custom_cognix_micro_synthesis_model"
    get_settings.cache_clear()
