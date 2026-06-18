import subprocess
import sys
from pathlib import Path

from app.config import get_settings
from app.services.ml_readiness import nli_status
from app.services.nli import classify_with_optional_nli, load_transformer_lora_nli
from app.services.transformer_lora_nli import (
    TransformerLoRANLIExample,
    load_transformer_lora_nli_model,
    train_transformer_lora_nli_model,
)


def nli_examples() -> list[TransformerLoRANLIExample]:
    return [
        TransformerLoRANLIExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
        TransformerLoRANLIExample("The file processed successfully.", "The file failed to process.", "contradiction"),
        TransformerLoRANLIExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
        TransformerLoRANLIExample("OCR extracts text.", "OCR converts visible words into searchable text.", "related"),
        TransformerLoRANLIExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
        TransformerLoRANLIExample("OCR extracts text.", "LoRA trains adapter weights.", "unrelated"),
    ]


def test_transformer_lora_nli_trains_saves_and_predicts(tmp_path: Path) -> None:
    output = tmp_path / "nli-transformer-lora.json"

    _model, metrics = train_transformer_lora_nli_model(nli_examples(), output, epochs=320)
    loaded = load_transformer_lora_nli_model(output)
    contradiction = loaded.predict("Coffee is safe before noon.", "Coffee is not safe before noon.")
    related = loaded.predict("OCR extracts text.", "OCR converts visible words into searchable text.")
    unrelated = loaded.predict("OCR extracts text.", "LoRA trains adapter weights.")

    assert output.exists()
    assert metrics["accuracy"] >= 0.8
    assert metrics["trainable_parameters"] > 0
    assert contradiction.label == "contradiction"
    assert related.label == "related"
    assert unrelated.label == "unrelated"


def test_transformer_lora_nli_runtime_backend_uses_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nli-transformer-lora.json"
    train_transformer_lora_nli_model(nli_examples(), output, epochs=320)
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "cognix-transformer-lora")
    monkeypatch.setenv("COGNIX_TRANSFORMER_LORA_NLI_MODEL_PATH", str(output))
    get_settings.cache_clear()
    load_transformer_lora_nli.cache_clear()

    verdict = classify_with_optional_nli("The file processed successfully.", "The file failed to process.")

    assert verdict is not None
    assert verdict[0] == "contradiction"
    assert verdict[1] > 0.5
    assert verdict[2] == str(output)
    get_settings.cache_clear()
    load_transformer_lora_nli.cache_clear()


def test_transformer_lora_nli_cli_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "cli-nli-transformer-lora.json"
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_transformer_lora_nli.py",
            "--output",
            str(output),
            "--epochs",
            "320",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert '"accuracy"' in result.stdout


def test_transformer_lora_nli_readiness_reports_ready_when_selected(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nli-transformer-lora.json"
    train_transformer_lora_nli_model(nli_examples(), output, epochs=320)
    monkeypatch.setenv("COGNIX_NLI_BACKEND", "cognix-transformer-lora")
    monkeypatch.setenv("COGNIX_TRANSFORMER_LORA_NLI_MODEL_PATH", str(output))
    get_settings.cache_clear()

    status = nli_status(get_settings())

    assert status["state"] == "ready"
    assert status["name"] == "trained_nli_contradiction_model"
    assert status["detail"]["artifact_exists"] is True
    get_settings.cache_clear()
