"""Fine-tuning readiness and LoRA training helpers for Cognix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db.model_artifacts import record_model_artifact


@dataclass(frozen=True)
class DatasetStats:
    path: str
    examples: int
    bytes: int
    sha256: str


def validate_sft_jsonl(path: Path) -> DatasetStats:
    """Validate chat-style SFT JSONL and return stable dataset stats."""
    if not path.exists():
        raise FileNotFoundError(path)
    examples = 0
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_number} must contain at least two chat messages.")
            if not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages):
                raise ValueError(f"Line {line_number} has no assistant message.")
            for message in messages:
                if not isinstance(message, dict) or not message.get("role") or not message.get("content"):
                    raise ValueError(f"Line {line_number} contains an invalid message.")
            examples += 1
    if examples == 0:
        raise ValueError("Dataset contains no trainable examples.")
    return DatasetStats(str(path), examples, path.stat().st_size, digest.hexdigest())


def build_training_manifest(
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    adapter_name: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
) -> dict[str, Any]:
    """Create a deterministic manifest for a LoRA training run."""
    stats = validate_sft_jsonl(dataset_path)
    return {
        "adapter_name": adapter_name,
        "base_model": base_model,
        "dataset": {
            "path": stats.path,
            "examples": stats.examples,
            "bytes": stats.bytes,
            "sha256": stats.sha256,
        },
        "output_dir": str(output_dir),
        "method": "lora_sft",
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
    }


def write_training_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write a training manifest beside a future adapter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def register_planned_lora_artifact(manifest: dict[str, Any]) -> int:
    """Record a planned LoRA artifact before or during training."""
    return record_model_artifact(
        name=str(manifest["adapter_name"]),
        base_model=str(manifest["base_model"]),
        artifact_type="lora_adapter",
        path=str(manifest["output_dir"]),
        status="planned",
        metrics={},
        training_manifest=manifest,
    )


def write_ollama_modelfile(base_model: str, adapter_path: Path, output_path: Path, system_prompt: str | None = None) -> Path:
    """Write an Ollama Modelfile for a trained Cognix adapter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = system_prompt or (
        "You are Cognix, a source-grounded knowledge assistant. Answer from evidence, "
        "state uncertainty, and preserve citations."
    )
    lines = [
        f"FROM {base_model}",
        f"ADAPTER {adapter_path}",
        f'SYSTEM """{prompt}"""',
        "PARAMETER temperature 0.2",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_model_package_manifest(
    adapter_name: str,
    base_model: str,
    adapter_path: Path,
    modelfile_path: Path,
) -> dict[str, Any]:
    """Describe a local custom Cognix model package."""
    return {
        "adapter_name": adapter_name,
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "modelfile_path": str(modelfile_path),
        "runtime": "ollama",
        "load_command": f"ollama create {adapter_name} -f {modelfile_path}",
    }
