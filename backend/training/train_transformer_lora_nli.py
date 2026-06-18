"""Train Cognix's transformer LoRA NLI model for claim-pair judgment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.database import init_db
from app.db.model_artifacts import record_model_artifact
from app.services.transformer_lora_nli import MODEL_TYPE, TransformerLoRANLIExample, train_transformer_lora_nli_model
from training.train_pair_model import builtin_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local transformer LoRA NLI artifact for Cognix.")
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset with left/right/label fields.")
    parser.add_argument("--output", type=Path, help="Output JSON artifact path.")
    parser.add_argument("--epochs", type=int, default=280)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args()

    examples = load_examples(args.dataset) if args.dataset else builtin_nli_examples()
    output = args.output or get_settings().resolved_transformer_lora_nli_model_path()
    model, metrics = train_transformer_lora_nli_model(
        examples,
        output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        rank=args.rank,
    )
    if args.register_artifact:
        init_db()
        record_model_artifact(
            name="cognix-nli-transformer-lora",
            base_model="cognix-frozen-tiny-transformer-encoder",
            artifact_type="nli_transformer_lora",
            path=str(output),
            status="trained",
            metrics=metrics,
            training_manifest={
                "task": "nli",
                "method": MODEL_TYPE,
                "labels": model.labels,
                "examples": len(examples),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "rank": args.rank,
                "dataset": str(args.dataset) if args.dataset else "builtin_seed_examples",
                "trainable_parameters": metrics["trainable_parameters"],
            },
        )
    print(json.dumps({"output": str(output), "metrics": metrics, "examples": len(examples)}, indent=2, sort_keys=True))
    return 0


def builtin_nli_examples() -> list[TransformerLoRANLIExample]:
    """Return built-in NLI seed examples converted for transformer LoRA training."""
    return [
        TransformerLoRANLIExample(example.left, example.right, example.label)
        for example in builtin_examples("nli")
    ]


def load_examples(path: Path) -> list[TransformerLoRANLIExample]:
    """Load claim-pair NLI examples from JSONL."""
    examples: list[TransformerLoRANLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(TransformerLoRANLIExample(str(record["left"]), str(record["right"]), str(record["label"])))
            except KeyError as exc:
                raise ValueError(f"Line {line_number} must contain left, right, and label.") from exc
    if not examples:
        raise ValueError("Transformer LoRA NLI dataset contains no examples.")
    return examples


if __name__ == "__main__":
    raise SystemExit(main())
