"""Train local Cognix neural cross-encoder models for reranking or NLI."""

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
from app.services.cross_encoder_model import MODEL_TYPE, CrossEncoderExample, train_cross_encoder_model
from app.services.pair_model import PairExample
from training.train_pair_model import builtin_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local Cognix neural cross-encoder.")
    parser.add_argument("--task", choices=["reranker", "nli"], required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset with left/right/label fields.")
    parser.add_argument("--output", type=Path, help="Output JSON model artifact path.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=28)
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args()

    examples = load_examples(args.dataset) if args.dataset else pair_to_cross_examples(builtin_examples(args.task))
    labels = ["irrelevant", "relevant"] if args.task == "reranker" else ["contradiction", "related", "unrelated"]
    output = args.output or default_output_path(args.task)
    model, metrics = train_cross_encoder_model(
        examples=examples,
        labels=labels,
        output_path=output,
        task=args.task,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
    )
    if args.register_artifact:
        init_db()
        record_model_artifact(
            name=f"cognix-{args.task}-tiny-cross-encoder",
            base_model=MODEL_TYPE,
            artifact_type=f"{args.task}_cross_encoder",
            path=str(output),
            status="trained",
            metrics=metrics,
            training_manifest={
                "task": args.task,
                "method": MODEL_TYPE,
                "labels": model.labels,
                "examples": len(examples),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "hidden_dim": args.hidden_dim,
                "dataset": str(args.dataset) if args.dataset else "builtin_seed_examples",
            },
        )
    print(json.dumps({"output": str(output), "metrics": metrics, "examples": len(examples)}, indent=2, sort_keys=True))
    return 0


def default_output_path(task: str) -> Path:
    settings = get_settings()
    if task == "reranker":
        return settings.resolved_local_cross_encoder_reranker_model_path()
    return settings.resolved_local_cross_encoder_nli_model_path()


def load_examples(path: Path) -> list[CrossEncoderExample]:
    examples: list[CrossEncoderExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(CrossEncoderExample(str(record["left"]), str(record["right"]), str(record["label"])))
            except KeyError as exc:
                raise ValueError(f"Line {line_number} must contain left, right, and label.") from exc
    if not examples:
        raise ValueError("Cross-encoder dataset contains no examples.")
    return examples


def pair_to_cross_examples(examples: list[PairExample]) -> list[CrossEncoderExample]:
    """Convert existing seed pair examples into cross-encoder examples."""
    return [CrossEncoderExample(example.left, example.right, example.label) for example in examples]


if __name__ == "__main__":
    raise SystemExit(main())
