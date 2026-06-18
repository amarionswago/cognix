"""Train local Cognix pair models for reranking or NLI.

The model is a small hashed-feature MLP, so it runs without downloading external
model weights. It is intended as a local trained baseline and artifact path,
not as a replacement for transformer cross-encoders when those are installed.
"""

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
from app.services.pair_model import PairExample, train_pair_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local Cognix pair-text model.")
    parser.add_argument("--task", choices=["reranker", "nli"], required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset with left/right/label fields.")
    parser.add_argument("--output", type=Path, help="Output JSON model artifact path.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args()

    examples = load_examples(args.dataset) if args.dataset else builtin_examples(args.task)
    labels = ["irrelevant", "relevant"] if args.task == "reranker" else ["contradiction", "related", "unrelated"]
    output = args.output or default_output_path(args.task)
    model, metrics = train_pair_model(
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
            name=f"cognix-{args.task}-pair-mlp",
            base_model="cognix_pair_mlp_v1",
            artifact_type=f"{args.task}_pair_model",
            path=str(output),
            status="trained",
            metrics=metrics,
            training_manifest={
                "task": args.task,
                "method": "hashed_pair_mlp",
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
        return settings.resolved_pair_reranker_model_path()
    return settings.resolved_pair_nli_model_path()


def load_examples(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(PairExample(str(record["left"]), str(record["right"]), str(record["label"])))
            except KeyError as exc:
                raise ValueError(f"Line {line_number} must contain left, right, and label.") from exc
    if not examples:
        raise ValueError("Pair-model dataset contains no examples.")
    return examples


def builtin_examples(task: str) -> list[PairExample]:
    """Return deterministic seed examples for local smoke training."""
    if task == "reranker":
        return [
            PairExample("semantic search meaning retrieval", "Semantic search retrieves documents by meaning.", "relevant"),
            PairExample("semantic search meaning retrieval", "Bank statements list card transactions.", "irrelevant"),
            PairExample("large pdf benchmark appendix", "Large PDF benchmark adversarial retrieval appendix.", "relevant"),
            PairExample("large pdf benchmark appendix", "Coffee is not safe before sleep studies.", "irrelevant"),
            PairExample("ocr scanned receipt total", "OCR extracted receipt total and invoice due date.", "relevant"),
            PairExample("ocr scanned receipt total", "Neural rerankers score query document pairs.", "irrelevant"),
            PairExample("calibration probability confidence", "Confidence calibration maps raw scores to observed correctness.", "relevant"),
            PairExample("calibration probability confidence", "A markdown parser extracts headings.", "irrelevant"),
            PairExample("reranker cross encoder", "Cross encoders score a query and document together.", "relevant"),
            PairExample("reranker cross encoder", "CSV rows contain transaction metrics.", "irrelevant"),
            PairExample("knowledge gap detection", "A knowledge gap appears when concepts lack wiki pages.", "relevant"),
            PairExample("knowledge gap detection", "Images require OCR before searchable text exists.", "irrelevant"),
        ]
    return [
        PairExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
        PairExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
        PairExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
        PairExample("The budget is 500 dollars.", "The budget is not 500 dollars.", "contradiction"),
        PairExample("OCR extracts text from images.", "OCR converts visible text into searchable text.", "related"),
        PairExample("OCR extracts text from images.", "LoRA trains adapter weights.", "unrelated"),
        PairExample("The file was processed successfully.", "The file failed to process.", "contradiction"),
        PairExample("RAG uses retrieval before generation.", "Retrieval augmented generation uses evidence before answering.", "related"),
        PairExample("RAG uses retrieval before generation.", "Coffee affects sleep latency.", "unrelated"),
        PairExample("The claim is stale after newer evidence.", "The claim is current after newer evidence.", "contradiction"),
        PairExample("A cross encoder scores text pairs.", "A pair reranker evaluates query document relevance.", "related"),
        PairExample("A cross encoder scores text pairs.", "A receipt contains a purchase total.", "unrelated"),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
