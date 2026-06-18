"""Bootstrap local trained Cognix model artifacts.

This command creates the repo-native trained artifacts that do not require
external model downloads: the pair reranker, pair NLI classifier, and
micro-synthesis policy. These are small baseline models intended to make local
ML paths immediately testable after setup.
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
from app.services.cognix_micro_model import MODEL_TYPE, train_cognix_micro_model
from app.services.cross_encoder_model import MODEL_TYPE as CROSS_ENCODER_MODEL_TYPE
from app.services.cross_encoder_model import train_cross_encoder_model
from app.services.pair_model import train_pair_model
from app.services.sft_adapter import MODEL_TYPE as SFT_ADAPTER_MODEL_TYPE
from app.services.sft_adapter import train_sft_adapter
from app.services.transformer_lora_nli import MODEL_TYPE as TRANSFORMER_LORA_NLI_MODEL_TYPE
from app.services.transformer_lora_nli import TransformerLoRANLIExample, train_transformer_lora_nli_model
from training.train_cross_encoder_model import pair_to_cross_examples
from training.train_pair_model import builtin_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local Cognix baseline model artifacts.")
    parser.add_argument("--sft-dataset", type=Path, default=Path(__file__).with_name("seed_qa_citation_sft.jsonl"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--register-artifacts", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    init_db()

    outputs: dict[str, object] = {}
    reranker_path = settings.resolved_pair_reranker_model_path()
    reranker_model, reranker_metrics = train_pair_model(
        builtin_examples("reranker"),
        ["irrelevant", "relevant"],
        reranker_path,
        "reranker",
        epochs=args.epochs,
        learning_rate=0.06,
    )
    outputs["reranker"] = {"path": str(reranker_path), "metrics": reranker_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-reranker-pair-mlp",
            base_model="cognix_pair_mlp_v1",
            artifact_type="reranker_pair_model",
            path=str(reranker_path),
            status="trained",
            metrics=reranker_metrics,
            training_manifest={
                "task": "reranker",
                "method": "hashed_pair_mlp",
                "labels": reranker_model.labels,
                "examples": len(builtin_examples("reranker")),
                "epochs": args.epochs,
                "dataset": "builtin_seed_examples",
            },
        )

    cross_reranker_path = settings.resolved_local_cross_encoder_reranker_model_path()
    cross_reranker_model, cross_reranker_metrics = train_cross_encoder_model(
        pair_to_cross_examples(builtin_examples("reranker")),
        ["irrelevant", "relevant"],
        cross_reranker_path,
        "reranker",
        epochs=args.epochs,
        learning_rate=0.07,
    )
    outputs["cross_encoder_reranker"] = {"path": str(cross_reranker_path), "metrics": cross_reranker_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-reranker-tiny-cross-encoder",
            base_model=CROSS_ENCODER_MODEL_TYPE,
            artifact_type="reranker_cross_encoder",
            path=str(cross_reranker_path),
            status="trained",
            metrics=cross_reranker_metrics,
            training_manifest={
                "task": "reranker",
                "method": CROSS_ENCODER_MODEL_TYPE,
                "labels": cross_reranker_model.labels,
                "examples": len(builtin_examples("reranker")),
                "epochs": args.epochs,
                "dataset": "builtin_seed_examples",
            },
        )

    nli_path = settings.resolved_pair_nli_model_path()
    nli_model, nli_metrics = train_pair_model(
        builtin_examples("nli"),
        ["contradiction", "related", "unrelated"],
        nli_path,
        "nli",
        epochs=args.epochs,
        learning_rate=0.06,
    )
    outputs["nli"] = {"path": str(nli_path), "metrics": nli_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-nli-pair-mlp",
            base_model="cognix_pair_mlp_v1",
            artifact_type="nli_pair_model",
            path=str(nli_path),
            status="trained",
            metrics=nli_metrics,
            training_manifest={
                "task": "nli",
                "method": "hashed_pair_mlp",
                "labels": nli_model.labels,
                "examples": len(builtin_examples("nli")),
                "epochs": args.epochs,
                "dataset": "builtin_seed_examples",
            },
        )

    cross_nli_path = settings.resolved_local_cross_encoder_nli_model_path()
    cross_nli_model, cross_nli_metrics = train_cross_encoder_model(
        pair_to_cross_examples(builtin_examples("nli")),
        ["contradiction", "related", "unrelated"],
        cross_nli_path,
        "nli",
        epochs=args.epochs,
        learning_rate=0.07,
    )
    outputs["cross_encoder_nli"] = {"path": str(cross_nli_path), "metrics": cross_nli_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-nli-tiny-cross-encoder",
            base_model=CROSS_ENCODER_MODEL_TYPE,
            artifact_type="nli_cross_encoder",
            path=str(cross_nli_path),
            status="trained",
            metrics=cross_nli_metrics,
            training_manifest={
                "task": "nli",
                "method": CROSS_ENCODER_MODEL_TYPE,
                "labels": cross_nli_model.labels,
                "examples": len(builtin_examples("nli")),
                "epochs": args.epochs,
                "dataset": "builtin_seed_examples",
            },
        )

    transformer_lora_nli_path = settings.resolved_transformer_lora_nli_model_path()
    transformer_lora_examples = [
        TransformerLoRANLIExample(example.left, example.right, example.label)
        for example in builtin_examples("nli")
    ]
    transformer_lora_nli_model, transformer_lora_nli_metrics = train_transformer_lora_nli_model(
        transformer_lora_examples,
        transformer_lora_nli_path,
        epochs=max(args.epochs, 280),
        learning_rate=0.03,
    )
    outputs["transformer_lora_nli"] = {"path": str(transformer_lora_nli_path), "metrics": transformer_lora_nli_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-nli-transformer-lora",
            base_model="cognix-frozen-tiny-transformer-encoder",
            artifact_type="nli_transformer_lora",
            path=str(transformer_lora_nli_path),
            status="trained",
            metrics=transformer_lora_nli_metrics,
            training_manifest={
                "task": "nli",
                "method": TRANSFORMER_LORA_NLI_MODEL_TYPE,
                "labels": transformer_lora_nli_model.labels,
                "examples": len(transformer_lora_examples),
                "epochs": max(args.epochs, 280),
                "learning_rate": 0.03,
                "dataset": "builtin_seed_examples",
                "trainable_parameters": transformer_lora_nli_metrics["trainable_parameters"],
            },
        )

    micro_path = settings.resolved_cognix_micro_model_path()
    _micro_model, micro_metrics = train_cognix_micro_model(args.sft_dataset, micro_path)
    outputs["micro_synthesis"] = {"path": str(micro_path), "metrics": micro_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-micro-synthesis",
            base_model=MODEL_TYPE,
            artifact_type="cognix_micro_synthesis",
            path=str(micro_path),
            status="trained",
            metrics=micro_metrics,
            training_manifest={
                "task": "answer_synthesis",
                "method": MODEL_TYPE,
                "dataset": str(args.sft_dataset),
                "examples": micro_metrics["examples"],
                "dataset_sha256": micro_metrics["dataset_sha256"],
            },
        )

    sft_adapter_path = settings.resolved_cognix_sft_adapter_path()
    _sft_adapter, sft_adapter_metrics = train_sft_adapter(args.sft_dataset, sft_adapter_path)
    outputs["sft_adapter"] = {"path": str(sft_adapter_path), "metrics": sft_adapter_metrics}
    if args.register_artifacts:
        record_model_artifact(
            name="cognix-sft-adapter",
            base_model=SFT_ADAPTER_MODEL_TYPE,
            artifact_type="cognix_sft_adapter",
            path=str(sft_adapter_path),
            status="trained",
            metrics=sft_adapter_metrics,
            training_manifest={
                "task": "answer_synthesis_adapter",
                "method": SFT_ADAPTER_MODEL_TYPE,
                "dataset": str(args.sft_dataset),
                "examples": sft_adapter_metrics["examples"],
                "dataset_sha256": sft_adapter_metrics["dataset_sha256"],
            },
        )

    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
