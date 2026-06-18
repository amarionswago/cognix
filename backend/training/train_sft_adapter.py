"""Train Cognix's dependency-free SFT adapter."""

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
from app.services.sft_adapter import MODEL_TYPE, train_sft_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Cognix's local SFT adapter.")
    parser.add_argument("--dataset", type=Path, required=True, help="Chat-style SFT JSONL dataset.")
    parser.add_argument("--output", type=Path, help="Output adapter JSON path.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args()

    output = args.output or get_settings().resolved_cognix_sft_adapter_path()
    _adapter, metrics = train_sft_adapter(args.dataset, output, epochs=args.epochs, learning_rate=args.learning_rate)
    if args.register_artifact:
        init_db()
        record_model_artifact(
            name="cognix-sft-adapter",
            base_model=MODEL_TYPE,
            artifact_type="cognix_sft_adapter",
            path=str(output),
            status="trained",
            metrics=metrics,
            training_manifest={
                "dataset": str(args.dataset),
                "output": str(output),
                "method": MODEL_TYPE,
                "examples": metrics["examples"],
                "dataset_sha256": metrics["dataset_sha256"],
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
        )
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
