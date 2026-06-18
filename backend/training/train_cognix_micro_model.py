"""Train a local Cognix micro-synthesis model from SFT JSONL."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Cognix's local answer synthesis policy.")
    parser.add_argument("--dataset", type=Path, required=True, help="Cognix SFT JSONL dataset.")
    parser.add_argument("--output", type=Path, help="Output model artifact path.")
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args()

    output = args.output or get_settings().resolved_cognix_micro_model_path()
    _model, metrics = train_cognix_micro_model(args.dataset, output)
    if args.register_artifact:
        init_db()
        record_model_artifact(
            name="cognix-micro-synthesis",
            base_model=MODEL_TYPE,
            artifact_type="cognix_micro_synthesis",
            path=str(output),
            status="trained",
            metrics=metrics,
            training_manifest={
                "dataset": str(args.dataset),
                "output": str(output),
                "method": MODEL_TYPE,
                "examples": metrics["examples"],
                "dataset_sha256": metrics["dataset_sha256"],
            },
        )
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
