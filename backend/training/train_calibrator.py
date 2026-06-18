"""Train a persisted probability calibrator from reviewed outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import init_db
from app.db.calibration import calibration_summary, train_calibration_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Cognix probability calibration from reviewed outcomes.")
    parser.add_argument("--task", default="answer_confidence")
    parser.add_argument("--min-examples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    args = parser.parse_args()

    init_db()
    summary_before = calibration_summary(args.task)
    model = train_calibration_model(
        args.task,
        min_examples=args.min_examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    payload = {
        "task": args.task,
        "summary_before": summary_before.__dict__,
        "model": model.__dict__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
