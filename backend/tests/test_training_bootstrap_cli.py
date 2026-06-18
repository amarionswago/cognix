import json
import os
import subprocess
import sys
from pathlib import Path

from app.config import get_settings
from app.database import init_db
from app.db.calibration import load_calibration_model, record_outcome, record_prediction
from app.services.ml_readiness import calibration_status


def test_bootstrap_local_models_cli_writes_all_local_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["COGNIX_DATA_DIR"] = str(tmp_path / "data")
    env["COGNIX_WIKI_DIR"] = str(tmp_path / "wiki")
    env["COGNIX_DATABASE_PATH"] = str(tmp_path / "library.sqlite")
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/bootstrap_local_models.py",
            "--epochs",
            "30",
            "--register-artifacts",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    for key in ("reranker", "cross_encoder_reranker", "nli", "cross_encoder_nli", "micro_synthesis"):
        path = Path(payload[key]["path"])
        assert path.exists(), key
        assert payload[key]["metrics"]["examples"] >= 2


def test_train_calibrator_cli_persists_logistic_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    get_settings.cache_clear()
    init_db()

    for index in range(24):
        prediction_id = record_prediction(
            "answer_confidence",
            {"question": f"supported {index}"},
            "supported",
            0.82,
            "test",
        )
        record_outcome(prediction_id, "supported", reviewer="eval")
    for index in range(24):
        prediction_id = record_prediction(
            "answer_confidence",
            {"question": f"unsupported {index}"},
            "supported",
            0.18,
            "test",
        )
        record_outcome(prediction_id, "unsupported", reviewer="eval")

    env = os.environ.copy()
    env["COGNIX_DATA_DIR"] = str(tmp_path / "data")
    env["COGNIX_WIKI_DIR"] = str(tmp_path / "wiki")
    env["COGNIX_DATABASE_PATH"] = str(tmp_path / "library.sqlite")
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_calibrator.py",
            "--task",
            "answer_confidence",
            "--min-examples",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    model = load_calibration_model("answer_confidence")
    status = calibration_status()
    assert model is not None
    assert model.method == "logistic_platt_score_v1"
    assert status["state"] == "ready"
    assert status["detail"]["method"] == "logistic_platt_score_v1"
    get_settings.cache_clear()
