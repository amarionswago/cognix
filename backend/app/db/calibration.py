"""Prediction logging and calibration utilities."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.database import db_session, utc_now

CALIBRATION_MODEL_MIN_EXAMPLES = 20
CALIBRATION_MODEL_MIN_CLASS_EXAMPLES = 2
CALIBRATION_MODEL_EPOCHS = 600
CALIBRATION_MODEL_LEARNING_RATE = 0.25
CALIBRATION_MODEL_L2 = 0.01
EPSILON = 1e-8


@dataclass(frozen=True)
class CalibrationSummary:
    task: str
    examples: int
    accuracy: float
    mean_confidence: float
    calibration_error: float


@dataclass(frozen=True)
class CalibratedScore:
    raw_score: float
    calibrated_score: float
    examples: int
    applied: bool
    method: str


@dataclass(frozen=True)
class CalibrationModel:
    """A persisted one-feature logistic calibrator."""

    task: str
    method: str
    weight: float
    bias: float
    examples: int
    positive_examples: int
    negative_examples: int
    brier_score: float
    log_loss: float


def record_prediction(
    task: str,
    input_payload: dict[str, Any],
    predicted_label: str,
    predicted_score: float,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Store a model or heuristic prediction for later calibration."""
    input_json = json.dumps(input_payload, sort_keys=True)
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_predictions
            (task, input_hash, predicted_label, predicted_score, model_name, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task,
                input_hash,
                predicted_label,
                max(0.0, min(1.0, predicted_score)),
                model_name,
                json.dumps(metadata or {}, sort_keys=True),
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def record_outcome(prediction_id: int, actual_label: str, reviewer: str = "system", notes: str = "") -> int:
    """Attach a ground-truth/reviewed label to a stored prediction."""
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_outcomes
            (prediction_id, actual_label, reviewer, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prediction_id, actual_label, reviewer, notes, utc_now()),
        )
        return int(cursor.lastrowid)


def calibration_summary(task: str) -> CalibrationSummary:
    """Compute a simple observed calibration summary for a task."""
    try:
        with db_session() as conn:
            rows = conn.execute(
                """
                SELECT mp.predicted_label, mp.predicted_score, po.actual_label
                FROM model_predictions mp
                JOIN prediction_outcomes po ON po.prediction_id = mp.id
                WHERE mp.task=?
                """,
                (task,),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        return CalibrationSummary(task=task, examples=0, accuracy=0.0, mean_confidence=0.0, calibration_error=0.0)
    correct = [1.0 if row["predicted_label"] == row["actual_label"] else 0.0 for row in rows]
    confidences = [float(row["predicted_score"]) for row in rows]
    accuracy = sum(correct) / len(correct)
    mean_confidence = sum(confidences) / len(confidences)
    calibration_error = abs(mean_confidence - accuracy)
    return CalibrationSummary(
        task=task,
        examples=len(rows),
        accuracy=round(accuracy, 4),
        mean_confidence=round(mean_confidence, 4),
        calibration_error=round(calibration_error, 4),
    )


def calibrated_probability(task: str, raw_score: float, min_examples: int = 5) -> CalibratedScore:
    """Map a raw score to observed correctness when enough outcomes exist."""
    score = max(0.0, min(1.0, raw_score))
    model = load_calibration_model(task)
    if model:
        calibrated = _sigmoid(model.weight * score + model.bias)
        return CalibratedScore(score, round(calibrated, 4), model.examples, True, model.method)

    try:
        with db_session() as conn:
            rows = conn.execute(
                """
                SELECT mp.predicted_label, mp.predicted_score, po.actual_label
                FROM model_predictions mp
                JOIN prediction_outcomes po ON po.prediction_id = mp.id
                WHERE mp.task=?
                """,
                (task,),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if len(rows) < min_examples:
        return CalibratedScore(score, score, len(rows), False, "identity_insufficient_outcomes")

    target_bin = score_bin(score)
    bin_rows = [row for row in rows if score_bin(float(row["predicted_score"])) == target_bin]
    if len(bin_rows) < max(2, min_examples // 2):
        bin_rows = rows
        method = "global_empirical_accuracy"
    else:
        method = "binned_empirical_accuracy"
    correct = [1.0 if row["predicted_label"] == row["actual_label"] else 0.0 for row in bin_rows]
    calibrated = sum(correct) / len(correct)
    # Light smoothing keeps small calibration sets from collapsing to exactly 0/1.
    smoothed = (calibrated * len(correct) + score * 2) / (len(correct) + 2)
    return CalibratedScore(score, round(smoothed, 4), len(rows), True, method)


def train_calibration_model(
    task: str,
    min_examples: int = CALIBRATION_MODEL_MIN_EXAMPLES,
    epochs: int = CALIBRATION_MODEL_EPOCHS,
    learning_rate: float = CALIBRATION_MODEL_LEARNING_RATE,
    l2: float = CALIBRATION_MODEL_L2,
) -> CalibrationModel:
    """Fit and persist a logistic calibrator for a task.

    The model estimates P(prediction is correct | raw confidence score). It is
    intentionally one-dimensional so the learned mapping remains inspectable.
    """
    examples = calibration_examples(task)
    if len(examples) < min_examples:
        raise ValueError(f"Need at least {min_examples} reviewed outcomes to train calibration for {task}.")

    positives = sum(label for _score, label in examples)
    negatives = len(examples) - positives
    if positives < CALIBRATION_MODEL_MIN_CLASS_EXAMPLES or negatives < CALIBRATION_MODEL_MIN_CLASS_EXAMPLES:
        raise ValueError("Calibration training needs both correct and incorrect reviewed examples.")

    weight = 0.0
    positive_rate = positives / len(examples)
    bias = _logit(min(0.99, max(0.01, positive_rate)))

    for _epoch in range(epochs):
        grad_weight = 0.0
        grad_bias = 0.0
        for score, label in examples:
            probability = _sigmoid(weight * score + bias)
            error = probability - label
            grad_weight += error * score
            grad_bias += error
        grad_weight = grad_weight / len(examples) + l2 * weight
        grad_bias = grad_bias / len(examples)
        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

    probabilities = [_sigmoid(weight * score + bias) for score, _label in examples]
    labels = [label for _score, label in examples]
    model = CalibrationModel(
        task=task,
        method="logistic_platt_score_v1",
        weight=round(weight, 8),
        bias=round(bias, 8),
        examples=len(examples),
        positive_examples=positives,
        negative_examples=negatives,
        brier_score=round(_brier_score(probabilities, labels), 6),
        log_loss=round(_log_loss(probabilities, labels), 6),
    )
    store_calibration_model(model)
    return model


def calibration_examples(task: str) -> list[tuple[float, int]]:
    """Return raw confidence scores and binary correctness labels."""
    try:
        with db_session() as conn:
            rows = conn.execute(
                """
                SELECT mp.predicted_label, mp.predicted_score, po.actual_label
                FROM model_predictions mp
                JOIN prediction_outcomes po ON po.prediction_id = mp.id
                WHERE mp.task=?
                ORDER BY mp.id
                """,
                (task,),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [
        (max(0.0, min(1.0, float(row["predicted_score"]))), 1 if row["predicted_label"] == row["actual_label"] else 0)
        for row in rows
    ]


def store_calibration_model(model: CalibrationModel) -> int:
    """Persist a trained calibrator and make it the active model for its task."""
    payload = {"weight": model.weight, "bias": model.bias}
    with db_session() as conn:
        conn.execute("UPDATE calibration_models SET status='superseded' WHERE task=? AND status='active'", (model.task,))
        cursor = conn.execute(
            """
            INSERT INTO calibration_models
            (task, method, parameters_json, examples, positive_examples, negative_examples,
             brier_score, log_loss, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                model.task,
                model.method,
                json.dumps(payload, sort_keys=True),
                model.examples,
                model.positive_examples,
                model.negative_examples,
                model.brier_score,
                model.log_loss,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def load_calibration_model(task: str) -> CalibrationModel | None:
    """Load the active persisted calibrator for a task."""
    try:
        with db_session() as conn:
            row = conn.execute(
                """
                SELECT task, method, parameters_json, examples, positive_examples,
                       negative_examples, brier_score, log_loss
                FROM calibration_models
                WHERE task=? AND status='active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (task,),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    parameters = json.loads(row["parameters_json"])
    return CalibrationModel(
        task=str(row["task"]),
        method=str(row["method"]),
        weight=float(parameters["weight"]),
        bias=float(parameters["bias"]),
        examples=int(row["examples"]),
        positive_examples=int(row["positive_examples"]),
        negative_examples=int(row["negative_examples"]),
        brier_score=float(row["brier_score"]),
        log_loss=float(row["log_loss"]),
    )


def score_bin(score: float, width: float = 0.2) -> int:
    """Return a fixed-width calibration bin index."""
    bounded = max(0.0, min(0.9999, score))
    return int(bounded / width)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _logit(probability: float) -> float:
    bounded = min(1.0 - EPSILON, max(EPSILON, probability))
    return math.log(bounded / (1.0 - bounded))


def _brier_score(probabilities: list[float], labels: list[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def _log_loss(probabilities: list[float], labels: list[int]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels):
        bounded = min(1.0 - EPSILON, max(EPSILON, probability))
        total += -(label * math.log(bounded) + (1 - label) * math.log(1 - bounded))
    return total / len(labels)


def store_training_example(
    task: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    source: str,
    quality_label: str = "unreviewed",
) -> int:
    """Store a future fine-tuning or evaluation example."""
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO training_examples
            (task, input_json, output_json, source, quality_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task,
                json.dumps(input_payload, sort_keys=True),
                json.dumps(output_payload, sort_keys=True),
                source,
                quality_label,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)
