"""Small trainable pair-text classifier for local reranking and NLI.

This is a deliberately compact neural baseline: hashed pair features feed a
single tanh hidden layer and a softmax output. It is not a transformer
cross-encoder, but it is a real trained local model artifact over text pairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "cognix_pair_mlp_v1"
DEFAULT_FEATURE_DIM = 96
DEFAULT_HIDDEN_DIM = 24
RESERVED_FEATURES = 8
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]*")
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "isn't", "aren't", "doesn't", "don't"}


@dataclass(frozen=True)
class PairExample:
    """A labeled pair-text training example."""

    left: str
    right: str
    label: str


@dataclass(frozen=True)
class PairPrediction:
    """A pair-model prediction."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class PairTextModel:
    """One-hidden-layer pair classifier with JSON persistence."""

    def __init__(
        self,
        labels: list[str],
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        seed: int = 17,
        weights1: list[list[float]] | None = None,
        bias1: list[float] | None = None,
        weights2: list[list[float]] | None = None,
        bias2: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(labels) < 2:
            raise ValueError("PairTextModel requires at least two labels.")
        self.labels = labels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.metadata = metadata or {}
        rng = random.Random(seed)
        self.weights1 = weights1 or [
            [rng.uniform(-0.08, 0.08) for _ in range(feature_dim)]
            for _hidden in range(hidden_dim)
        ]
        self.bias1 = bias1 or [0.0 for _hidden in range(hidden_dim)]
        self.weights2 = weights2 or [
            [rng.uniform(-0.08, 0.08) for _hidden in range(hidden_dim)]
            for _label in labels
        ]
        self.bias2 = bias2 or [0.0 for _label in labels]

    def predict(self, left: str, right: str) -> PairPrediction:
        """Predict the most likely label for a text pair."""
        probabilities = self.predict_proba(left, right)
        label = max(probabilities, key=probabilities.get)
        return PairPrediction(label=label, confidence=round(probabilities[label], 4), probabilities=probabilities)

    def predict_proba(self, left: str, right: str) -> dict[str, float]:
        """Return softmax probabilities for a text pair."""
        features = pair_features(left, right, self.feature_dim)
        hidden, logits, probabilities = self._forward(features)
        _ = hidden, logits
        return {label: round(probability, 6) for label, probability in zip(self.labels, probabilities)}

    def train(
        self,
        examples: list[PairExample],
        epochs: int = 80,
        learning_rate: float = 0.05,
        l2: float = 0.0005,
    ) -> dict[str, float]:
        """Train the model with stochastic gradient descent."""
        if not examples:
            raise ValueError("Cannot train pair model with no examples.")
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        for example in examples:
            if example.label not in label_to_index:
                raise ValueError(f"Unknown label {example.label!r}; expected one of {self.labels}.")

        rng = random.Random(23)
        for _epoch in range(epochs):
            shuffled = examples[:]
            rng.shuffle(shuffled)
            for example in shuffled:
                features = pair_features(example.left, example.right, self.feature_dim)
                hidden, logits, probabilities = self._forward(features)
                _ = logits
                target_index = label_to_index[example.label]
                grad_logits = probabilities[:]
                grad_logits[target_index] -= 1.0

                old_weights2 = [row[:] for row in self.weights2]
                for label_index in range(len(self.labels)):
                    for hidden_index in range(self.hidden_dim):
                        gradient = grad_logits[label_index] * hidden[hidden_index] + l2 * self.weights2[label_index][hidden_index]
                        self.weights2[label_index][hidden_index] -= learning_rate * gradient
                    self.bias2[label_index] -= learning_rate * grad_logits[label_index]

                grad_hidden = [0.0 for _hidden in range(self.hidden_dim)]
                for hidden_index in range(self.hidden_dim):
                    upstream = sum(grad_logits[label_index] * old_weights2[label_index][hidden_index] for label_index in range(len(self.labels)))
                    grad_hidden[hidden_index] = upstream * (1.0 - hidden[hidden_index] * hidden[hidden_index])

                for hidden_index in range(self.hidden_dim):
                    for feature_index, feature_value in enumerate(features):
                        if feature_value == 0.0:
                            continue
                        gradient = grad_hidden[hidden_index] * feature_value + l2 * self.weights1[hidden_index][feature_index]
                        self.weights1[hidden_index][feature_index] -= learning_rate * gradient
                    self.bias1[hidden_index] -= learning_rate * grad_hidden[hidden_index]
        return evaluate_pair_model(self, examples)

    def save(self, path: Path) -> Path:
        """Persist the trained model to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": MODEL_TYPE,
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "weights1": self.weights1,
            "bias1": self.bias1,
            "weights2": self.weights2,
            "bias2": self.bias2,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _forward(self, features: list[float]) -> tuple[list[float], list[float], list[float]]:
        hidden = []
        for hidden_index in range(self.hidden_dim):
            value = self.bias1[hidden_index] + sum(
                self.weights1[hidden_index][feature_index] * feature
                for feature_index, feature in enumerate(features)
            )
            hidden.append(math.tanh(value))
        logits = []
        for label_index in range(len(self.labels)):
            logits.append(
                self.bias2[label_index]
                + sum(self.weights2[label_index][hidden_index] * hidden_value for hidden_index, hidden_value in enumerate(hidden))
            )
        probabilities = softmax(logits)
        return hidden, logits, probabilities


def load_pair_model(path: Path) -> PairTextModel:
    """Load a trained pair model JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unsupported pair model type: {payload.get('model_type')}")
    return PairTextModel(
        labels=[str(label) for label in payload["labels"]],
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        weights1=payload["weights1"],
        bias1=payload["bias1"],
        weights2=payload["weights2"],
        bias2=payload["bias2"],
        metadata=payload.get("metadata", {}),
    )


def train_pair_model(
    examples: list[PairExample],
    labels: list[str],
    output_path: Path,
    task: str,
    epochs: int = 80,
    learning_rate: float = 0.05,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> tuple[PairTextModel, dict[str, float]]:
    """Train and save a pair model artifact."""
    model = PairTextModel(labels=labels, hidden_dim=hidden_dim, metadata={"task": task})
    metrics = model.train(examples, epochs=epochs, learning_rate=learning_rate)
    model.metadata.update({"task": task, "examples": len(examples), "metrics": metrics})
    model.save(output_path)
    return model, metrics


def evaluate_pair_model(model: PairTextModel, examples: list[PairExample]) -> dict[str, float]:
    """Evaluate accuracy and mean confidence on examples."""
    correct = 0
    confidence = 0.0
    for example in examples:
        prediction = model.predict(example.left, example.right)
        correct += int(prediction.label == example.label)
        confidence += prediction.confidence
    return {
        "examples": float(len(examples)),
        "accuracy": round(correct / max(1, len(examples)), 4),
        "mean_confidence": round(confidence / max(1, len(examples)), 4),
    }


def pair_features(left: str, right: str, feature_dim: int = DEFAULT_FEATURE_DIM) -> list[float]:
    """Build normalized hashed pair features for two texts."""
    if feature_dim <= RESERVED_FEATURES:
        raise ValueError("feature_dim must exceed reserved scalar feature count.")
    vector = [0.0 for _ in range(feature_dim)]
    hash_dim = feature_dim - RESERVED_FEATURES
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    for token in left_tokens:
        vector[stable_hash("left:" + token, hash_dim)] += 1.0
    for token in right_tokens:
        vector[stable_hash("right:" + token, hash_dim)] += 1.0
    for token in left_set & right_set:
        vector[stable_hash("overlap:" + token, hash_dim)] += 1.5
    vector[-8] = len(left_set & right_set) / max(1, len(left_set | right_set))
    vector[-7] = len(left_tokens) / 200
    vector[-6] = len(right_tokens) / 200
    vector[-5] = 1.0 if has_negation(left_tokens) else 0.0
    vector[-4] = 1.0 if has_negation(right_tokens) else 0.0
    vector[-3] = 1.0 if has_negation(left_tokens) != has_negation(right_tokens) else 0.0
    vector[-2] = abs(len(left_tokens) - len(right_tokens)) / 200
    vector[-1] = 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def tokenize(text: str) -> list[str]:
    """Tokenize text for pair-model features."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def has_negation(tokens: list[str]) -> bool:
    """Return whether token list contains a negation marker."""
    return any(token in NEGATIONS for token in tokens)


def stable_hash(value: str, modulo: int) -> int:
    """Stable integer hash independent of Python hash randomization."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]
