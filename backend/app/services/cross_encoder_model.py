"""Small local neural cross-encoder for pairwise reranking and NLI.

The model reads both texts together, builds learned token/segment embeddings,
forms pair interaction features, and trains a one-hidden-layer softmax
classifier. It is intentionally lightweight and dependency-free. It is not a
transformer, but it is a real local neural cross-encoder artifact path.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "cognix_tiny_cross_encoder_v1"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]*")
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "isn't", "aren't", "doesn't", "don't", "failed"}
DEFAULT_EMBEDDING_DIM = 18
DEFAULT_HIDDEN_DIM = 28
MAX_VOCAB = 512
UNK = "<unk>"


@dataclass(frozen=True)
class CrossEncoderExample:
    """A labeled text-pair example."""

    left: str
    right: str
    label: str


@dataclass(frozen=True)
class CrossEncoderPrediction:
    """A cross-encoder prediction."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class TinyCrossEncoder:
    """Dependency-free neural cross-encoder with JSON persistence."""

    def __init__(
        self,
        labels: list[str],
        vocabulary: dict[str, int],
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        seed: int = 31,
        token_embeddings: list[list[float]] | None = None,
        segment_embeddings: list[list[float]] | None = None,
        weights1: list[list[float]] | None = None,
        bias1: list[float] | None = None,
        weights2: list[list[float]] | None = None,
        bias2: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(labels) < 2:
            raise ValueError("TinyCrossEncoder requires at least two labels.")
        self.labels = labels
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = embedding_dim * 4 + 5
        self.metadata = metadata or {}
        rng = random.Random(seed)
        self.token_embeddings = token_embeddings or [
            [rng.uniform(-0.25, 0.25) for _dim in range(embedding_dim)]
            for _token in range(len(vocabulary))
        ]
        self.segment_embeddings = segment_embeddings or [
            [rng.uniform(-0.08, 0.08) for _dim in range(embedding_dim)]
            for _segment in range(2)
        ]
        self.weights1 = weights1 or [
            [rng.uniform(-0.08, 0.08) for _feature in range(self.feature_dim)]
            for _hidden in range(hidden_dim)
        ]
        self.bias1 = bias1 or [0.0 for _hidden in range(hidden_dim)]
        self.weights2 = weights2 or [
            [rng.uniform(-0.08, 0.08) for _hidden in range(hidden_dim)]
            for _label in labels
        ]
        self.bias2 = bias2 or [0.0 for _label in labels]

    def predict(self, left: str, right: str) -> CrossEncoderPrediction:
        """Predict the best label for a text pair."""
        probabilities = self.predict_proba(left, right)
        label = max(probabilities, key=probabilities.get)
        return CrossEncoderPrediction(label, round(probabilities[label], 4), probabilities)

    def predict_proba(self, left: str, right: str) -> dict[str, float]:
        """Return softmax probabilities for a text pair."""
        _hidden, _logits, probabilities = self._forward(self.cross_features(left, right))
        return {label: round(probability, 6) for label, probability in zip(self.labels, probabilities)}

    def train(
        self,
        examples: list[CrossEncoderExample],
        epochs: int = 140,
        learning_rate: float = 0.05,
        l2: float = 0.0005,
    ) -> dict[str, float]:
        """Train the cross-encoder classifier with SGD cross-entropy."""
        if not examples:
            raise ValueError("Cannot train cross-encoder with no examples.")
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        for example in examples:
            if example.label not in label_to_index:
                raise ValueError(f"Unknown label {example.label!r}; expected one of {self.labels}.")

        rng = random.Random(37)
        for _epoch in range(epochs):
            shuffled = examples[:]
            rng.shuffle(shuffled)
            for example in shuffled:
                features = self.cross_features(example.left, example.right)
                hidden, _logits, probabilities = self._forward(features)
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
        return evaluate_cross_encoder(self, examples)

    def cross_features(self, left: str, right: str) -> list[float]:
        """Encode a text pair as joint cross-encoder interaction features."""
        left_tokens = tokenize(left)
        right_tokens = tokenize(right)
        left_vec = self._mean_segment_vector(left_tokens, 0)
        right_vec = self._mean_segment_vector(right_tokens, 1)
        diff = [abs(left_vec[index] - right_vec[index]) for index in range(self.embedding_dim)]
        product = [left_vec[index] * right_vec[index] for index in range(self.embedding_dim)]
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        overlap = len(left_set & right_set) / max(1, len(left_set | right_set))
        scalar_features = [
            overlap,
            len(left_tokens) / 200,
            len(right_tokens) / 200,
            1.0 if has_negation(left_tokens) != has_negation(right_tokens) else 0.0,
            1.0,
        ]
        return left_vec + right_vec + diff + product + scalar_features

    def _mean_segment_vector(self, tokens: list[str], segment: int) -> list[float]:
        if not tokens:
            tokens = [UNK]
        values = [0.0 for _dim in range(self.embedding_dim)]
        for token in tokens:
            token_id = self.vocabulary.get(token, self.vocabulary[UNK])
            token_vec = self.token_embeddings[token_id]
            segment_vec = self.segment_embeddings[segment]
            for dim in range(self.embedding_dim):
                values[dim] += token_vec[dim] + segment_vec[dim]
        norm = max(1, len(tokens))
        return [value / norm for value in values]

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
        return hidden, logits, softmax(logits)

    def save(self, path: Path) -> Path:
        """Persist the trained cross-encoder artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": MODEL_TYPE,
            "labels": self.labels,
            "vocabulary": self.vocabulary,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "token_embeddings": self.token_embeddings,
            "segment_embeddings": self.segment_embeddings,
            "weights1": self.weights1,
            "bias1": self.bias1,
            "weights2": self.weights2,
            "bias2": self.bias2,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def build_vocabulary(examples: list[CrossEncoderExample], max_vocab: int = MAX_VOCAB) -> dict[str, int]:
    """Build a deterministic vocabulary from pair examples."""
    counter: Counter[str] = Counter()
    for example in examples:
        counter.update(tokenize(example.left))
        counter.update(tokenize(example.right))
    vocabulary = {UNK: 0}
    for token, _count in counter.most_common(max_vocab - 1):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def train_cross_encoder_model(
    examples: list[CrossEncoderExample],
    labels: list[str],
    output_path: Path,
    task: str,
    epochs: int = 140,
    learning_rate: float = 0.05,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> tuple[TinyCrossEncoder, dict[str, float]]:
    """Train and save a local neural cross-encoder artifact."""
    model = TinyCrossEncoder(labels, build_vocabulary(examples), hidden_dim=hidden_dim, metadata={"task": task})
    metrics = model.train(examples, epochs=epochs, learning_rate=learning_rate)
    model.metadata.update({"task": task, "examples": len(examples), "metrics": metrics})
    model.save(output_path)
    return model, metrics


def load_cross_encoder_model(path: Path) -> TinyCrossEncoder:
    """Load a local neural cross-encoder artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unsupported cross-encoder model type: {payload.get('model_type')}")
    return TinyCrossEncoder(
        labels=[str(label) for label in payload["labels"]],
        vocabulary={str(key): int(value) for key, value in payload["vocabulary"].items()},
        embedding_dim=int(payload["embedding_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        token_embeddings=payload["token_embeddings"],
        segment_embeddings=payload["segment_embeddings"],
        weights1=payload["weights1"],
        bias1=payload["bias1"],
        weights2=payload["weights2"],
        bias2=payload["bias2"],
        metadata=payload.get("metadata", {}),
    )


def evaluate_cross_encoder(model: TinyCrossEncoder, examples: list[CrossEncoderExample]) -> dict[str, float]:
    """Evaluate training accuracy and confidence."""
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


def tokenize(text: str) -> list[str]:
    """Tokenize text for the local cross-encoder."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def has_negation(tokens: list[str]) -> bool:
    """Return whether a token sequence contains a negation marker."""
    return any(token in NEGATIONS for token in tokens)


def softmax(logits: list[float]) -> list[float]:
    """Stable softmax."""
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]
