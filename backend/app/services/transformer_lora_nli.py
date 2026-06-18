"""Self-contained transformer LoRA NLI model for Cognix claim pairs.

This module gives Cognix a verified local transformer+LoRA fine-tuning path for
its NLI contradiction model without requiring a large chat-model download. The
base encoder is a frozen tiny transformer feature extractor; training updates
only low-rank LoRA adapter matrices and classifier bias for the labels
`contradiction`, `related`, and `unrelated`.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_TYPE = "cognix_transformer_lora_nli_v1"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]*")
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "isn't", "aren't", "doesn't", "don't", "failed"}
OPPOSING_TERMS = {
    "safe": {"unsafe", "not-safe"},
    "successfully": {"failed", "failure"},
    "processed": {"failed"},
    "increase": {"decrease", "reduce", "reduces", "lower"},
    "higher": {"lower"},
    "supports": {"contradicts", "refutes"},
    "true": {"false"},
}
UNK = "<unk>"
CLS = "<cls>"
SEP = "<sep>"
DEFAULT_LABELS = ["contradiction", "related", "unrelated"]


@dataclass(frozen=True)
class TransformerLoRANLIExample:
    """One labeled NLI claim-pair example."""

    left: str
    right: str
    label: str


@dataclass(frozen=True)
class TransformerLoRANLIPrediction:
    """Prediction from the transformer LoRA NLI model."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class TransformerLoRANLIModel:
    """Frozen tiny transformer encoder with trainable LoRA classifier adapter."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported transformer LoRA NLI model type: {payload.get('model_type')}")
        self.payload = payload
        self.labels = [str(label) for label in payload["labels"]]
        self.vocabulary = {str(key): int(value) for key, value in payload["vocabulary"].items()}
        self.max_length = int(payload["max_length"])
        self.embedding_dim = int(payload["embedding_dim"])
        self.hidden_dim = int(payload["hidden_dim"])
        self.rank = int(payload["rank"])
        self.lora_alpha = float(payload["lora_alpha"])
        self.feature_dim = int(payload["feature_dim"])
        self.token_embeddings = np.array(payload["token_embeddings"], dtype=np.float64)
        self.position_embeddings = np.array(payload["position_embeddings"], dtype=np.float64)
        self.wq = np.array(payload["wq"], dtype=np.float64)
        self.wk = np.array(payload["wk"], dtype=np.float64)
        self.wv = np.array(payload["wv"], dtype=np.float64)
        self.wo = np.array(payload["wo"], dtype=np.float64)
        self.ff1 = np.array(payload["ff1"], dtype=np.float64)
        self.ff2 = np.array(payload["ff2"], dtype=np.float64)
        self.classifier_base = np.array(payload["classifier_base"], dtype=np.float64)
        self.lora_a = np.array(payload["lora_a"], dtype=np.float64)
        self.lora_b = np.array(payload["lora_b"], dtype=np.float64)
        self.bias = np.array(payload["bias"], dtype=np.float64)
        self.metadata = dict(payload.get("metadata") or {})

    def predict(self, left: str, right: str) -> TransformerLoRANLIPrediction:
        """Predict the NLI label for a claim pair."""
        probabilities = self.predict_proba(left, right)
        label = max(probabilities, key=probabilities.get)
        return TransformerLoRANLIPrediction(label, round(probabilities[label], 4), probabilities)

    def predict_proba(self, left: str, right: str) -> dict[str, float]:
        """Return label probabilities for a claim pair."""
        feature = self.encode_pair(left, right)
        logits = self._logits(feature)
        probabilities = stable_softmax(logits)
        return {label: round(float(probability), 6) for label, probability in zip(self.labels, probabilities)}

    def encode_pair(self, left: str, right: str) -> np.ndarray:
        """Encode a claim pair with a frozen transformer and pair features."""
        left_tokens = tokenize(left)
        right_tokens = tokenize(right)
        token_ids = self._pair_token_ids(left_tokens, right_tokens)
        positions = np.arange(len(token_ids))
        x = self.token_embeddings[token_ids] + self.position_embeddings[positions]

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        scores = (q @ k.T) / math.sqrt(self.embedding_dim)
        attention = row_softmax(scores)
        attention_out = (attention @ v) @ self.wo
        x = layer_norm(x + attention_out)
        feed_forward = np.tanh(x @ self.ff1) @ self.ff2
        x = layer_norm(x + feed_forward)
        cls_vector = x[0]
        cls_norm = np.linalg.norm(cls_vector)
        if cls_norm > 0:
            cls_vector = cls_vector / cls_norm

        left_set = set(left_tokens)
        right_set = set(right_tokens)
        overlap = len(left_set & right_set) / max(1, len(left_set | right_set))
        min_overlap = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
        scalar_features = np.array(
            [
                overlap,
                min_overlap,
                1.0 if has_negation(left_tokens) != has_negation(right_tokens) else 0.0,
                1.0 if has_opposing_terms(left_set, right_set) else 0.0,
                len(left_tokens) / 80.0,
                len(right_tokens) / 80.0,
                1.0,
            ],
            dtype=np.float64,
        )
        return np.concatenate([cls_vector, scalar_features])

    def _pair_token_ids(self, left_tokens: list[str], right_tokens: list[str]) -> np.ndarray:
        available = self.max_length - 3
        left_budget = max(1, available // 2)
        right_budget = max(1, available - left_budget)
        tokens = [CLS, *left_tokens[:left_budget], SEP, *right_tokens[:right_budget], SEP]
        ids = [self.vocabulary.get(token, self.vocabulary[UNK]) for token in tokens[: self.max_length]]
        return np.array(ids, dtype=np.int64)

    def _logits(self, feature: np.ndarray) -> np.ndarray:
        scale = self.lora_alpha / max(1, self.rank)
        lora_delta = (feature @ self.lora_a) @ self.lora_b
        return feature @ self.classifier_base + scale * lora_delta + self.bias

    def train(
        self,
        examples: list[TransformerLoRANLIExample],
        epochs: int = 260,
        learning_rate: float = 0.03,
        l2: float = 0.0005,
    ) -> dict[str, float]:
        """Fine-tune LoRA adapter matrices with SGD cross-entropy."""
        if not examples:
            raise ValueError("Cannot train transformer LoRA NLI with no examples.")
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        features = []
        targets = []
        for example in examples:
            if example.label not in label_to_index:
                raise ValueError(f"Unknown label {example.label!r}; expected one of {self.labels}.")
            features.append(self.encode_pair(example.left, example.right))
            targets.append(label_to_index[example.label])

        rng = random.Random(911)
        scale = self.lora_alpha / max(1, self.rank)
        for _epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for index in order:
                feature = features[index]
                target = targets[index]
                adapter_hidden = feature @ self.lora_a
                logits = feature @ self.classifier_base + scale * (adapter_hidden @ self.lora_b) + self.bias
                probabilities = stable_softmax(logits)
                grad_logits = probabilities
                grad_logits[target] -= 1.0

                old_b = self.lora_b.copy()
                grad_b = np.clip(scale * np.outer(adapter_hidden, grad_logits) + l2 * self.lora_b, -1.0, 1.0)
                grad_a = np.clip(scale * np.outer(feature, old_b @ grad_logits) + l2 * self.lora_a, -1.0, 1.0)
                grad_logits = np.clip(grad_logits, -1.0, 1.0)
                self.lora_b -= learning_rate * grad_b
                self.lora_a -= learning_rate * grad_a
                self.bias -= learning_rate * grad_logits
        return evaluate_transformer_lora_nli(self, examples)

    def save(self, path: Path) -> Path:
        """Persist the trained transformer LoRA NLI artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.payload.update(
            {
                "lora_a": self.lora_a.tolist(),
                "lora_b": self.lora_b.tolist(),
                "bias": self.bias.tolist(),
                "metadata": self.metadata,
            }
        )
        path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def train_transformer_lora_nli_model(
    examples: list[TransformerLoRANLIExample],
    output_path: Path,
    epochs: int = 260,
    learning_rate: float = 0.03,
    embedding_dim: int = 24,
    hidden_dim: int = 48,
    rank: int = 4,
    lora_alpha: float = 8.0,
    max_length: int = 48,
) -> tuple[TransformerLoRANLIModel, dict[str, float]]:
    """Create, LoRA fine-tune, and save a transformer NLI artifact."""
    vocabulary = build_vocabulary(examples)
    payload = initialize_payload(
        vocabulary=vocabulary,
        labels=DEFAULT_LABELS,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        lora_alpha=lora_alpha,
        max_length=max_length,
        seed=9001,
    )
    model = TransformerLoRANLIModel(payload)
    metrics = model.train(examples, epochs=epochs, learning_rate=learning_rate)
    model.metadata.update(
        {
            "task": "nli",
            "method": MODEL_TYPE,
            "examples": len(examples),
            "metrics": metrics,
            "trainable_parameters": trainable_parameter_count(model),
        }
    )
    model.save(output_path)
    return model, metrics


def load_transformer_lora_nli_model(path: Path) -> TransformerLoRANLIModel:
    """Load a trained transformer LoRA NLI artifact."""
    return TransformerLoRANLIModel(json.loads(path.read_text(encoding="utf-8")))


def initialize_payload(
    vocabulary: dict[str, int],
    labels: list[str],
    embedding_dim: int,
    hidden_dim: int,
    rank: int,
    lora_alpha: float,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    """Initialize frozen transformer weights and trainable LoRA matrices."""
    rng = np.random.default_rng(seed)
    feature_dim = embedding_dim + 7
    return {
        "model_type": MODEL_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "vocabulary": vocabulary,
        "max_length": max_length,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "feature_dim": feature_dim,
        "token_embeddings": rng.normal(0.0, 0.16, size=(len(vocabulary), embedding_dim)).tolist(),
        "position_embeddings": rng.normal(0.0, 0.04, size=(max_length, embedding_dim)).tolist(),
        "wq": rng.normal(0.0, 0.10, size=(embedding_dim, embedding_dim)).tolist(),
        "wk": rng.normal(0.0, 0.10, size=(embedding_dim, embedding_dim)).tolist(),
        "wv": rng.normal(0.0, 0.10, size=(embedding_dim, embedding_dim)).tolist(),
        "wo": rng.normal(0.0, 0.10, size=(embedding_dim, embedding_dim)).tolist(),
        "ff1": rng.normal(0.0, 0.10, size=(embedding_dim, hidden_dim)).tolist(),
        "ff2": rng.normal(0.0, 0.10, size=(hidden_dim, embedding_dim)).tolist(),
        "classifier_base": rng.normal(0.0, 0.02, size=(feature_dim, len(labels))).tolist(),
        "lora_a": rng.normal(0.0, 0.005, size=(feature_dim, rank)).tolist(),
        "lora_b": rng.normal(0.0, 0.005, size=(rank, len(labels))).tolist(),
        "bias": np.zeros(len(labels), dtype=np.float64).tolist(),
        "metadata": {
            "base_encoder": "frozen_tiny_transformer_encoder",
            "adapter": "lora_classifier_adapter",
            "trainable": ["lora_a", "lora_b", "bias"],
        },
    }


def build_vocabulary(examples: list[TransformerLoRANLIExample], max_vocab: int = 512) -> dict[str, int]:
    """Build deterministic vocabulary for claim-pair transformer encoding."""
    counter: Counter[str] = Counter()
    for example in examples:
        counter.update(tokenize(example.left))
        counter.update(tokenize(example.right))
    vocabulary = {UNK: 0, CLS: 1, SEP: 2}
    for token, _count in counter.most_common(max_vocab - len(vocabulary)):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def evaluate_transformer_lora_nli(model: TransformerLoRANLIModel, examples: list[TransformerLoRANLIExample]) -> dict[str, float]:
    """Return simple training metrics for the NLI artifact."""
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
        "trainable_parameters": float(trainable_parameter_count(model)),
    }


def trainable_parameter_count(model: TransformerLoRANLIModel) -> int:
    """Count trainable LoRA adapter and bias parameters."""
    return int(model.lora_a.size + model.lora_b.size + model.bias.size)


def tokenize(text: str) -> list[str]:
    """Tokenize claim text."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def has_negation(tokens: list[str]) -> bool:
    """Return whether a token sequence contains a negation marker."""
    return any(token in NEGATIONS for token in tokens)


def has_opposing_terms(left_terms: set[str], right_terms: set[str]) -> bool:
    """Return whether two term sets include explicit opposition."""
    for term, opposites in OPPOSING_TERMS.items():
        if term in left_terms and opposites & right_terms:
            return True
        if term in right_terms and opposites & left_terms:
            return True
    return False


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable vector softmax."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def row_softmax(scores: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax for attention weights."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def layer_norm(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Layer-normalize the last dimension."""
    mean = np.mean(values, axis=-1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon)
