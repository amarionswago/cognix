"""Dependency-free Cognix SFT adapter.

This adapter trains from chat-style SFT JSONL and learns a small neural
prototype selector over question/evidence features. It is a real trained local
artifact, but it is not a foundation model or transformer LoRA adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.cognix_micro_model import load_sft_examples
from app.services.retrieval_types import RetrievedChunk

MODEL_TYPE = "cognix_sft_adapter_v1"
FEATURE_DIM = 128
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]*")


@dataclass(frozen=True)
class AdapterPrototype:
    """One learned answer prototype from SFT data."""

    question: str
    assistant: str
    title: str
    answer_heading: str


class CognixSFTAdapter:
    """Tiny trained SFT adapter that selects answer prototypes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported SFT adapter model type: {payload.get('model_type')}")
        self.payload = payload
        self.prototypes = [
            AdapterPrototype(
                question=str(item["question"]),
                assistant=str(item["assistant"]),
                title=str(item.get("title") or "Research Memo"),
                answer_heading=str(item.get("answer_heading") or "Answer"),
            )
            for item in payload.get("prototypes", [])
        ]
        self.weights = [[float(value) for value in row] for row in payload.get("weights", [])]
        self.bias = [float(value) for value in payload.get("bias", [])]

    def select_prototype(self, question: str, chunks: list[RetrievedChunk]) -> tuple[AdapterPrototype, float]:
        """Select the closest learned answer prototype."""
        if not self.prototypes:
            raise ValueError("SFT adapter contains no prototypes.")
        probabilities = self.predict_proba(question, chunks)
        best = max(range(len(probabilities)), key=lambda index: probabilities[index])
        return self.prototypes[best], round(probabilities[best], 4)

    def predict_proba(self, question: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Return prototype probabilities."""
        features = adapter_features(question, chunks)
        logits = [
            self.bias[index] + sum(weight * feature for weight, feature in zip(row, features))
            for index, row in enumerate(self.weights)
        ]
        return softmax(logits)

    def synthesize(self, question: str, chunks: list[RetrievedChunk], style: str = "memo") -> str:
        """Generate an answer using the selected SFT prototype's structure."""
        prototype, confidence = self.select_prototype(question, chunks)
        title = "Answer" if style == "brief" else prototype.title
        lines = [f"# {title}", "", f"**Question:** {question}", "", f"## {prototype.answer_heading}"]
        if not chunks:
            lines.append("I could not find enough indexed evidence to answer from the current library.")
        else:
            for chunk in chunks[: (2 if style == "brief" else 5)]:
                lines.append(f"- {best_sentence(question, chunk.excerpt)} `source: {chunk.source_path}, chunk: {chunk.chunk_id}`")
        if style != "brief":
            lines.extend(["", "## Evidence"])
            for index, chunk in enumerate(chunks, start=1):
                lines.append(f"{index}. {chunk.excerpt} `source: {chunk.source_path}, chunk: {chunk.chunk_id}`")
            lines.extend(
                [
                    "",
                    "## Adapter Notes",
                    f"- SFT adapter prototype confidence: {confidence}",
                    "- Answer remains grounded in retrieved chunks.",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def save(self, path: Path) -> Path:
        """Persist the adapter artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def train_sft_adapter(dataset_path: Path, output_path: Path, epochs: int = 180, learning_rate: float = 0.12) -> tuple[CognixSFTAdapter, dict[str, Any]]:
    """Train and save a Cognix SFT adapter from chat-style JSONL."""
    examples = load_sft_examples(dataset_path)
    if not examples:
        raise ValueError("SFT adapter dataset contains no examples.")
    raw_bytes = dataset_path.read_bytes()
    prototypes = [
        {
            "question": example.question,
            "assistant": example.assistant,
            "title": first_heading(example.assistant, "#") or "Research Memo",
            "answer_heading": first_heading_containing(example.assistant, "answer") or "Answer",
        }
        for example in examples
    ]
    labels = list(range(len(examples)))
    rng = random.Random(41)
    weights = [[rng.uniform(-0.02, 0.02) for _feature in range(FEATURE_DIM)] for _label in labels]
    bias = [0.0 for _label in labels]
    feature_rows = [adapter_features(example.question, []) for example in examples]

    for _epoch in range(epochs):
        order = labels[:]
        rng.shuffle(order)
        for index in order:
            features = feature_rows[index]
            logits = [bias[label] + sum(weight * feature for weight, feature in zip(weights[label], features)) for label in labels]
            probabilities = softmax(logits)
            for label in labels:
                gradient = probabilities[label] - (1.0 if label == index else 0.0)
                for feature_index, feature_value in enumerate(features):
                    if feature_value:
                        weights[label][feature_index] -= learning_rate * gradient * feature_value
                bias[label] -= learning_rate * gradient

    payload = {
        "model_type": MODEL_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "examples": len(examples),
        "feature_dim": FEATURE_DIM,
        "prototypes": prototypes,
        "weights": weights,
        "bias": bias,
    }
    adapter = CognixSFTAdapter(payload)
    adapter.save(output_path)
    accuracy = adapter_training_accuracy(adapter, examples)
    metrics = {
        "examples": len(examples),
        "dataset_sha256": payload["dataset_sha256"],
        "prototype_count": len(prototypes),
        "training_accuracy": accuracy,
    }
    return adapter, metrics


def load_sft_adapter(path: Path) -> CognixSFTAdapter:
    """Load a trained SFT adapter artifact."""
    return CognixSFTAdapter(json.loads(path.read_text(encoding="utf-8")))


def adapter_training_accuracy(adapter: CognixSFTAdapter, examples) -> float:
    """Return training-set prototype selection accuracy."""
    correct = 0
    for index, example in enumerate(examples):
        probabilities = adapter.predict_proba(example.question, [])
        predicted = max(range(len(probabilities)), key=lambda item: probabilities[item])
        correct += int(predicted == index)
    return round(correct / max(1, len(examples)), 4)


def adapter_features(question: str, chunks: list[RetrievedChunk]) -> list[float]:
    """Build normalized hashed features from question and evidence."""
    vector = [0.0 for _feature in range(FEATURE_DIM)]
    texts = [question, *[chunk.source_path + " " + chunk.excerpt for chunk in chunks]]
    for prefix, text in enumerate(texts):
        for token in tokenize(text):
            slot = stable_hash(f"{prefix}:{token}", FEATURE_DIM - 4)
            vector[slot] += 1.0
    vector[-4] = len(chunks) / 10
    vector[-3] = sum(chunk.score for chunk in chunks) / max(1, len(chunks))
    vector[-2] = len(tokenize(question)) / 40
    vector[-1] = 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def first_heading(text: str, marker: str) -> str | None:
    """Return the first markdown heading matching a marker."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker + " "):
            return stripped[len(marker) :].strip()
    return None


def first_heading_containing(text: str, needle: str) -> str | None:
    """Return the first markdown heading containing a word."""
    lowered_needle = needle.lower()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and lowered_needle in stripped.lower():
            return stripped.lstrip("#").strip()
    return None


def best_sentence(question: str, text: str) -> str:
    """Select the sentence with most query-token overlap."""
    query_terms = set(tokenize(question))
    candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if part.strip()]
    if not candidates:
        return text.strip()
    return max(candidates, key=lambda sentence: len(query_terms & set(tokenize(sentence))))


def tokenize(text: str) -> list[str]:
    """Tokenize adapter text."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text) if len(match.group(0)) > 2]


def stable_hash(value: str, modulo: int) -> int:
    """Deterministic hash."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % modulo


def softmax(logits: list[float]) -> list[float]:
    """Stable softmax."""
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]
