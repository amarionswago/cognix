"""Natural Language Inference backends for claim-pair judgment."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.cross_encoder_model import TinyCrossEncoder, load_cross_encoder_model
from app.services.pair_model import PairTextModel, load_pair_model
from app.services.transformer_lora_nli import TransformerLoRANLIModel, load_transformer_lora_nli_model

LOCAL_CROSS_ENCODER_BACKENDS = {"cognix-cross-encoder", "local-cross-encoder", "tiny-cross-encoder"}
TRANSFORMER_LORA_BACKENDS = {"cognix-transformer-lora", "transformer-lora", "lora-nli"}


def classify_with_optional_nli(left: str, right: str) -> tuple[str, float, str] | None:
    """Classify a claim pair with a trained NLI model when configured."""
    settings = get_settings()
    backend = settings.nli_backend.strip().lower()
    if backend in TRANSFORMER_LORA_BACKENDS:
        model = load_transformer_lora_nli()
        if model is None:
            return None
        prediction = model.predict(left, right)
        return prediction.label, prediction.confidence, str(settings.resolved_transformer_lora_nli_model_path())
    if backend in LOCAL_CROSS_ENCODER_BACKENDS:
        model = load_local_cross_encoder_nli()
        if model is None:
            return None
        prediction = model.predict(left, right)
        if prediction.label == "contradiction":
            return "contradiction", prediction.confidence, str(settings.resolved_local_cross_encoder_nli_model_path())
        if prediction.label == "unrelated":
            return "unrelated", prediction.confidence, str(settings.resolved_local_cross_encoder_nli_model_path())
        return "related", prediction.confidence, str(settings.resolved_local_cross_encoder_nli_model_path())
    if backend in {"cognix-pair", "pair-mlp", "local-pair"}:
        model = load_pair_nli()
        if model is None:
            return None
        prediction = model.predict(left, right)
        if prediction.label == "contradiction":
            return "contradiction", prediction.confidence, str(settings.resolved_pair_nli_model_path())
        if prediction.label == "unrelated":
            return "unrelated", prediction.confidence, str(settings.resolved_pair_nli_model_path())
        return "related", prediction.confidence, str(settings.resolved_pair_nli_model_path())
    if backend not in {"cross-encoder", "cross_encoder", "neural", "nli"}:
        return None
    model = load_nli_model()
    if model is None:
        return None
    try:
        scores = model.predict([(left, right)])
    except Exception:
        return None
    labels = model_labels(model)
    score_values = [float(value) for value in scores[0]]
    best_index = max(range(len(score_values)), key=lambda index: score_values[index])
    raw_label = labels.get(best_index, str(best_index)).lower()
    confidence = max(0.0, min(1.0, score_values[best_index]))
    if "contradiction" in raw_label:
        return "contradiction", confidence, get_settings().nli_model
    if "entail" in raw_label or "neutral" in raw_label:
        return "related", confidence, get_settings().nli_model
    return "related", confidence, get_settings().nli_model


@lru_cache(maxsize=1)
def load_nli_model():
    """Load the configured NLI cross-encoder lazily."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        return CrossEncoder(get_settings().nli_model)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_local_cross_encoder_nli() -> TinyCrossEncoder | None:
    """Load the local trained NLI cross-encoder if present."""
    path = get_settings().resolved_local_cross_encoder_nli_model_path()
    if not path.exists():
        return None
    try:
        return load_cross_encoder_model(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_transformer_lora_nli() -> TransformerLoRANLIModel | None:
    """Load the local transformer LoRA NLI artifact if present."""
    path = get_settings().resolved_transformer_lora_nli_model_path()
    if not path.exists():
        return None
    try:
        return load_transformer_lora_nli_model(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_pair_nli() -> PairTextModel | None:
    """Load the local trained pair NLI model if present."""
    path = get_settings().resolved_pair_nli_model_path()
    if not path.exists():
        return None
    try:
        return load_pair_model(path)
    except Exception:
        return None


def model_labels(model) -> dict[int, str]:
    """Return label mapping for a CrossEncoder if available."""
    config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        return {int(key): str(value) for key, value in id2label.items()}
    return {0: "contradiction", 1: "entailment", 2: "neutral"}
