"""Advanced ML capability readiness checks for Cognix."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.database import db_session
from app.db.calibration import load_calibration_model
from app.services.embeddings import sentence_transformer_available


def advanced_ml_readiness() -> dict[str, Any]:
    """Return machine-verifiable status for advanced ML capabilities."""
    settings = get_settings()
    return {
        "summary": readiness_summary(),
        "capabilities": [
            embedding_status(settings),
            reranker_status(settings),
            nli_status(settings),
            training_status(settings),
            adapter_status(),
            micro_synthesis_status(settings),
            sft_adapter_status(settings),
            vision_status(settings),
            calibration_status(),
            large_eval_status(),
        ],
    }


def readiness_summary() -> dict[str, int]:
    """Return counts by readiness state."""
    caps = [
        embedding_status(get_settings()),
        reranker_status(get_settings()),
        nli_status(get_settings()),
        training_status(get_settings()),
        adapter_status(),
        micro_synthesis_status(get_settings()),
        sft_adapter_status(get_settings()),
        vision_status(get_settings()),
        calibration_status(),
        large_eval_status(),
    ]
    counts = {"ready": 0, "configured": 0, "fallback": 0, "missing": 0}
    for cap in caps:
        state = str(cap["state"])
        counts[state] = counts.get(state, 0) + 1
    return counts


def embedding_status(settings) -> dict[str, Any]:
    backend = settings.local_embedding_backend
    if backend.lower() in {"sentence-transformer", "sentence-transformers", "neural", "local-neural"}:
        ready = sentence_transformer_available()
        return capability(
            "local_neural_embeddings",
            "ready" if ready else "missing",
            f"SentenceTransformers backend configured: {settings.sentence_transformer_model}",
            {"model": settings.sentence_transformer_model, "dependency": "sentence-transformers"},
        )
    return capability("local_neural_embeddings", "fallback", "Hash embeddings are active.", {"backend": backend})


def reranker_status(settings) -> dict[str, Any]:
    backend = settings.reranker_backend.lower()
    local_cross_path = settings.resolved_local_cross_encoder_reranker_model_path()
    if backend in {"cognix-cross-encoder", "local-cross-encoder", "tiny-cross-encoder"}:
        exists = local_cross_path.exists()
        return capability(
            "neural_cross_encoder_reranker",
            "ready" if exists else "missing",
            "Local trained Cognix neural cross-encoder reranker artifact is available."
            if exists
            else "Cognix cross-encoder reranker is selected but no trained artifact exists yet.",
            {"backend": settings.reranker_backend, "model_path": str(local_cross_path), "artifact_exists": exists},
        )
    pair_path = settings.resolved_pair_reranker_model_path()
    if backend in {"cognix-pair", "pair-mlp", "local-pair"}:
        exists = pair_path.exists()
        return capability(
            "neural_cross_encoder_reranker",
            "ready" if exists else "missing",
            "Local trained Cognix pair reranker artifact is available."
            if exists
            else "Cognix pair reranker is selected but no trained artifact exists yet.",
            {"backend": settings.reranker_backend, "model_path": str(pair_path), "artifact_exists": exists},
        )
    configured = backend in {"cross-encoder", "cross_encoder", "neural"}
    dependency = importlib.util.find_spec("sentence_transformers") is not None
    if configured and dependency:
        state = "configured"
    elif configured:
        state = "missing"
    else:
        state = "fallback"
    return capability(
        "neural_cross_encoder_reranker",
        state,
        "Cross-encoder reranker configured." if configured else "Deterministic reranker is active.",
        {"backend": settings.reranker_backend, "model": settings.cross_encoder_model, "dependency_present": dependency},
    )


def nli_status(settings) -> dict[str, Any]:
    backend = settings.nli_backend.lower()
    transformer_lora_path = settings.resolved_transformer_lora_nli_model_path()
    if backend in {"cognix-transformer-lora", "transformer-lora", "lora-nli"}:
        exists = transformer_lora_path.exists()
        return capability(
            "trained_nli_contradiction_model",
            "ready" if exists else "missing",
            "Local transformer LoRA NLI artifact is selected and available."
            if exists
            else "Transformer LoRA NLI backend is selected but no trained artifact exists yet.",
            {
                "backend": settings.nli_backend,
                "model_path": str(transformer_lora_path),
                "artifact_exists": exists,
                "training_command": ".venv/bin/python backend/training/train_transformer_lora_nli.py --register-artifact",
            },
        )
    local_cross_path = settings.resolved_local_cross_encoder_nli_model_path()
    if backend in {"cognix-cross-encoder", "local-cross-encoder", "tiny-cross-encoder"}:
        exists = local_cross_path.exists()
        return capability(
            "trained_nli_contradiction_model",
            "ready" if exists else "missing",
            "Local trained Cognix neural cross-encoder NLI artifact is available."
            if exists
            else "Cognix cross-encoder NLI is selected but no trained artifact exists yet.",
            {"backend": settings.nli_backend, "model_path": str(local_cross_path), "artifact_exists": exists},
        )
    pair_path = settings.resolved_pair_nli_model_path()
    if backend in {"cognix-pair", "pair-mlp", "local-pair"}:
        exists = pair_path.exists()
        return capability(
            "trained_nli_contradiction_model",
            "ready" if exists else "missing",
            "Local trained Cognix pair NLI artifact is available."
            if exists
            else "Cognix pair NLI is selected but no trained artifact exists yet.",
            {"backend": settings.nli_backend, "model_path": str(pair_path), "artifact_exists": exists},
        )
    configured = backend in {"cross-encoder", "cross_encoder", "neural", "nli"}
    dependency = importlib.util.find_spec("sentence_transformers") is not None
    if configured and dependency:
        state = "configured"
    elif configured:
        state = "missing"
    else:
        state = "fallback"
    return capability(
        "trained_nli_contradiction_model",
        state,
        "Trained NLI backend configured." if configured else "Heuristic NLI fallback is active.",
        {"backend": settings.nli_backend, "model": settings.nli_model, "dependency_present": dependency},
    )


def training_status(settings) -> dict[str, Any]:
    dependencies = ["torch", "transformers", "datasets", "peft", "trl", "accelerate"]
    present = {name: importlib.util.find_spec(name) is not None for name in dependencies}
    ready = all(present.values())
    return capability(
        "lora_fine_tuning_stack",
        "ready" if ready else "missing",
        "LoRA training dependencies are installed." if ready else "LoRA dry-run works, but training dependencies are missing.",
        {"dependencies": present},
    )


def adapter_status() -> dict[str, Any]:
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT name, path, status, metrics_json
            FROM model_artifacts
            WHERE artifact_type='lora_adapter'
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()
    trained = [row for row in rows if row["status"] in {"trained", "packaged", "active"} and Path(str(row["path"])).exists()]
    state = "ready" if trained else ("configured" if rows else "missing")
    return capability(
        "custom_cognix_adapter_artifact",
        state,
        "A trained/packaged adapter artifact exists." if trained else "No verified trained Cognix adapter artifact exists yet.",
        {"artifacts": rows},
    )


def micro_synthesis_status(settings) -> dict[str, Any]:
    """Report whether Cognix's local trained synthesis policy is usable."""
    path = settings.resolved_cognix_micro_model_path()
    exists = path.exists()
    selected = settings.synthesis_backend.lower() in {"cognix-micro", "micro", "local-trained"}
    if exists and selected:
        state = "ready"
        message = "Local trained Cognix micro-synthesis artifact is selected and available."
    elif exists:
        state = "configured"
        message = "Local trained Cognix micro-synthesis artifact exists but is not the active synthesis backend."
    elif selected:
        state = "missing"
        message = "Cognix micro-synthesis backend is selected but no trained artifact exists yet."
    else:
        state = "fallback"
        message = "Provider/deterministic answer synthesis is active."
    return capability(
        "custom_cognix_micro_synthesis_model",
        state,
        message,
        {
            "backend": settings.synthesis_backend,
            "model_path": str(path),
            "artifact_exists": exists,
            "training_command": ".venv/bin/python backend/training/train_cognix_micro_model.py --dataset data/exports/training/qa_citation-reviewed-sft.jsonl --register-artifact",
        },
    )


def sft_adapter_status(settings) -> dict[str, Any]:
    """Report whether Cognix's local trained SFT adapter is usable."""
    path = settings.resolved_cognix_sft_adapter_path()
    exists = path.exists()
    selected = settings.synthesis_backend.lower() in {"cognix-sft-adapter", "sft-adapter", "local-sft"}
    if exists and selected:
        state = "ready"
        message = "Local trained Cognix SFT adapter is selected and available."
    elif exists:
        state = "configured"
        message = "Local trained Cognix SFT adapter exists but is not the active synthesis backend."
    elif selected:
        state = "missing"
        message = "Cognix SFT adapter backend is selected but no trained artifact exists yet."
    else:
        state = "fallback"
        message = "Provider/deterministic or micro-synthesis answer path is active."
    return capability(
        "custom_cognix_sft_adapter",
        state,
        message,
        {
            "backend": settings.synthesis_backend,
            "model_path": str(path),
            "artifact_exists": exists,
            "training_command": ".venv/bin/python backend/training/train_sft_adapter.py --dataset data/exports/training/qa_citation-reviewed-sft.jsonl --register-artifact",
        },
    )


def vision_status(settings) -> dict[str, Any]:
    backend = settings.vision_backend.lower()
    tesseract = shutil.which("tesseract")
    pdftoppm = shutil.which("pdftoppm")
    if backend == "openai":
        return capability(
            "ocr_vision_pipeline",
            "configured",
            "OpenAI vision is configured with local OCR/PDF OCR fallback.",
            {
                "backend": backend,
                "local_tesseract": bool(tesseract),
                "local_pdftoppm": bool(pdftoppm),
                "openai_vision_model": settings.openai_vision_model,
            },
        )
    local_ready = bool(tesseract and pdftoppm)
    return capability(
        "ocr_vision_pipeline",
        "ready" if local_ready else "fallback",
        "Local image and scanned-PDF OCR tools are available."
        if local_ready
        else "Image/PDF tracking works, but full local OCR requires tesseract and pdftoppm.",
        {"backend": backend, "local_tesseract": bool(tesseract), "local_pdftoppm": bool(pdftoppm)},
    )


def calibration_status() -> dict[str, Any]:
    model = load_calibration_model("answer_confidence")
    with db_session() as conn:
        count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM model_predictions mp
            JOIN prediction_outcomes po ON po.prediction_id = mp.id
            WHERE mp.task='answer_confidence'
            """
        ).fetchone()["count"]
    if model is not None:
        return capability(
            "calibrated_probability_model",
            "ready",
            "Persisted logistic answer-confidence calibrator is active.",
            {
                "answer_confidence_outcomes": int(count),
                "method": model.method,
                "examples": model.examples,
                "positive_examples": model.positive_examples,
                "negative_examples": model.negative_examples,
                "brier_score": model.brier_score,
                "log_loss": model.log_loss,
            },
        )
    return capability(
        "calibrated_probability_model",
        "ready" if int(count) >= 5 else "configured",
        "Empirical calibration is active." if int(count) >= 5 else "Calibration plumbing exists; at least 5 outcomes are needed for active calibration.",
        {"answer_confidence_outcomes": int(count), "minimum_for_active_calibration": 5},
    )


def large_eval_status() -> dict[str, Any]:
    return capability(
        "large_file_benchmark_suite",
        "ready",
        "Smoke and multi-family large eval commands are available.",
        {
            "smoke_command": ".venv/bin/python backend/evals/run_evals.py",
            "large_command": ".venv/bin/python backend/evals/run_evals.py --suite large --report data/eval-large-report.json",
            "families": ["html", "csv", "json", "python", "log", "markdown", "pdf-fallback", "email"],
        },
    )


def capability(name: str, state: str, message: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "state": state, "message": message, "detail": detail}
