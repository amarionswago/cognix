"""Cognix product evaluation runner.

This is not a unit test replacement. It builds a temporary library, ingests a
small labeled corpus, and checks product-level behavior: retrieval, evidence
grounding, intelligence findings, and large-ish file handling.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.database import db_session, init_db
from app.db.calibration import calibrated_probability, load_calibration_model, record_outcome, record_prediction, train_calibration_model
from app.services.cross_encoder_model import CrossEncoderExample, load_cross_encoder_model, train_cross_encoder_model
from app.services.ingest import run_ingest
from app.services.intelligence.runner import run_intelligence_pass
from app.services.retrieval import retrieve_evidence
from app.services.retrieval_types import RetrievedChunk
from app.services.sft_adapter import load_sft_adapter, train_sft_adapter
from app.services.transformer_lora_nli import TransformerLoRANLIExample, load_transformer_lora_nli_model, train_transformer_lora_nli_model

LARGE_MIN_TOTAL_BYTES = 2_500_000
LARGE_MAX_INGEST_SECONDS = 45.0
LARGE_MAX_RETRIEVAL_SECONDS = 4.0
LARGE_MAX_PEAK_RSS_MB = 900.0


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    metric: float
    detail: str


@dataclass(frozen=True)
class FileBenchmark:
    family: str
    source_path: str
    size_bytes: int
    status: str
    chunks: int
    extracted_documents: int
    expected_query: str
    retrieval_hit: bool
    retrieval_latency_ms: float
    top_sources: list[str]


QUESTIONS = [
    {
        "question": "any evidence on an html file talking about artificial intelligence",
        "expected_source": "articles/ai_overview.html",
    },
    {
        "question": "what does the library say about semantic search",
        "expected_source": "research/semantic_search.md",
    },
    {
        "question": "any pdfs about hacking",
        "expected_source": "documents/hacking_notes.pdf",
    },
]

LARGE_QUESTIONS = [
    {
        "family": "html",
        "question": "large html benchmark neural retrieval section",
        "expected_source": "articles/large_ai_reference.html",
    },
    {
        "family": "csv",
        "question": "large csv benchmark transaction embedding latency",
        "expected_source": "data/large_metrics.csv",
    },
    {
        "family": "json",
        "question": "large json benchmark calibration outcomes",
        "expected_source": "exports/large_records.json",
    },
    {
        "family": "code",
        "question": "large python benchmark reranker pipeline",
        "expected_source": "code/large_pipeline.py",
    },
    {
        "family": "log",
        "question": "durable ingest checkpoints semantic retrieval large file",
        "expected_source": "logs/large_eval.log",
    },
    {
        "family": "markdown",
        "question": "large markdown benchmark temporal intelligence review",
        "expected_source": "research/large_review.md",
    },
    {
        "family": "pdf_text_fallback",
        "question": "large pdf benchmark adversarial retrieval appendix",
        "expected_source": "documents/large_appendix.pdf",
    },
    {
        "family": "email",
        "question": "large email benchmark knowledge audit thread",
        "expected_source": "exports/large_thread.eml",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Cognix product evaluations.")
    parser.add_argument("--suite", choices=["smoke", "large"], default="smoke")
    parser.add_argument("--scale", choices=["standard", "stress"], default="standard")
    parser.add_argument("--report", type=Path, help="Optional path for a JSON benchmark report.")
    args = parser.parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="cognix-eval-"))
    try:
        configure_temp_instance(temp_root)
        stage("writing fixtures")
        write_fixture_corpus(get_settings().resolved_raw_dir(), suite=args.suite, scale=args.scale)
        stage("initializing database")
        init_db()
        stage("running ingest")
        start_rss = peak_rss_mb()
        ingest_started = time.perf_counter()
        ingest_result = run_ingest("eval")
        ingest_seconds = time.perf_counter() - ingest_started
        end_rss = peak_rss_mb()
        stage("running evaluation checks")
        results = [
            evaluate_ingest(ingest_result, ingest_seconds),
            evaluate_retrieval(),
            evaluate_ocr_vision_artifacts(),
            evaluate_calibrated_probability_model(),
            evaluate_local_cross_encoder_reranker(),
            evaluate_local_cross_encoder_nli(),
            evaluate_transformer_lora_nli_finetune(),
            evaluate_sft_adapter_model(),
        ]
        if args.suite == "smoke":
            results.append(evaluate_intelligence())
        if args.suite == "large":
            large_report, large_results = evaluate_large_file_benchmark(ingest_result, ingest_seconds, start_rss, end_rss, args.scale)
            results.extend(large_results)
        else:
            large_report = {}
        report = {
            "passed": all(result.passed for result in results),
            "suite": args.suite,
            "scale": args.scale,
            "results": [asdict(result) for result in results],
            "large_file_benchmark": large_report,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    finally:
        get_settings.cache_clear()
        shutil.rmtree(temp_root, ignore_errors=True)


def configure_temp_instance(temp_root: Path) -> None:
    os.environ["COGNIX_DATA_DIR"] = str(temp_root / "data")
    os.environ["COGNIX_WIKI_DIR"] = str(temp_root / "wiki")
    os.environ["COGNIX_DATABASE_PATH"] = str(temp_root / "library.sqlite")
    os.environ["COGNIX_LOCAL_EMBEDDING_BACKEND"] = "hash"
    os.environ["COGNIX_CHROMA_ENABLED"] = "false"
    get_settings.cache_clear()


def write_fixture_corpus(raw_dir: Path, suite: str = "smoke", scale: str = "standard") -> None:
    (raw_dir / "articles").mkdir(parents=True, exist_ok=True)
    (raw_dir / "code").mkdir(parents=True, exist_ok=True)
    (raw_dir / "data").mkdir(parents=True, exist_ok=True)
    (raw_dir / "research").mkdir(parents=True, exist_ok=True)
    (raw_dir / "documents").mkdir(parents=True, exist_ok=True)
    (raw_dir / "exports").mkdir(parents=True, exist_ok=True)
    (raw_dir / "logs").mkdir(parents=True, exist_ok=True)
    (raw_dir / "scans").mkdir(parents=True, exist_ok=True)
    (raw_dir / "articles" / "ai_overview.html").write_text(
        """
        <html><head><title>Artificial Intelligence Research Notes</title>
        <meta name="description" content="A concise page about neural systems."></head>
        <body><h1>Artificial Intelligence</h1>
        <p>Artificial Intelligence is the field of building systems that perform tasks requiring reasoning,
        learning, perception, and language understanding. Neural networks are one machine learning method used
        in modern AI systems.</p></body></html>
        """,
        encoding="utf-8",
    )
    (raw_dir / "research" / "semantic_search.md").write_text(
        (
            "# Semantic Search\n\n"
            "Semantic Search is a retrieval technique that finds documents by meaning rather than exact words. "
            "Semantic Search helps knowledge systems connect related evidence. "
            "Semantic Search improves research workflows by ranking conceptually similar passages.\n\n"
        )
        * 12,
        encoding="utf-8",
    )
    (raw_dir / "documents" / "hacking_notes.pdf").write_text(
        (
            "Hacking research notes. Hacking is discussed here as a cybersecurity research topic. "
            "The document covers penetration testing, defensive security, and responsible disclosure. "
        )
        * 20,
        encoding="utf-8",
    )
    (raw_dir / "research" / "coffee_positive.md").write_text(
        (
            "Coffee is safe for sleep research participants when consumed before noon. "
            "Coffee is safe for sleep research participants when consumed before noon. "
        )
        * 30,
        encoding="utf-8",
    )
    (raw_dir / "research" / "coffee_negative.md").write_text(
        (
            "Coffee is not safe for sleep research participants when consumed before noon. "
            "Coffee is not safe for sleep research participants when consumed before noon. "
        )
        * 30,
        encoding="utf-8",
    )
    (raw_dir / "logs" / "large_eval.log").write_text(
        "large file evaluation line about durable ingest checkpoints and semantic retrieval\n"
        * (400 if suite == "smoke" else large_multiplier(scale, 8000)),
        encoding="utf-8",
    )
    write_ocr_fixture_family(raw_dir)
    if suite == "large":
        write_large_fixture_family(raw_dir, scale)


def write_ocr_fixture_family(raw_dir: Path) -> None:
    """Write portable image and scanned-PDF fixtures for OCR artifact checks."""
    # 1x1 PNG. It may OCR as empty on machines with Tesseract installed, which is
    # still a valid extraction-artifact path to verify.
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lN2cLwAAAABJRU5ErkJggg=="
    )
    (raw_dir / "scans" / "ocr_whiteboard.png").write_bytes(png_bytes)
    write_blank_pdf(raw_dir / "scans" / "scanned_notes.pdf")


def write_blank_pdf(path: Path) -> None:
    """Write a valid no-text PDF so Cognix exercises scanned-PDF OCR fallback."""
    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=120)
        with path.open("wb") as handle:
            writer.write(handle)
    except Exception:
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
        )


def write_large_fixture_family(raw_dir: Path, scale: str) -> None:
    """Write multi-format large fixtures for ingest/retrieval stress."""
    multiplier = large_multiplier(scale, 1)
    (raw_dir / "articles" / "large_ai_reference.html").write_text(
        "<html><head><title>Large HTML Benchmark</title></head><body>"
        + (
            "<section><h2>Neural Retrieval Section</h2><p>"
            "This large HTML benchmark discusses neural retrieval, embeddings, and chunk ranking for Cognix. "
            "The unique marker is large html benchmark neural retrieval section.</p></section>"
        )
        * (1400 * multiplier)
        + "</body></html>",
        encoding="utf-8",
    )
    (raw_dir / "data" / "large_metrics.csv").write_text(
        "id,metric,description\n"
        + "\n".join(
            f"{index},embedding_latency,large csv benchmark transaction embedding latency row {index}"
            for index in range(2500 * multiplier)
        ),
        encoding="utf-8",
    )
    (raw_dir / "exports" / "large_records.json").write_text(
        json.dumps(
            [
                {
                    "id": index,
                    "topic": "calibration outcomes",
                    "description": f"large json benchmark calibration outcomes record {index}",
                }
                for index in range(1800 * multiplier)
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (raw_dir / "code" / "large_pipeline.py").write_text(
        (
            "def reranker_pipeline_benchmark():\n"
            "    return 'large python benchmark reranker pipeline with cross encoder scoring'\n\n"
        )
        * (1800 * multiplier),
        encoding="utf-8",
    )
    (raw_dir / "research" / "large_review.md").write_text(
        (
            "# Temporal Intelligence Review\n\n"
            "The large markdown benchmark temporal intelligence review describes stale claims, source dates, "
            "confidence warnings, and proactive library audit behavior.\n\n"
        )
        * (1800 * multiplier),
        encoding="utf-8",
    )
    (raw_dir / "documents" / "large_appendix.pdf").write_text(
        (
            "Large PDF benchmark adversarial retrieval appendix. "
            "This text-recoverable PDF fallback fixture checks that Cognix still indexes PDF-like files "
            "when the PDF parser cannot read a valid binary header. "
        )
        * (1700 * multiplier),
        encoding="utf-8",
    )
    (raw_dir / "exports" / "large_thread.eml").write_text(
        (
            "From: researcher@example.com\n"
            "To: cognix@example.com\n"
            "Subject: Large email benchmark knowledge audit thread\n\n"
            "The large email benchmark knowledge audit thread discusses proactive findings, source review, "
            "and customer-grade evidence inspection.\n\n"
        )
        * (1300 * multiplier),
        encoding="utf-8",
    )


def large_multiplier(scale: str, standard_value: int) -> int:
    """Scale fixture size without changing benchmark semantics."""
    if scale == "stress":
        return standard_value * 4
    return standard_value


def evaluate_ingest(ingest_result: dict, elapsed_seconds: float) -> EvalResult:
    discovered = int(ingest_result.get("discovered", 0))
    processed = int(ingest_result.get("processed", 0))
    failed = int(ingest_result.get("failed", 0))
    coverage = processed / max(1, discovered)
    return EvalResult(
        name="ingest_coverage",
        passed=discovered >= 6 and failed == 0 and coverage >= 0.99,
        metric=round(coverage, 4),
        detail=f"discovered={discovered}, processed={processed}, failed={failed}, elapsed_seconds={elapsed_seconds:.3f}",
    )


def evaluate_retrieval() -> EvalResult:
    hits = 0
    details = []
    for item in QUESTIONS:
        pack = retrieve_evidence(item["question"], limit=5)
        sources = [chunk.source_path for chunk in pack.chunks]
        hit = item["expected_source"] in sources
        hits += int(hit)
        details.append({"question": item["question"], "hit": hit, "sources": sources})
    recall_at_5 = hits / len(QUESTIONS)
    return EvalResult(
        name="retrieval_recall_at_5",
        passed=recall_at_5 >= 1.0,
        metric=round(recall_at_5, 4),
        detail=json.dumps(details, sort_keys=True),
    )


def evaluate_ocr_vision_artifacts() -> EvalResult:
    """Check that image and scanned-PDF inputs produce auditable extraction rows."""
    expected = {
        "scans/ocr_whiteboard.png": "image",
        "scans/scanned_notes.pdf": "pdf",
    }
    placeholders = ",".join("?" for _path in expected)
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT rf.relative_path, rf.status, ea.method, ea.confidence,
                   ea.page_count, ea.text_length, ea.warnings_json
            FROM raw_files rf
            LEFT JOIN extraction_artifacts ea ON ea.file_id = rf.id
            WHERE rf.relative_path IN ({placeholders})
            ORDER BY rf.relative_path
            """,
            tuple(expected),
        ).fetchall()
    details = []
    passed_rows = 0
    for row in rows:
        relative_path = str(row["relative_path"])
        method = str(row["method"] or "")
        text_length = int(row["text_length"] or 0)
        confidence = float(row["confidence"] or 0.0)
        family = expected.get(relative_path, "")
        method_ok = family in method and ("ocr" in method or "vision" in method)
        row_passed = (
            row["status"] == "processed"
            and method_ok
            and 0.0 <= confidence <= 1.0
            and text_length > 0
            and row["warnings_json"] is not None
        )
        passed_rows += int(row_passed)
        details.append(
            {
                "path": relative_path,
                "status": row["status"],
                "method": method,
                "confidence": round(confidence, 4),
                "page_count": int(row["page_count"] or 0),
                "text_length": text_length,
                "passed": row_passed,
            }
        )
    coverage = passed_rows / max(1, len(expected))
    return EvalResult(
        name="ocr_vision_artifact_coverage",
        passed=len(rows) == len(expected) and coverage >= 1.0,
        metric=round(coverage, 4),
        detail=json.dumps(details, sort_keys=True),
    )


def evaluate_calibrated_probability_model() -> EvalResult:
    """Train and verify a persisted answer-confidence calibration model."""
    for index in range(24):
        prediction_id = record_prediction(
            "answer_confidence",
            {"eval_case": f"supported-{index}"},
            "supported",
            0.82 + (index % 4) * 0.03,
            "cognix-eval-confidence",
        )
        record_outcome(prediction_id, "supported", reviewer="eval", notes="Synthetic eval calibration positive.")
    for index in range(24):
        prediction_id = record_prediction(
            "answer_confidence",
            {"eval_case": f"unsupported-{index}"},
            "supported",
            0.12 + (index % 4) * 0.04,
            "cognix-eval-confidence",
        )
        record_outcome(prediction_id, "unsupported", reviewer="eval", notes="Synthetic eval calibration negative.")

    model = train_calibration_model("answer_confidence", min_examples=20)
    loaded = load_calibration_model("answer_confidence")
    high = calibrated_probability("answer_confidence", 0.9)
    low = calibrated_probability("answer_confidence", 0.15)
    passed = (
        loaded is not None
        and high.applied
        and low.applied
        and high.method == "logistic_platt_score_v1"
        and low.method == "logistic_platt_score_v1"
        and high.calibrated_score > low.calibrated_score
        and high.calibrated_score >= 0.75
        and low.calibrated_score <= 0.25
    )
    return EvalResult(
        name="calibrated_probability_model",
        passed=passed,
        metric=round(high.calibrated_score - low.calibrated_score, 4),
        detail=json.dumps(
            {
                "method": model.method,
                "examples": model.examples,
                "positive_examples": model.positive_examples,
                "negative_examples": model.negative_examples,
                "brier_score": model.brier_score,
                "log_loss": model.log_loss,
                "high_calibrated": high.calibrated_score,
                "low_calibrated": low.calibrated_score,
            },
            sort_keys=True,
        ),
    )


def evaluate_local_cross_encoder_reranker() -> EvalResult:
    """Train and verify the local neural cross-encoder reranker artifact path."""
    data_dir = get_settings().resolved_data_dir()
    output = data_dir / "models" / "eval-cognix-reranker-cross-encoder.json"
    examples = [
        CrossEncoderExample("semantic search", "Semantic search retrieves documents by meaning.", "relevant"),
        CrossEncoderExample("semantic search", "Bank transaction exports list card payments.", "irrelevant"),
        CrossEncoderExample("ocr receipt", "OCR extracts visible receipt text from images.", "relevant"),
        CrossEncoderExample("ocr receipt", "Knowledge graphs connect research concepts.", "irrelevant"),
        CrossEncoderExample("large pdf ingest", "Large PDF ingest should preserve page text and chunks.", "relevant"),
        CrossEncoderExample("large pdf ingest", "Music playlists record album listening history.", "irrelevant"),
    ]
    model, metrics = train_cross_encoder_model(
        examples,
        ["irrelevant", "relevant"],
        output,
        "reranker",
        epochs=180,
        learning_rate=0.07,
    )
    loaded = load_cross_encoder_model(output)
    semantic_relevant = loaded.predict_proba("semantic search", "Semantic retrieval finds evidence by meaning.")["relevant"]
    semantic_irrelevant = loaded.predict_proba("semantic search", "Debit card exports and merchant totals.")["relevant"]
    pdf_relevant = loaded.predict_proba("large pdf ingest", "The parser chunks large PDF page text for retrieval.")["relevant"]
    pdf_irrelevant = loaded.predict_proba("large pdf ingest", "Spotify history tracks songs and albums.")["relevant"]
    separation = min(semantic_relevant - semantic_irrelevant, pdf_relevant - pdf_irrelevant)
    passed = (
        output.exists()
        and model.metadata.get("task") == "reranker"
        and metrics["accuracy"] >= 0.8
        and semantic_relevant > semantic_irrelevant
        and pdf_relevant > pdf_irrelevant
        and separation >= 0.05
    )
    return EvalResult(
        name="local_neural_cross_encoder_reranker",
        passed=passed,
        metric=round(separation, 4),
        detail=json.dumps(
            {
                "artifact": str(output),
                "accuracy": metrics["accuracy"],
                "mean_confidence": metrics["mean_confidence"],
                "semantic_relevant": semantic_relevant,
                "semantic_irrelevant": semantic_irrelevant,
                "pdf_relevant": pdf_relevant,
                "pdf_irrelevant": pdf_irrelevant,
            },
            sort_keys=True,
        ),
    )


def evaluate_local_cross_encoder_nli() -> EvalResult:
    """Train and verify the local neural cross-encoder NLI artifact path."""
    data_dir = get_settings().resolved_data_dir()
    output = data_dir / "models" / "eval-cognix-nli-cross-encoder.json"
    examples = [
        CrossEncoderExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
        CrossEncoderExample("The file processed successfully.", "The file failed to process.", "contradiction"),
        CrossEncoderExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
        CrossEncoderExample("OCR extracts text.", "OCR converts visible words into searchable text.", "related"),
        CrossEncoderExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
        CrossEncoderExample("OCR extracts text.", "LoRA trains adapter weights.", "unrelated"),
    ]
    _model, metrics = train_cross_encoder_model(
        examples,
        ["contradiction", "related", "unrelated"],
        output,
        "nli",
        epochs=240,
        learning_rate=0.07,
    )
    loaded = load_cross_encoder_model(output)
    contradiction = loaded.predict("Coffee is safe before noon.", "Coffee is not safe before noon.")
    related = loaded.predict("OCR extracts text.", "OCR converts visible words into searchable text.")
    unrelated = loaded.predict("OCR extracts text.", "LoRA trains adapter weights.")
    passed = (
        output.exists()
        and metrics["accuracy"] >= 0.8
        and contradiction.label == "contradiction"
        and related.label == "related"
        and unrelated.label == "unrelated"
        and contradiction.confidence > 0.4
    )
    return EvalResult(
        name="local_trained_nli_cross_encoder",
        passed=passed,
        metric=float(metrics["accuracy"]),
        detail=json.dumps(
            {
                "artifact": str(output),
                "accuracy": metrics["accuracy"],
                "mean_confidence": metrics["mean_confidence"],
                "contradiction": {"label": contradiction.label, "confidence": contradiction.confidence},
                "related": {"label": related.label, "confidence": related.confidence},
                "unrelated": {"label": unrelated.label, "confidence": unrelated.confidence},
            },
            sort_keys=True,
        ),
    )


def evaluate_transformer_lora_nli_finetune() -> EvalResult:
    """Train and verify Cognix's transformer LoRA NLI fine-tune path."""
    data_dir = get_settings().resolved_data_dir()
    output = data_dir / "models" / "eval-cognix-nli-transformer-lora.json"
    examples = [
        TransformerLoRANLIExample("Coffee is safe before noon.", "Coffee is not safe before noon.", "contradiction"),
        TransformerLoRANLIExample("The file processed successfully.", "The file failed to process.", "contradiction"),
        TransformerLoRANLIExample("Semantic search retrieves by meaning.", "Semantic search finds documents by meaning.", "related"),
        TransformerLoRANLIExample("OCR extracts text.", "OCR converts visible words into searchable text.", "related"),
        TransformerLoRANLIExample("Semantic search retrieves by meaning.", "Bank statements list transactions.", "unrelated"),
        TransformerLoRANLIExample("OCR extracts text.", "LoRA trains adapter weights.", "unrelated"),
    ]
    _model, metrics = train_transformer_lora_nli_model(examples, output, epochs=320)
    loaded = load_transformer_lora_nli_model(output)
    contradiction = loaded.predict("Coffee is safe before noon.", "Coffee is not safe before noon.")
    related = loaded.predict("OCR extracts text.", "OCR converts visible words into searchable text.")
    unrelated = loaded.predict("OCR extracts text.", "LoRA trains adapter weights.")
    passed = (
        output.exists()
        and metrics["accuracy"] >= 0.8
        and metrics["trainable_parameters"] > 0
        and contradiction.label == "contradiction"
        and related.label == "related"
        and unrelated.label == "unrelated"
    )
    return EvalResult(
        name="transformer_lora_nli_finetune",
        passed=passed,
        metric=float(metrics["accuracy"]),
        detail=json.dumps(
            {
                "artifact": str(output),
                "accuracy": metrics["accuracy"],
                "mean_confidence": metrics["mean_confidence"],
                "trainable_parameters": metrics["trainable_parameters"],
                "contradiction": {"label": contradiction.label, "confidence": contradiction.confidence},
                "related": {"label": related.label, "confidence": related.confidence},
                "unrelated": {"label": unrelated.label, "confidence": unrelated.confidence},
            },
            sort_keys=True,
        ),
    )


def evaluate_sft_adapter_model() -> EvalResult:
    """Train and verify Cognix's dependency-free SFT adapter."""
    data_dir = get_settings().resolved_data_dir()
    dataset = data_dir / "eval" / "sft_adapter_eval.jsonl"
    output = data_dir / "models" / "eval-cognix-sft-adapter.json"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: What is semantic search?"},
                {
                    "role": "assistant",
                    "content": "# Research Memo\n\n**Question:** What is semantic search?\n\n## Answer\n- Semantic search retrieves by meaning.\n\n## Evidence\n1. semantic.md",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Cognix."},
                {"role": "user", "content": "Question: Why does OCR matter?"},
                {
                    "role": "assistant",
                    "content": "# Research Memo\n\n**Question:** Why does OCR matter?\n\n## Answer\n- OCR turns visible text into searchable evidence.\n\n## Evidence\n1. ocr.md",
                },
            ]
        },
    ]
    dataset.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    _adapter, metrics = train_sft_adapter(dataset, output, epochs=140)
    loaded = load_sft_adapter(output)
    prototype, confidence = loaded.select_prototype("What is semantic search?", [])
    answer = loaded.synthesize(
        "What is semantic search?",
        [RetrievedChunk(1, "semantic.md", "Semantic search retrieves documents by meaning.", 0.9, "research")],
    )
    passed = (
        output.exists()
        and metrics["training_accuracy"] >= 1.0
        and "semantic search" in prototype.question.lower()
        and confidence > 0.5
        and "semantic.md" in answer
        and "Adapter Notes" in answer
    )
    return EvalResult(
        name="custom_cognix_sft_adapter",
        passed=passed,
        metric=float(metrics["training_accuracy"]),
        detail=json.dumps(
            {
                "artifact": str(output),
                "examples": metrics["examples"],
                "training_accuracy": metrics["training_accuracy"],
                "selected_prototype": prototype.question,
                "prototype_confidence": confidence,
            },
            sort_keys=True,
        ),
    )


def evaluate_intelligence() -> EvalResult:
    result = run_intelligence_pass("eval", use_llm=False)
    findings_created = int(result.get("findings_created", 0))
    return EvalResult(
        name="intelligence_findings",
        passed=result.get("status") == "completed" and findings_created >= 1 and Path(str(result.get("briefing_path"))).exists(),
        metric=float(findings_created),
        detail=json.dumps(result, sort_keys=True),
    )


def evaluate_large_file_indexing() -> EvalResult:
    hits = 0
    details = []
    for item in LARGE_QUESTIONS:
        pack = retrieve_evidence(item["question"], limit=10)
        sources = [chunk.source_path for chunk in pack.chunks]
        hit = item["expected_source"] in sources
        hits += int(hit)
        details.append({"question": item["question"], "hit": hit, "sources": sources})
    recall = hits / len(LARGE_QUESTIONS)
    return EvalResult(
        name="large_file_family_recall_at_10",
        passed=recall >= 1.0,
        metric=round(recall, 4),
        detail=json.dumps(details, sort_keys=True),
    )


def evaluate_large_file_benchmark(
    ingest_result: dict,
    ingest_seconds: float,
    start_rss_mb: float,
    end_rss_mb: float,
    scale: str,
) -> tuple[dict, list[EvalResult]]:
    """Evaluate large-file behavior across size, coverage, latency, memory, and recall."""
    file_rows = large_file_rows()
    total_bytes = sum(int(row["size_bytes"]) for row in file_rows)
    processed = sum(1 for row in file_rows if row["status"] == "processed")
    failed = sum(1 for row in file_rows if row["status"] == "failed")
    coverage = processed / max(1, len(file_rows))
    chunk_count = count_rows("chunks")
    embedding_count = count_rows("chunk_embeddings")
    throughput_mb_s = (total_bytes / 1_000_000) / max(ingest_seconds, 0.001)
    peak_mb = max(start_rss_mb, end_rss_mb)
    retrieval_benchmarks = benchmark_large_retrieval(file_rows)
    recall = sum(1 for item in retrieval_benchmarks if item.retrieval_hit) / max(1, len(retrieval_benchmarks))
    max_retrieval_seconds = max((item.retrieval_latency_ms for item in retrieval_benchmarks), default=0.0) / 1000

    report = {
        "scale": scale,
        "thresholds": {
            "min_total_bytes": LARGE_MIN_TOTAL_BYTES,
            "max_ingest_seconds": LARGE_MAX_INGEST_SECONDS,
            "max_retrieval_seconds": LARGE_MAX_RETRIEVAL_SECONDS,
            "max_peak_rss_mb": LARGE_MAX_PEAK_RSS_MB,
            "min_recall_at_10": 1.0,
            "min_supported_file_coverage": 0.99,
        },
        "summary": {
            "raw_file_count": len(file_rows),
            "supported_large_file_count": len(retrieval_benchmarks),
            "total_bytes": total_bytes,
            "processed_large_files": processed,
            "failed_large_files": failed,
            "coverage": round(coverage, 4),
            "chunks": chunk_count,
            "embeddings": embedding_count,
            "ingest_seconds": round(ingest_seconds, 4),
            "throughput_mb_s": round(throughput_mb_s, 4),
            "peak_rss_mb": round(peak_mb, 2),
            "recall_at_10": round(recall, 4),
            "max_retrieval_seconds": round(max_retrieval_seconds, 4),
            "ingest_result": ingest_result,
        },
        "files": [asdict(item) for item in retrieval_benchmarks],
    }

    return report, [
        EvalResult(
            name="large_total_bytes",
            passed=total_bytes >= LARGE_MIN_TOTAL_BYTES,
            metric=float(total_bytes),
            detail=f"total_bytes={total_bytes}, minimum={LARGE_MIN_TOTAL_BYTES}",
        ),
        EvalResult(
            name="large_supported_file_coverage",
            passed=coverage >= 0.99 and failed == 0,
            metric=round(coverage, 4),
            detail=f"processed={processed}, total={len(file_rows)}, failed={failed}",
        ),
        EvalResult(
            name="large_chunk_embedding_parity",
            passed=chunk_count > 0 and embedding_count == chunk_count,
            metric=float(embedding_count - chunk_count),
            detail=f"chunks={chunk_count}, embeddings={embedding_count}",
        ),
        EvalResult(
            name="large_ingest_throughput_mb_s",
            passed=ingest_seconds <= LARGE_MAX_INGEST_SECONDS and throughput_mb_s > 0,
            metric=round(throughput_mb_s, 4),
            detail=f"elapsed_seconds={ingest_seconds:.4f}, max_seconds={LARGE_MAX_INGEST_SECONDS}",
        ),
        EvalResult(
            name="large_peak_rss_mb",
            passed=peak_mb <= LARGE_MAX_PEAK_RSS_MB,
            metric=round(peak_mb, 2),
            detail=f"peak_rss_mb={peak_mb:.2f}, max_peak_rss_mb={LARGE_MAX_PEAK_RSS_MB}",
        ),
        EvalResult(
            name="large_file_family_recall_at_10",
            passed=recall >= 1.0,
            metric=round(recall, 4),
            detail=json.dumps([asdict(item) for item in retrieval_benchmarks], sort_keys=True),
        ),
        EvalResult(
            name="large_retrieval_latency",
            passed=max_retrieval_seconds <= LARGE_MAX_RETRIEVAL_SECONDS,
            metric=round(max_retrieval_seconds, 4),
            detail=f"max_retrieval_seconds={max_retrieval_seconds:.4f}, limit={LARGE_MAX_RETRIEVAL_SECONDS}",
        ),
    ]


def benchmark_large_retrieval(file_rows: list[sqlite3.Row]) -> list[FileBenchmark]:
    """Run per-family retrieval probes and attach database file/chunk profile data."""
    by_path = {str(row["relative_path"]): row for row in file_rows}
    benchmarks: list[FileBenchmark] = []
    for item in LARGE_QUESTIONS:
        started = time.perf_counter()
        pack = retrieve_evidence(item["question"], limit=10)
        retrieval_latency_ms = (time.perf_counter() - started) * 1000
        sources = [chunk.source_path for chunk in pack.chunks]
        row = by_path.get(item["expected_source"])
        benchmarks.append(
            FileBenchmark(
                family=item["family"],
                source_path=item["expected_source"],
                size_bytes=int(row["size_bytes"]) if row else 0,
                status=str(row["status"]) if row else "missing",
                chunks=int(row["chunks"]) if row else 0,
                extracted_documents=int(row["extracted_documents"]) if row else 0,
                expected_query=item["question"],
                retrieval_hit=item["expected_source"] in sources,
                retrieval_latency_ms=round(retrieval_latency_ms, 3),
                top_sources=sources,
            )
        )
    return benchmarks


def large_file_rows() -> list[sqlite3.Row]:
    """Return database profile rows for large benchmark fixtures."""
    expected_paths = [item["expected_source"] for item in LARGE_QUESTIONS]
    placeholders = ",".join("?" for _path in expected_paths)
    with db_session() as conn:
        return conn.execute(
            f"""
            SELECT rf.relative_path, rf.size_bytes, rf.status,
                   COUNT(DISTINCT c.id) AS chunks,
                   COUNT(DISTINCT ed.id) AS extracted_documents
            FROM raw_files rf
            LEFT JOIN extracted_documents ed ON ed.file_id = rf.id
            LEFT JOIN chunks c ON c.file_id = rf.id
            WHERE rf.relative_path IN ({placeholders})
            GROUP BY rf.id
            ORDER BY rf.relative_path
            """,
            expected_paths,
        ).fetchall()


def count_rows(table: str) -> int:
    """Count rows in a known benchmark table."""
    if table not in {"chunks", "chunk_embeddings"}:
        raise ValueError(f"Unsupported table for benchmark count: {table}")
    with db_session() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def peak_rss_mb() -> float:
    """Return peak resident set size in MB for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports bytes. The Linux path is
    # what matters in the current dev/runtime environment.
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1_000_000
    return usage.ru_maxrss / 1024


def stage(name: str) -> None:
    print(f"[cognix-eval] {name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
