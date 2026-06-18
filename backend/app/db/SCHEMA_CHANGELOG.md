# Cognix Schema Changelog

## 20260617_0001_cognix_v2_foundation

Adds the durable database layer required for Cognix v2 proactive intelligence.

New tables:

- `schema_migrations` records applied migrations.
- `claims` stores factual/definition/temporal/etc. claims extracted from chunks.
- `concept_mentions` stores concept occurrences used by gap detection and graph construction.
- `intelligence_findings` stores proactive findings before they are rendered to wiki/UI.
- `graph_edges` stores lightweight concept graph relationships.
- `confidence_scores` stores evidence-confidence scores and full JSON breakdowns for answers.
- `briefings` stores generated Intelligence Brief metadata.
- `intelligence_runs` stores manual/nightly intelligence job execution records.

The migration is idempotent and non-destructive. It does not rewrite v1 tables.

## 20260617_0002_embedding_metadata

Adds provider metadata to `chunk_embeddings`:

- `provider` records the embedding backend, such as `local` or `openai`.
- `embedding_source` records whether the vector belongs to a document chunk, claim, or another future index type.

This keeps Cognix from silently mixing incompatible embedding spaces.

## 20260617_0003_evaluation_and_calibration

Adds the evaluation and future-training substrate:

- `model_predictions` stores task-level predictions, confidence scores, and model metadata.
- `prediction_outcomes` stores reviewed or eval-provided ground truth labels for those predictions.
- `training_examples` stores examples that can later be exported for supervised fine-tuning or evaluation.

These tables let Cognix measure whether confidence scores and model outputs are actually correct.

## 20260617_0004_model_artifacts

Adds `model_artifacts`, a registry for planned and trained local model artifacts such as LoRA adapters.

This lets Cognix track what base model an adapter belongs to, where it is stored, whether it is only planned or actually trained, and what training manifest produced it.

## 20260617_0005_calibration_models

Adds `calibration_models`, a persisted probability-calibration table.

Each row stores a fitted calibrator for a task:

- method name
- learned parameters
- number of training examples
- positive and negative example counts
- Brier score
- log loss
- active/superseded status

The current calibrator is a one-feature logistic model that maps raw answer-confidence scores to estimated correctness probability.

## 20260617_0006_extraction_artifacts

Adds `extraction_artifacts`, an audit table for parser/OCR/vision outputs.

Each row records:

- file/document ids
- extraction artifact type
- extraction method
- confidence score
- page count
- extracted text length
- warnings
- method-specific metadata

This lets Cognix distinguish high-confidence text extraction from weak OCR, missing OCR tools, scanned-PDF fallbacks, and future cloud vision outputs.
