# Cognix

Cognix is a local-first Personal Neural Library. It ingests raw files, keeps originals untouched, extracts and chunks text, stores operational state in SQLite, indexes chunks for retrieval, answers questions through a light web UI, saves answers as draft markdown outputs, and tracks library health.

The project is intentionally built as understandable code instead of hiding the core workflow behind a heavy AI framework.

## What Works In Version 1

- Local FastAPI backend.
- React + TypeScript web UI.
- Immutable `data/raw/` source vault.
- SQLite catalog for files, chunks, jobs, outputs, errors, and health findings.
- Text, markdown, JSON, CSV, HTML, and PDF text extraction.
- Placeholder handling for image OCR/vision-ready files.
- Chunking and local deterministic embeddings.
- Local semantic/keyword retrieval.
- Question answering from the web UI.
- Draft analysis files saved into `wiki/outputs/analysis/`.
- Source summary markdown pages in `wiki/sources/`.
- Health checks and markdown health reports.
- VS Code / code-oss workspace recommendations.

## Why Deterministic Local Synthesis First

The architecture is hybrid local/cloud, but this first implementation works without API keys. The `backend/app/services/llm.py` service currently creates source-grounded draft answers from retrieved evidence. Cloud or local model providers can replace that service while preserving the same retrieval, citation, and output flow.

## Project Layout

```text
backend/       FastAPI backend
frontend/      React/Vite web UI
config/        source, model, budget, and priority config
data/raw/      immutable source files
data/processed extracted text cache
data/chroma/   vector-store location
wiki/          markdown knowledge layer
.vscode/       code-oss workspace integration
```

## Quick Start

Run Cognix from the repo root:

```bash
scripts/start_cognix.sh
```

Then open:

```text
http://127.0.0.1:5173/
```

See [USER_GUIDE.md](USER_GUIDE.md) for full setup, dependency, ingest, provider, and usage instructions.

For controls, raw folder placement, and current feature coverage, see [USER_GUIDE.md](USER_GUIDE.md).

## Blueprint

The full design record is in [ARCHITECTURE_BLUEPRINT.md](ARCHITECTURE_BLUEPRINT.md).
