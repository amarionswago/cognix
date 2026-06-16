# Personal Neural Library - Architecture Blueprint

Date finalized: 2026-06-16

## 1. Project Vision

The Personal Neural Library is a local-first AI knowledge system that ingests raw personal data, preserves the originals, extracts and indexes usable content, compiles a structured markdown wiki, answers questions through a local web UI, saves generated outputs back into the wiki, and continuously audits the library for quality.

The system is designed as a broad personal mirror across documents, code, AI chats, browser history, finance, health, social data, location data, media history, education, and creative work.

The main user-facing surface is a light-themed local web UI. code-oss / VS Code remains the workspace for inspecting, editing, and previewing generated markdown, diagrams, slides, and database files.

## 2. Core Principles

- Raw files are immutable. The `raw/` directory is the source vault and is never modified by the system.
- Derived artifacts are reproducible. Extracted text, OCR output, embeddings, wiki pages, summaries, charts, and reports can be regenerated from raw sources.
- Local-first by default. Version 1 stores project data on the local machine only.
- Sensitive data is labeled rather than excluded.
- Cloud model calls are routed through sensitivity and budget policies.
- Markdown remains the durable human-readable knowledge layer.
- SQLite tracks operational state.
- ChromaDB stores vector embeddings for semantic retrieval.
- The local web UI is the primary daily interface.
- Every factual claim should be traceable to source evidence where practical.

## 3. Final Decision Log

### Domain 1 - Raw Data

- Include all major data categories: articles, PDFs, notes, screenshots, photos, code repos, AI chats, email, browser data, finance, health, social media, location, media history, education, creative work, legal/identity documents, and other personal files.
- Starting volume is medium: 1-20 GB and thousands of files.
- Design the codebase for much larger daily growth.
- Keep `raw/` immutable.
- Exclude no data category by default.
- Use OCR plus vision for images, screenshots, scanned PDFs, forms, diagrams, and visual documents.
- Support manual batch dumps, frequent manual additions, automatic watchers, and scheduled imports.

### Domain 2 - Ingest Pipeline

- Support manual ingest, scheduled jobs, and file watchers.
- Implement staged automation: explicit ingest first, scheduler second, watcher third.
- Use custom folder priority for processing.
- Use configurable ingest behavior by folder.
- Use SQLite as the main ingest catalog.
- Skip failed files and log failures instead of halting the full run.
- Allow limited retry for transient failures such as timeouts or locked files.

### Domain 3 - Compilation Engine

- Use hybrid wiki organization: domain pages, source summaries, timelines, indexes, outputs, contradictions, and decisions.
- Article types:
  - `concept`
  - `person`
  - `project`
  - `source_summary`
  - `timeline`
  - `analysis`
  - `dataset`
  - `decision`
  - `contradiction`
  - `index`
- Use controlled LLM autonomy:
  - auto-write low-risk pages
  - require review for sensitive or high-impact edits
- Use strict citations.
- Use slug filenames with readable titles in frontmatter.

### Domain 4 - Knowledge Architecture

- Use Markdown + SQLite + ChromaDB.
- Add vector search from version 1 for extracted chunks.
- Use semi-structured relationships:

```yaml
relationships:
  related_to:
    - sleep-quality
  part_of:
    - health-patterns
```

- Support keyword search, semantic search, metadata filters, graph traversal, and timeline search.
- Use standard temporal metadata:
  - source date
  - ingest date
  - observed date
  - valid-from / valid-to when known

### Domain 5 - Q&A System

- Use a strict local web UI as the Q&A interface.
- CLI commands may exist internally for development and automation, but they are not the primary user workflow.
- Output style is configurable, defaulting to a cited research memo.
- Use web search automatically for public/non-sensitive enrichment.
- Ask permission before using web search for sensitive or personal questions.
- Show final answer, sources, and retrieval summary.
- Save generated answers automatically as draft `analysis` pages.

### Domain 6 - Output And Rendering

- Version 1 output formats:
  - markdown research memos
  - markdown wiki articles
  - markdown tables
  - CSV tables
  - Mermaid diagrams
  - PNG charts
  - Marp slide decks
- Draft outputs appear in a web UI review screen.
- Use standard VS Code integration:
  - recommended extensions
  - workspace settings
  - tasks for preview/build workflows
- Track outputs in both files and SQLite.
- Use a light, clean, modern web UI with tasteful "neural library" identity.

### Domain 7 - Linting And Self-Improvement

- Auto-fix deterministic issues only:
  - frontmatter normalization
  - index updates
  - formatting
  - mechanical link repairs
- Semantic fixes require review.
- Run health checks:
  - after compile
  - nightly
  - on demand from the web UI
- Store findings in both markdown reports and the web UI.
- Use web imputation automatically for public/non-sensitive facts.
- Ask permission for sensitive/personal imputation.
- Show a visible Library Health Score.

### Domain 8 - Tech Stack

- Backend: Python.
- Web framework: FastAPI.
- Frontend: React + TypeScript + Vite.
- Main database: SQLite.
- Vector store: ChromaDB.
- OCR/Vision: local OCR plus cloud vision.
- LLM strategy: hybrid local + cloud.
- Avoid LangChain/LlamaIndex in the core version.
- Keep orchestration explicit and beginner-readable.

### Domain 9 - Scalability And Growth

- Design version 1 to comfortably handle about 100k chunks.
- Use a SQLite-backed job queue.
- Reprocess selectively based on parser, prompt, OCR, embedding, or model version changes.
- Add fine-tuning only after the wiki has high-quality curated data.
- Use budget limits by task, source, and provider.

### Domain 10 - Specific Data Sources

- Support all listed personal source families:
  - social
  - communication
  - health
  - finance
  - code
  - AI interactions
  - documents
  - location
  - browser
  - media
  - education
  - creative work
- Local-only by default:
  - finance
  - health
  - legal/identity documents
  - private messages/email
  - location history
  - credentials/secrets
- First real imports:
  - AI chat exports
  - local code repos
  - PDFs/articles/markdown documents
  - browser bookmarks/history
  - one structured dataset such as finance or health
- Enable secrets detection during ingest.
- Store version 1 data locally only.

### Domain 11 - Architecture

- Adopt the architecture diagram in this blueprint.
- Use the local web UI as the primary operating surface.
- Use code-oss / VS Code as the editor, preview, and workspace surface.
- Use files for durable knowledge and SQLite/ChromaDB for operational/search state.

### Domain 12 - Minimal Codebase

- Version 1 implements the full loop, not every future source adapter.
- Use FastAPI backend + React/Vite frontend + SQLite + ChromaDB + markdown wiki.
- Keep code modular without heavy AI frameworks.
- Build in staged order:
  1. scaffold
  2. database
  3. ingest
  4. embeddings
  5. web UI shell
  6. Q&A
  7. output saving
  8. compiler
  9. health/lint
  10. adapters
  11. VS Code integration

## 4. Complete Architecture Diagram

```text
                         PERSONAL NEURAL LIBRARY

+--------------------------------------------------------------------+
|                         DATA SOURCES                               |
| articles | PDFs | notes | screenshots | docs | code | AI chats      |
| email | browser | finance | health | social | location | media      |
| education | creative work                                           |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| raw/ - IMMUTABLE LOCAL SOURCE VAULT                                |
| Originals are never edited. Everything derived is regenerated from  |
| source files when parsers, prompts, OCR, or models improve.         |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| INGEST TRIGGERS                                                    |
| Web UI manual import | scheduled jobs | file watcher                |
| Internal developer command may exist, but Web UI is primary.         |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| INGEST PIPELINE                                                    |
| discover -> hash -> classify -> detect secrets -> parse -> OCR      |
| -> queue vision -> chunk -> embed -> log success/failure            |
+--------------------------------------------------------------------+
                 |                              |
                 v                              v
+-------------------------------+    +-------------------------------+
| SQLite                         |    | ChromaDB                       |
| files, versions, chunks, jobs,  |    | vector embeddings for          |
| errors, outputs, health, costs, |    | semantic search over chunks    |
| parser/model/prompt versions    |    |                               |
+-------------------------------+    +-------------------------------+
                 |                              |
                 +--------------+---------------+
                                |
                                v
+--------------------------------------------------------------------+
| COMPILATION ENGINE                                                 |
| reads changed chunks -> writes source summaries -> identifies       |
| concepts -> updates wiki pages -> records contradictions -> updates |
| indexes -> routes sensitive/high-impact edits to review             |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| wiki/ - MARKDOWN KNOWLEDGE LAYER                                   |
| domain pages | source summaries | concepts | people | projects      |
| timelines | analyses | datasets | decisions | contradictions        |
| outputs | indexes | health reports                                  |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| SEARCH + Q&A ORCHESTRATOR                                          |
| query understanding -> keyword search -> semantic search -> filters |
| -> graph traversal -> timeline search -> evidence pack -> compute   |
| if needed -> LLM synthesis -> citation verification                 |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| LOCAL WEB UI                                                       |
| ask questions | review sources | view retrieval summaries | manage  |
| outputs | approve sensitive web/cloud use | health dashboard | jobs |
| budget controls | draft review                                      |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| OUTPUTS + FEEDBACK LOOP                                            |
| markdown memos | wiki articles | CSV/tables | Mermaid diagrams      |
| PNG charts | Marp slides                                            |
| saved to wiki/outputs/ and tracked in SQLite as draft artifacts      |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
| HEALTH/LINT ENGINE                                                 |
| deterministic auto-fixes | semantic review findings | broken links  |
| citation gaps | stale claims | duplicates | contradictions | score  |
+--------------------------------------------------------------------+
```

## 5. End-To-End Data Flow

Example: adding a ChatGPT export.

1. The export is placed in `raw/ai_chats/`.
2. The web UI import action, scheduler, or watcher notices it.
3. The ingest pipeline computes a SHA-256 content hash.
4. SQLite checks whether this exact version was already processed.
5. The source adapter recognizes the ChatGPT export format.
6. Messages, timestamps, titles, and conversation text are extracted.
7. Secrets and sensitivity labels are applied.
8. Conversations are chunked into searchable units.
9. Chunks are saved in SQLite.
10. Chunk embeddings are stored in ChromaDB.
11. The compiler creates source summaries and concept candidates.
12. Relevant wiki pages are created or updated.
13. Indexes and backlinks are refreshed.
14. Health checks verify structure, links, citations, and embeddings.
15. The data is now available to Q&A.

## 6. End-To-End Question Flow

Example: asking "When do I do my best creative work?"

1. The question is typed into the local web UI.
2. The backend classifies it as analytical and personal.
3. It builds a retrieval plan across creative work, code, health, browser, AI chats, and temporal metadata.
4. It runs semantic search in ChromaDB.
5. It runs keyword and metadata searches in SQLite and wiki files.
6. It expands through backlinks and relationship metadata.
7. It builds an evidence pack with chunks, dates, sources, and sensitivity labels.
8. If computation is needed, Python/SQLite calculates trends.
9. The LLM synthesizes a cited research memo.
10. The system checks that important factual claims cite sources.
11. The answer appears in the web UI with sources and retrieval summary.
12. The answer is saved as a draft `analysis` page in `wiki/outputs/analysis/`.
13. The output becomes part of future retrieval and compilation.

## 7. Directory Structure

```text
cognix/
  README.md
  GETTING_STARTED.md
  ARCHITECTURE_BLUEPRINT.md

  backend/
    pyproject.toml
    app/
      main.py
      config.py
      database.py

      models/
        files.py
        chunks.py
        jobs.py
        outputs.py
        health.py

      services/
        ingest.py
        parsers.py
        chunking.py
        embeddings.py
        llm.py
        retrieval.py
        compiler.py
        outputs.py
        health.py
        jobs.py
        security.py

      api/
        ingest.py
        ask.py
        outputs.py
        health.py
        jobs.py
        settings.py

  frontend/
    package.json
    index.html
    src/
      App.tsx
      main.tsx
      api/
      components/
      pages/
      styles/

  data/
    raw/
    processed/
    chroma/
    library.sqlite
    logs/

  wiki/
    personal/
    work/
    code/
    health/
    finance/
    learning/
    media/
    people/
    places/
    concepts/
    sources/
    outputs/
      analysis/
      reports/
      slides/
      charts/
      diagrams/
      tables/
    contradictions/
    decisions/
    datasets/
    timelines/
    _indexes/
    _health/

  config/
    sources.yml
    models.yml
    budgets.yml
    priorities.yml

  .vscode/
    extensions.json
    settings.json
    tasks.json
```

## 8. Main Data Stores

### raw/

Immutable source vault. The system reads from this directory but does not rewrite, normalize, rename, or delete original files.

### data/processed/

Derived extracted text, OCR output, intermediate artifacts, and cached processing results.

### SQLite

Operational database for:

- raw files
- file versions
- extracted documents
- chunks
- ingest jobs
- ingest errors
- outputs
- health findings
- source metadata
- temporal metadata
- costs and provider usage
- parser/prompt/model versions

### ChromaDB

Persistent vector store for embeddings over extracted chunks. Used for semantic retrieval during Q&A and compilation.

### wiki/

Markdown knowledge layer containing domain pages, source summaries, concepts, outputs, indexes, decisions, contradictions, datasets, timelines, and health reports.

## 9. Article Template

```markdown
---
title: Example Concept
slug: example-concept
type: concept
status: active
created: 2026-06-16
updated: 2026-06-16
sensitivity: personal
tags: []
source_date:
observed_at:
valid_from:
valid_to:
sources: []
relationships:
  related_to: []
  part_of: []
---

# Example Concept

## Summary

Short human-readable explanation.

## Key Facts

- Source-backed fact.

## Details

Longer synthesized explanation.

## Related

- [[related-page]]

## Contradictions / Uncertainty

- Conflicting or weak evidence.

## Open Questions

- Missing information to investigate later.
```

## 10. Sensitivity Model

Recommended labels:

- `public`: public articles, public repos, public posts.
- `personal`: notes, AI chats, browser data, education, creative work.
- `sensitive`: finance, health, email, private chats, legal documents, identity documents, location.
- `secret`: credentials, API keys, passwords, tokens, private keys, identity numbers.

Default behavior:

- `public`: may use cloud models or web enrichment according to budget settings.
- `personal`: may use cloud models only if allowed by source policy.
- `sensitive`: local-only by default; ask permission before cloud/web use.
- `secret`: never sent to cloud models by default; use local-only handling and review.

## 11. Search And Retrieval Modes

The Q&A orchestrator supports:

- keyword search for exact terms
- semantic search through ChromaDB embeddings
- metadata filters by date, source, sensitivity, and type
- graph traversal through links and frontmatter relationships
- timeline search using source/ingest/observed/valid dates

Retrieval should produce an evidence pack that includes:

- chunk IDs
- source paths
- dates
- excerpts
- relevance scores
- sensitivity labels
- retrieval method

## 12. Output System

Version 1 generates:

- markdown research memos
- markdown wiki articles
- markdown tables
- CSV tables
- Mermaid diagrams
- PNG charts
- Marp slide decks

Outputs are saved to:

```text
wiki/outputs/
  analysis/
  reports/
  slides/
  charts/
  diagrams/
  tables/
```

Outputs are also tracked in SQLite with:

- output ID
- path
- type
- status
- query
- sources used
- created date
- sensitivity
- review state

## 13. Health And Lint Checks

Deterministic checks:

- missing frontmatter
- invalid article type
- missing title
- slug/title mismatch
- missing status
- invalid sensitivity label
- broken wiki links
- orphan articles
- duplicate slugs
- indexes missing pages
- source references pointing to missing files/chunks
- files ingested but never compiled
- chunks with no embeddings
- failed ingest jobs needing review

Semantic checks:

- duplicate concept pages
- contradictions
- stale claims
- unsupported claims
- vague summaries
- concepts mentioned often but missing pages
- low-quality generated analyses

Health outputs:

- web UI dashboard
- markdown reports in `wiki/_health/`
- SQLite health findings
- visible Library Health Score

## 14. Tech Stack

Backend:

- Python
- FastAPI
- SQLite
- ChromaDB
- local OCR
- cloud vision
- hybrid local/cloud LLM providers

Frontend:

- React
- TypeScript
- Vite
- light modern UI
- review screens
- health dashboard
- job status views
- settings and budget controls

Core philosophy:

- avoid heavy LangChain/LlamaIndex dependency in version 1
- keep orchestration readable
- use explicit service modules
- keep prompts and provider routing inspectable

VS Code / code-oss:

- editor and preview workspace
- Markdown Preview
- Marp slide preview
- Mermaid rendering
- YAML support
- SQLite inspection
- workspace settings and recommended extensions

## 15. Minimal Viable Build

Version 1 must implement the complete loop:

1. Accept files in `data/raw/`.
2. Track them in SQLite.
3. Extract text from common text, markdown, and PDF files.
4. Provide hooks for OCR and image processing.
5. Chunk extracted text.
6. Embed chunks into ChromaDB.
7. Generate basic markdown source summaries/wiki pages.
8. Ask questions from the web UI.
9. Retrieve relevant chunks.
10. Produce cited markdown answers.
11. Save answers as draft outputs.
12. Show jobs, errors, outputs, and health status in the UI.

Version 1 does not need every future source adapter to be complete.

## 16. Implementation Order

1. Project scaffolding
   - create directories
   - create config files
   - create README and setup docs

2. SQLite schema
   - files
   - chunks
   - jobs
   - outputs
   - health findings
   - model/provider usage

3. Ingest core
   - discover files
   - hash files
   - classify folders
   - parse text/markdown/PDF
   - log failures

4. Chunking and embeddings
   - split extracted text
   - store chunks
   - embed chunks
   - persist vectors in ChromaDB

5. Web UI shell
   - light theme
   - navigation
   - status cards
   - API connectivity

6. Ingest UI
   - trigger ingest
   - view files
   - view jobs
   - view errors

7. Q&A core
   - ask question
   - retrieve chunks
   - synthesize answer
   - cite sources

8. Output saving
   - save answers into `wiki/outputs/analysis/`
   - record outputs in SQLite
   - mark as draft

9. Output review UI
   - promote
   - archive
   - delete
   - inspect sources

10. Compilation v1
    - source summaries
    - basic concept pages
    - index updates

11. Health/lint v1
    - frontmatter checks
    - broken links
    - orphan pages
    - failed jobs
    - missing embeddings

12. VS Code integration
    - recommended extensions
    - settings
    - tasks
    - markdown preview support

13. First source adapters
    - AI chat exports
    - local code repos
    - PDFs/articles/markdown
    - browser data
    - one structured dataset such as finance or health

## 17. Open Questions

All major brainstorm decisions are resolved.

Details to decide during implementation:

- exact local OCR engine
- exact cloud LLM and vision providers
- exact embedding model
- exact first structured dataset: finance or health
- exact first browser source: bookmarks, history, or saved pages
- exact web UI visual system

These are implementation choices, not architecture blockers.
