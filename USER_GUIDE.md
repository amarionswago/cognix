# Cognix User Guide

This guide explains how to use the Cognix web UI, where to place data, and what each main control does.

Everything needed to install, run, and use Cognix is included here.

## First-Clone Setup Protocol

Use this when you just cloned Cognix from GitHub.

### Step 1: Clone The Repository

```bash
git clone <repo-url> cognix
cd cognix
```

Replace `<repo-url>` with the actual GitHub URL.

### Step 2: Install System Requirements

You need:

```text
Python 3.11+
Node.js 20+
npm
```

Recommended:

```text
tesseract        local OCR for images/screenshots/scans
pdftoppm         local scanned-PDF page rendering, usually from poppler-utils
ollama           optional local LLM provider
git              clone and manage repos
```

Check:

```bash
python --version
node --version
npm --version
```

Check optional tools:

```bash
tesseract --version
pdftoppm -v
ollama --version
```

If `tesseract` is missing, Cognix still tracks images but stores OCR status text instead of extracted image text. If `pdftoppm` is missing, selectable PDFs still work, but scanned/image-only PDFs cannot be locally OCRed.

If `ollama` is missing, only cloud providers and the local deterministic fallback are available.

### Step 3: Start Cognix With One Command

```bash
scripts/start_cognix.sh
```

That one command:

- creates `.venv` if missing
- installs/updates backend dependencies
- installs frontend dependencies if missing
- creates runtime folders
- starts the backend
- starts the web UI

Web UI:

```text
http://127.0.0.1:5173/
```

Backend:

```text
http://127.0.0.1:8000
```

Press `Ctrl+C` in that terminal to stop both services.

### Step 3.1: Run Product Evaluation Before Shipping

Before a customer-facing release, run the baseline checks:

```bash
.venv/bin/python -m pytest -q backend/tests
npm --prefix frontend run build
.venv/bin/python backend/evals/run_evals.py
.venv/bin/python backend/evals/run_evals.py --suite large --report data/eval-large-report.json
```

What these prove:

- backend unit tests still pass
- the frontend still builds
- a temporary labeled library can ingest files
- retrieval finds the expected HTML, PDF, and markdown sources
- image and scanned-PDF inputs produce auditable OCR extraction artifact rows
- the persisted logistic confidence calibrator can train and apply from labeled eval outcomes
- the intelligence pass creates findings and a saved brief
- the large suite checks larger ingest/retrieval behavior without running the full intelligence audit
- the large suite writes an optional JSON report with file-family metrics

What they do **not** prove yet:

- every huge real-world file will ingest perfectly
- every answer is correct
- ChromaDB performance is fully benchmarked under every customer machine profile
- OCR/vision quality is production-complete

Treat failures here as release blockers.

The large benchmark currently measures:

- generated large-file bytes
- supported file coverage
- failed large-file count
- OCR/vision artifact coverage
- calibrated probability model training/application
- chunk count
- chunk/embedding parity
- ingest elapsed time
- ingest throughput in MB/s
- peak RSS memory
- per-family retrieval latency
- per-family recall@10

The current large fixture families are HTML, CSV, JSON, Python/code, log/text, markdown, text-recoverable PDF fallback, and email export.

### Optional Local ML Backends

Cognix works without heavy ML packages, but stronger local ML can be enabled on capable machines.

Install optional ML dependencies:

```bash
.venv/bin/python -m pip install -e "backend[ml]"
```

Enable local neural embeddings:

```bash
export COGNIX_LOCAL_EMBEDDING_BACKEND=sentence-transformers
export COGNIX_SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Enable trained cross-encoder reranking:

```bash
export COGNIX_RERANKER_BACKEND=cross-encoder
export COGNIX_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Enable trained NLI contradiction classification:

```bash
export COGNIX_NLI_BACKEND=cross-encoder
export COGNIX_NLI_MODEL=cross-encoder/nli-deberta-v3-small
```

If those models are unavailable, Cognix falls back to deterministic local behavior instead of failing startup.

Train Cognix's local pair reranker without external model downloads:

```bash
.venv/bin/python backend/training/train_pair_model.py --task reranker --register-artifact
export COGNIX_RERANKER_BACKEND=cognix-pair
```

Train Cognix's local pair NLI model without external model downloads:

```bash
.venv/bin/python backend/training/train_pair_model.py --task nli --register-artifact
export COGNIX_NLI_BACKEND=cognix-pair
```

The local pair models are small trained neural baselines over text pairs. They are not transformer cross-encoders, but they give Cognix its own trainable reranker and contradiction-classifier artifacts under `data/models/`.

Train all local Cognix baseline model artifacts at once:

```bash
.venv/bin/python backend/training/bootstrap_local_models.py --register-artifacts
```

This writes:

- `data/models/cognix-reranker-pair.json`
- `data/models/cognix-reranker-cross-encoder.json`
- `data/models/cognix-nli-pair.json`
- `data/models/cognix-nli-cross-encoder.json`
- `data/models/cognix-micro-synthesis.json`

Then enable the local trained paths you want:

```bash
export COGNIX_RERANKER_BACKEND=cognix-cross-encoder
export COGNIX_NLI_BACKEND=cognix-cross-encoder
export COGNIX_SYNTHESIS_BACKEND=cognix-micro
```

Use `cognix-pair` instead of `cognix-cross-encoder` if you want the smaller hashed pair MLP. The cross-encoder artifacts read both texts together and are the stronger local trained baseline, but they are still not transformer-class models.

Train Cognix's local micro-synthesis model from reviewed Q&A examples:

```bash
.venv/bin/python backend/training/train_cognix_micro_model.py \
  --dataset data/exports/training/qa_citation-reviewed-sft.jsonl \
  --register-artifact
export COGNIX_SYNTHESIS_BACKEND=cognix-micro
```

This model learns Cognix's answer structure, evidence-section style, citation habits, and lightweight term salience from reviewed SFT examples. It still answers from retrieved source chunks. It is not a foundation model or a LoRA adapter, but it is a real local trained Cognix synthesis artifact under `data/models/`.

Enable OpenAI vision for image extraction:

```bash
export COGNIX_VISION_BACKEND=openai
export COGNIX_OPENAI_VISION_MODEL=gpt-4.1-mini
```

Privacy note: local OCR keeps image/PDF processing on your machine. OpenAI vision sends image bytes to the configured OpenAI account so the model can extract visible text and visual context.

### Fine-Tuning Readiness

Cognix can export reviewed Q&A examples and validate a LoRA fine-tuning run.

Export reviewed examples to SFT JSONL from Python:

```bash
.venv/bin/python -c "from app.services.training_export import export_sft_jsonl; print(export_sft_jsonl('qa_citation', 'reviewed'))" 
```

Validate a future LoRA run without training:

```bash
.venv/bin/python backend/training/train_lora.py \
  --dataset data/exports/training/qa_citation-reviewed-sft.jsonl \
  --output-dir data/models/cognix-lora \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --adapter-name cognix-lora \
  --dry-run
```

Install optional training dependencies only on a machine intended for training:

```bash
.venv/bin/python -m pip install -e "backend[training]"
```

Run the same command without `--dry-run` to start an actual LoRA training job.

Important: Cognix now has the fine-tuning pipeline, dataset validator, manifest writer, and model artifact registry. The repo does not ship with a pre-trained Cognix adapter yet.

### Confidence Calibration

Cognix starts with a heuristic answer-confidence score. After enough answers or eval predictions have reviewed outcomes, it can train a small probability model that maps raw confidence to observed correctness.

Train the calibrator from reviewed/eval outcomes:

```bash
.venv/bin/python backend/training/train_calibrator.py --task answer_confidence
```

This requires at least 20 reviewed/eval outcomes for `answer_confidence`, including both correct and incorrect examples. After training, Cognix saves the calibrator in SQLite and future answer confidence uses the persisted probability model.

The product eval also checks this path automatically with synthetic labeled eval outcomes. That proves the calibration mechanism works. Real production calibration should be trained from reviewed customer/library outcomes, not from the synthetic eval labels.

After a real adapter exists, Cognix can generate a local model package manifest and Ollama Modelfile from Python:

```bash
.venv/bin/python -c "from pathlib import Path; from app.services.finetuning import write_ollama_modelfile, build_model_package_manifest; p=write_ollama_modelfile('llama3.2', Path('data/models/cognix-lora'), Path('data/models/cognix-lora/Modelfile')); print(build_model_package_manifest('cognix-lora', 'llama3.2', Path('data/models/cognix-lora'), p)['load_command'])"
```

That prints the command to create the named local model, for example:

```bash
ollama create cognix-lora -f data/models/cognix-lora/Modelfile
```

### Advanced ML Readiness Check

Cognix exposes a readiness endpoint for the advanced ML features:

```text
http://127.0.0.1:8000/api/ml/readiness
```

It reports whether these are actually available on the current machine:

- local neural embeddings
- neural reranker
- trained NLI contradiction model
- LoRA training stack
- custom Cognix adapter artifact
- OCR/vision pipeline
- calibrated confidence
- large-file benchmark suite

States:

- `ready`: usable now
- `configured`: settings or records exist, but runtime proof is not complete
- `fallback`: Cognix is using the deterministic fallback
- `missing`: required dependency, model, key, or artifact is absent

### Step 4: Runtime Folders

The repo includes placeholders, but this command is safe:

```bash
mkdir -p data/raw data/processed data/chroma data/logs wiki/outputs/analysis wiki/_health wiki/_indexes
```

### Step 5: Optional Model Provider Keys

Cognix works without API keys using local deterministic synthesis, but model providers can improve answers.

Set keys in your shell before starting the backend:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Or paste keys later in:

```text
Settings -> Model Providers
```

## What Cognix Does

Cognix is a local-first knowledge library with a working core loop:

1. Put files into `data/raw/`.
2. Run ingest from the web UI.
3. Cognix extracts text, chunks it, stores metadata in SQLite, and indexes chunks for retrieval.
4. Ask questions in the web UI.
5. Cognix retrieves relevant chunks, creates a cited draft answer, and saves it into `wiki/outputs/analysis/`.
6. Review outputs and run health checks.

## Main Web UI Controls

Open the app at:

```text
http://127.0.0.1:5173/
```

### Ask

Use this to ask questions against your ingested library.

Controls:

- Question box: type the question you want Cognix to answer.
- Memo: default research-style answer with evidence.
- Brief: shorter answer.
- Deep: longer answer mode; currently uses the same local retrieval base and is ready for future LLM upgrades.
- Ask Cognix: runs retrieval, generates a draft answer, saves it into the wiki.

Saved answers go to:

```text
wiki/outputs/analysis/
```

### Ingest

Use this after adding files to `data/raw/`.

Controls:

- Run ingest: scans `data/raw/`, processes new/changed files, writes chunks and metadata.
- Recent Files: shows files Cognix has seen and their status.
- Recent Jobs: shows ingest jobs and whether they completed.
- Ingest Errors: shows files that failed and why.

Raw files are not edited.

### Outputs

Use this to manage generated analysis drafts.

Controls:

- Refresh: reload output records from SQLite.
- Check icon: mark an output as promoted.
- Archive icon: mark an output as archived.
- Trash icon: mark the output record as deleted.

The current trash action updates the SQLite record status. It does not delete the markdown file from disk.

### Intelligence

Use this to run Cognix as a proactive knowledge auditor.

Controls:

- Run intelligence: runs the offline/no-cost intelligence pass by default.
- Knowledge gaps: concepts Cognix sees repeatedly but cannot find as structured wiki pages.
- Contradictions: confirmed or candidate conflicts between extracted claims.
- Resolve: marks a contradiction finding as resolved after you review it.
- Concept Graph Pulse: quick view of frequent concepts from the current gap findings.
- Intelligence Brief: the latest generated markdown brief.

The intelligence pass currently performs:

- concept extraction,
- knowledge gap detection,
- claim extraction,
- contradiction candidate detection,
- stale claim candidate detection,
- Intelligence Brief generation.

Generated briefs are saved here:

```text
wiki/_intelligence/
```

Important:

- The default web UI run avoids cloud LLM calls.
- Provider-backed LLM judgment can be enabled through the API later when you want deeper contradiction confirmation.
- Findings are stored in SQLite first, then rendered into the brief. This keeps the UI and wiki auditable.

### Health

Use this to inspect library quality.

Controls:

- Run health check: checks structure, chunks, errors, and missing required wiki folders.
- Score: quick diagnostic number.
- Files: number of raw file records.
- Chunks: number of indexed chunks.
- Errors: ingest errors.
- Open Findings: current health findings.

Latest health report:

```text
wiki/_health/latest-health-report.md
```

### Settings

Use this to persist local user preferences.

Controls:

- Name: local profile name saved for this Cognix install.
- Theme: switch between White and Dark.
- Default answer style: default for Ask mode.
- Raw data note: optional local note about where this installation stores exports/imports.
- Save profile: persists settings in SQLite.
- Model Providers: configure Local fallback, Ollama, OpenAI, or Anthropic.
- Save: stores provider settings locally for that provider.
- Test connection: saves the current provider settings, then checks whether that provider and selected model actually work.
- Background Work: turn automatic file watching and scheduled ingest on or off.
- Interval seconds: how often the watcher/scheduler should run.

This is not an internet account. Cognix is local-first, so the profile persists on the machine where Cognix runs.

## Model Provider API Keys

Cognix does not ship with any API key.

Each user should connect their own provider key in one of two ways:

1. Set a provider environment variable before starting the backend.
2. Paste their key into **Settings -> Model Providers** in the web UI.

Supported provider environment variables:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

Supported providers:

- Local fallback: built into Cognix. It needs no API key and no model server. It creates grounded draft answers from retrieved evidence, so Cognix still works immediately after cloning.
- Ollama: a local model runtime. It needs the Ollama app/server running on your machine and at least one installed model, but it does not need a cloud API key.
- OpenAI: cloud model provider. It needs a valid OpenAI API key.
- Anthropic: cloud model provider. It needs a valid Anthropic API key.

Local fallback vs Ollama:

- Local fallback is Cognix's internal no-key answer path. It is reliable for basic source-grounded answers, but it is not a full language model.
- Ollama runs an actual local language model on your machine. It can produce richer answers, but it requires installing Ollama, downloading a model, and having enough CPU/RAM/VRAM. Any model that is installed and runnable through Ollama can be used; type its exact model name in the Model field.

Save vs Test:

- Save stores the key or model choice locally.
- Test connection first saves the visible provider settings, then verifies the provider.

Cognix does not treat any random string as a working key. For OpenAI and Anthropic, **Test connection** sends a tiny real request using the selected model. If the key is invalid, the model name is wrong, the key lacks access to that model, the key is exhausted, billing/quota is blocked, or the network is blocked, Cognix marks the exact problem instead of showing Connected. For Ollama, **Test connection** sends a tiny local generation request to the exact model name you entered. If that model runs, it connects; if it is missing, Cognix marks Model missing.

To see local Ollama models outside the UI:

```bash
ollama list
```

If the list is empty, install one first:

```bash
ollama pull llama3.2
```

Then enter that installed model name in **Settings -> Model Providers -> Ollama -> Model** and click **Test**.

The UI shows whether a provider is:

- Not connected: no key is available.
- Configured: a key exists but has not been verified yet.
- Connected: Cognix successfully reached that provider.
- Exhausted: the provider appears to be out of quota, credits, billing access, or usable balance.
- Rate limited: the provider key is valid, but the provider is temporarily refusing more requests.
- Bad key: the provider rejected the API key.
- Model missing: the selected model is unavailable, misspelled, not installed, or not allowed for that key.
- Failed: Cognix tried to verify the provider, but the provider rejected the request or could not be reached for another reason.

Environment keys are preferred over locally saved keys. Locally saved keys are stored in that user's local SQLite database at `data/library.sqlite`, which is ignored by Git.

If a key was stored only in `data/library.sqlite` and that database is deleted, Cognix cannot recover the key. The user must paste it again or set the environment variable.

Do not commit API keys to GitHub.

## Where To Drop Data

All source files go somewhere under:

```text
data/raw/
```

Yes: copying data into `data/raw/` is the main dumping workflow.

Think of `data/raw/` as the intake vault. You copy or move files, folders, exports, repos, PDFs, images, CSVs, notes, and archives into the right subfolder. Cognix reads from `data/raw/`, extracts useful text/metadata, and writes derived files elsewhere.

Important:

- Cognix does not edit raw files.
- You can dump nested folders.
- You can dump many files at once.
- After dumping, run **Ingest -> Run ingest** in the UI.
- Or enable **Settings -> Background Work -> File watcher** to scan repeatedly.

Recommended folders:

```text
data/raw/articles/
data/raw/documents/
data/raw/notes/
data/raw/images/
data/raw/code/
data/raw/ai_chats/
data/raw/email/
data/raw/browser/
data/raw/finance/
data/raw/health/
data/raw/social/
data/raw/location/
data/raw/media/
data/raw/education/
data/raw/creative/
data/raw/legal/
```

You can create these folders manually. Cognix will scan nested folders.

## Step-By-Step: Add Any File, Repo, Photo, Text, Or Export

Use this when you have "anything at all" and just want to get it into Cognix.

### Step 1: Decide The Closest Raw Folder

Put the item in the closest matching folder:

```text
AI chat export            -> data/raw/ai_chats/
Code repo or code file    -> data/raw/code/
PDF, DOC, document        -> data/raw/documents/
Markdown or text note     -> data/raw/notes/
Photo or screenshot       -> data/raw/images/
Saved article/web page    -> data/raw/articles/
Browser export/bookmarks  -> data/raw/browser/
Bank/transaction export   -> data/raw/finance/
Health export             -> data/raw/health/
Social export             -> data/raw/social/
Location export           -> data/raw/location/
Music/video history       -> data/raw/media/
Course/school material    -> data/raw/education/
Writing/art/music project -> data/raw/creative/
Legal/identity document   -> data/raw/legal/
Not sure                  -> data/raw/documents/
```

The folder does not have to be perfect. It mainly helps Cognix classify the source type.

### Step 2: Keep The Original File Untouched

Do not edit the file after dropping it in `data/raw/`.

Cognix treats `data/raw/` as the permanent source vault. It reads files from there and writes extracted text, chunks, outputs, and wiki pages somewhere else.

### Step 3: If It Is A Folder Or Repo

For a local repo:

```text
data/raw/code/my-project/
```

For a zipped repo:

1. Extract the zip.
2. Put the extracted folder under `data/raw/code/`.
3. Keep the original zip too if you want, but Cognix gets more value from the extracted files.

Example:

```text
data/raw/code/my-app/package.json
data/raw/code/my-app/src/
data/raw/code/my-app/README.md
```

Current behavior:

- Text/code files are parsed.
- Git metadata is preserved in the raw repo folder; source files are parsed for retrieval.
- Dependency folders, cache folders, saved-page asset folders, and binary build artifacts are ignored by default.
- This means folders like `.venv/`, `venv/`, `node_modules/`, `site-packages/`, `__pycache__/`, `dist/`, `build/`, `.pytest_cache/`, `.ruff_cache/`, `.git/`, and `*_files/` stay in `data/raw/` but are not indexed.

Why this matters:

- A small-looking repo can contain thousands of dependency/cache files.
- Indexing those files makes ingest slow and makes Ask/Search return dependency noise instead of your actual project.
- Cognix keeps the raw folder untouched, but it only indexes files that are likely to contain useful knowledge.

### Large Files

Cognix can ingest large files, but large does not mean instant.

What happens during ingest:

- Cognix reads the file and extracts text.
- It splits the extracted text into searchable chunks.
- It embeds and indexes those chunks in batches.
- It writes source summaries and wiki records from the indexed evidence.

Large PDFs, long text files, big exports, large HTML pages, and large logs can all produce hundreds or thousands of chunks. That takes longer than a small note because Cognix is building searchable memory, not only copying the file.

Important limits:

- Text-based PDFs are much faster than scanned/image-only PDFs.
- Photos/screenshots need OCR, which is CPU-heavy and depends on `tesseract`.
- Scanned/image-only PDFs need `pdftoppm` plus `tesseract` for local page rendering and OCR.
- Huge binary files that do not contain useful text are skipped or recorded as unsupported instead of being treated as knowledge.
- Original files always stay untouched in `data/raw/`.

### Step 4: If It Is Raw Text

Create a `.txt` or `.md` file under `data/raw/notes/`.

Example:

```text
data/raw/notes/random-thoughts.md
```

Markdown is recommended because headings help future compilation:

```markdown
# Random Thoughts

Paste or write anything here.
```

### Step 5: If It Is A Photo, Screenshot, Or Scan

Put it under:

```text
data/raw/images/
```

Examples:

```text
data/raw/images/receipt-2026-06-16.jpg
data/raw/images/whiteboard-session.png
data/raw/images/legal-scan.webp
```

Current behavior:

- Cognix tracks the file.
- Cognix runs local OCR if `tesseract` is installed.
- If OpenAI vision is enabled, Cognix can ask the configured OpenAI vision model to extract visible text and visual context.
- Cognix stores OCR method, confidence, page count, text length, warnings, and metadata in SQLite.
- If OCR is not installed or finds no text, Cognix stores a clear placeholder and a low-confidence extraction artifact.

Optional correction note:

If OCR gets important text wrong, create a companion note:

```text
data/raw/images/receipt-2026-06-16.jpg
data/raw/notes/receipt-2026-06-16.md
```

In the note, type or paste the corrected visible text. Cognix will index both the original image extraction and your correction note.

### Step 6: If It Is A PDF

Put it under:

```text
data/raw/documents/
```

Current behavior:

- Selectable-text PDFs are parsed.
- Scanned/image-only PDFs use `pdftoppm` plus `tesseract` when both tools are installed.
- If local scanned-PDF OCR tools are unavailable, Cognix tracks the PDF clearly and stores an OCR-missing warning instead of silently pretending the file was read.
- OCR/PDF extraction metadata is stored in SQLite for audit and future health checks.

If a scanned PDF matters right now, create a companion markdown note with the key text or summary.

### Step 7: If It Is A CSV, JSON, HTML, Email, Or Export

Use the domain folder first:

```text
data/raw/browser/bookmarks.html
data/raw/finance/transactions.csv
data/raw/health/apple-health-export.xml
data/raw/ai_chats/conversations.json
```

Current behavior:

- `.csv`, `.json`, `.html`, `.md`, `.txt`, `.xml`, `.eml`, `.mbox`, logs, and selectable `.pdf` are useful now.
- AI chat JSON, browser HTML, finance CSV, health exports, and email-like files get basic adapter handling.

### Step 8: Run Ingest

Open Cognix:

```text
http://127.0.0.1:5173/
```

Go to:

```text
Ingest -> Run ingest
```

Then check:

- Recent Files
- Recent Jobs
- Ingest Errors

Optional automatic ingest:

```text
Settings -> Background Work
```

Turn on:

- File watcher: repeatedly scans `data/raw/` for new/changed files.
- Scheduled ingest: runs ingest, source summaries, and health checks on an interval.

### Step 9: Ask A Question

After ingest finishes, go to:

```text
Ask
```

Ask something specific first:

```text
What is in the file I just added?
Summarize my latest repo notes.
What does this finance CSV show?
Find anything related to sleep and coding.
```

### Step 10: Review The Output

Generated answers are saved as draft markdown files:

```text
wiki/outputs/analysis/
```

Review them in:

```text
Outputs
```

or open them directly in code-oss / VS Code.

## Quick Examples

### Example: Add A Code Repo

```text
data/raw/code/cognix/
```

Then:

1. Run ingest.
2. Ask: "What does this repo do?"
3. Check `wiki/sources/` for generated source summaries.

### Example: Add A Screenshot

```text
data/raw/images/error-screen.png
```

Optional companion note:

```text
data/raw/notes/error-screen.md
```

Then:

1. Run ingest.
2. Ask about the companion note or image filename.

### Example: Add Raw Thoughts

```text
data/raw/notes/2026-06-16-thoughts.md
```

Then:

1. Run ingest.
2. Ask: "What themes appear in my latest thoughts?"

### Example: Add A Bank CSV

```text
data/raw/finance/june-transactions.csv
```

Then:

1. Run ingest.
2. Ask: "What merchants appear in this transaction file?"

## What If Cognix Cannot Parse It Yet?

Do not delete the file.

If Cognix cannot fully parse a file, it should still either:

- track the file,
- create a placeholder,
- or log an ingest error.

Check:

```text
Ingest -> Ingest Errors
```

Best workaround:

Create a companion markdown note beside the source category explaining what the file is and why it matters.

Example:

```text
data/raw/documents/old-format-file.xyz
data/raw/notes/old-format-file-context.md
```

## What Goes Where

### Articles

Put saved web pages, markdown exports, HTML files, and article PDFs here:

```text
data/raw/articles/
```

Supported now:

- `.md`
- `.txt`
- `.html`
- `.pdf` with selectable text

### Documents And PDFs

Put PDFs, text docs, scans, and exports here:

```text
data/raw/documents/
```

Supported now:

- text extraction from selectable PDFs
- OCR text when local OCR is available
- clear OCR status text when local OCR is unavailable
- extraction artifact rows that record method, confidence, page count, text length, and warnings

### Notes

Put markdown and text notes here:

```text
data/raw/notes/
```

Supported now:

- `.md`
- `.txt`
- `.rst`

### Images And Screenshots

Put screenshots, photos, scanned pages, and image files here:

```text
data/raw/images/
```

Supported now:

- files are detected and tracked
- local OCR runs when `tesseract` is installed
- clear OCR status text is stored when OCR is unavailable
- extraction artifact rows record method, confidence, text length, and warnings

### Code

Put exported code folders, cloned repos, or zip-extracted repositories here:

```text
data/raw/code/
```

Supported now:

- common text/code files are parsed
- source files are parsed for retrieval
- Git metadata remains available in the raw repository folder
- generated dependency/cache/build folders are skipped during ingest so repo dumps stay fast and search stays focused

### AI Chats

Put ChatGPT, Claude, Codex, Gemini, or local model exports here:

```text
data/raw/ai_chats/
```

Supported now:

- `.json`
- `.html`
- `.md`
- `.txt`

### Email And Messages

Put email exports, message exports, and communication logs here:

```text
data/raw/email/
```

Supported now:

- text-like files are parsed
- `.eml` and `.mbox` files are treated as email text exports

### Browser Data

Put bookmarks, history exports, and saved browsing data here:

```text
data/raw/browser/
```

Supported now:

- `.html`
- `.json`
- `.csv`
- `.md`

### Finance

Put statements and transaction exports here:

```text
data/raw/finance/
```

Supported now:

- `.csv` previews
- selectable `.pdf` text

### Health

Put Apple Health, Fitbit, Garmin, Oura, or other health exports here:

```text
data/raw/health/
```

Supported now:

- text-like exports
- `.json`
- `.csv`

### Social, Location, Media, Education, Creative, Legal

Use:

```text
data/raw/social/
data/raw/location/
data/raw/media/
data/raw/education/
data/raw/creative/
data/raw/legal/
```

Text, markdown, JSON, CSV, XML, email exports, HTML, images with local OCR, and selectable PDFs are supported.

## Important Storage Notes

- `data/raw/` is the only source vault.
- `data/processed/` contains derived extracted text.
- `data/library.sqlite` contains operational state.
- `wiki/` contains markdown knowledge and generated outputs.
- Deleting `data/library.sqlite` resets the catalog, but your raw files remain.
- Deleting `wiki/` removes generated knowledge files.

## How To Access And Use The Wiki

The wiki is the markdown knowledge layer Cognix writes beside the web UI.

Open it from the project folder:

```text
wiki/
```

In code-oss / VS Code:

1. Open the Cognix project folder.
2. Expand the `wiki/` folder in the file explorer.
3. Open any `.md` file.
4. Use Markdown Preview to read it as rendered markdown.

Main wiki folders:

```text
wiki/sources/            generated source summaries for ingested files
wiki/_indexes/           generated index pages, including source-index.md
wiki/_health/            health reports
wiki/outputs/analysis/   saved Ask answers and research memos
wiki/outputs/slides/     future slide decks
wiki/outputs/charts/     future charts
wiki/concepts/           future concept articles
wiki/decisions/          future decision records
```

What gets saved there:

- When you run ingest, Cognix extracts text into `data/processed/`, stores searchable chunks in SQLite/Chroma, and writes source summary pages into `wiki/sources/`.
- When you ask a question with saving enabled, Cognix writes the answer as markdown into `wiki/outputs/analysis/`.
- When you run health checks, Cognix writes the latest report into `wiki/_health/latest-health-report.md`.

What to do with the wiki:

- Read generated source summaries in `wiki/sources/`.
- Read saved research answers in `wiki/outputs/analysis/`.
- Use `wiki/_indexes/source-index.md` as a table of contents for ingested sources.
- Edit markdown manually only if you want to. Cognix treats the wiki as generated knowledge, so the normal workflow is to add raw data, ingest it, ask questions, and let Cognix write the markdown.

Important difference:

- `data/raw/` is the permanent source vault.
- `wiki/` is the readable knowledge/output layer.
- `data/library.sqlite` and `data/chroma/` are the searchable index behind the UI.
