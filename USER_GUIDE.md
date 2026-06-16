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
ollama --version
```

If `tesseract` is missing, Cognix still tracks images but stores OCR status text instead of extracted image text.

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

Do not commit keys.

### Step 6: GitHub Safety Rules

Do not commit local runtime data:

```text
data/raw/
data/library.sqlite
data/processed/
data/chroma/
data/logs/
.env
```

The included `.gitignore` protects the database, caches, logs, dependencies, and `.env` files.

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
- Binary build artifacts are not useful and can be ignored.

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
- If OCR is not installed or finds no text, Cognix stores a clear placeholder.

Optional companion note:

If the image contains important text, create a companion note:

```text
data/raw/images/receipt-2026-06-16.jpg
data/raw/notes/receipt-2026-06-16.md
```

In the note, type or paste the important visible text.

### Step 6: If It Is A PDF

Put it under:

```text
data/raw/documents/
```

Current behavior:

- Selectable-text PDFs are parsed.
- Scanned/image-only PDFs are tracked clearly; extractable PDFs are parsed directly.

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

### Code

Put exported code folders, cloned repos, or zip-extracted repositories here:

```text
data/raw/code/
```

Supported now:

- common text/code files are parsed
- source files are parsed for retrieval
- Git metadata remains available in the raw repository folder

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

## Shipping On GitHub

Recommended repository behavior:

- Commit source code, docs, config examples, and empty `.gitkeep` folders.
- Do not commit local research/source files in `data/raw/` unless you intentionally want them in the repository.
- Do not commit `data/library.sqlite`.
- Do not commit API keys or `.env` files.
- Do not commit generated private outputs unless intentionally sharing them.

The included `.gitignore` already protects the local database, processed cache, Chroma data, logs, virtual environment, and frontend dependencies.
