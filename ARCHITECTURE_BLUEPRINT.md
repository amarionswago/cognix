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

## Current ML/AI Implementation Status

This section records what is actually implemented in the current Cognix build. It is intentionally strict: Cognix should not claim a trained proprietary model, fine-tuning, or neural graph reasoning until those systems exist.

### Plain-English ML Concept Map

This subsection explains the machine-learning concepts Cognix currently uses or is wired to use. It separates the concept from the implementation so the product does not hide behind vague "neural" language.

#### Embeddings

Plain English: an embedding turns text into a list of numbers. Texts with similar meaning should end up with similar number patterns.

Technical version: Cognix maps each chunk or claim to a vector. The default vectorizer is deterministic hashed bag-of-words, while the optional neural vectorizer uses SentenceTransformers. OpenAI embeddings are also supported when explicitly enabled. These vectors are used for semantic retrieval and claim comparison.

Implemented status: implemented, with hash embeddings as the default and neural/cloud embeddings as configurable options.

#### Cosine Similarity

Plain English: cosine similarity checks whether two text vectors point in the same direction. If they do, Cognix treats the texts as related.

Technical version: Cognix computes the normalized dot product between query, chunk, and claim vectors. Higher scores indicate stronger semantic or lexical alignment, depending on the embedding backend.

Implemented status: implemented for retrieval, claim comparison, and evidence scoring.

#### Hybrid Retrieval

Plain English: Cognix searches in more than one way. It looks for exact words and filenames, but it also searches by meaning.

Technical version: retrieval combines keyword search, SQLite vector search, ChromaDB vector search, file-name matching, file-type filters, query decomposition, Reciprocal Rank Fusion, and re-ranking.

Implemented status: implemented.

#### Reciprocal Rank Fusion

Plain English: if several search methods agree that a source is useful, Cognix pushes that source higher.

Technical version: RRF merges ranked lists with a score based on each result's rank in each retrieval method. This gives stable retrieval without needing a trained ranking model.

Implemented status: implemented.

#### Query Decomposition

Plain English: if a question is broad, Cognix can break it into smaller searches before answering.

Technical version: Cognix uses the configured LLM provider when available, otherwise deterministic splitting, to create subqueries. It retrieves evidence for each subquery and deduplicates the results.

Implemented status: implemented.

#### Re-Ranking

Plain English: after Cognix finds possible sources, it sorts them again to keep the best evidence near the top.

Technical version: the default re-ranker uses deterministic signals such as query-token coverage, keyword groups, file-type match, source diversity, and vector score. Optional backends include a SentenceTransformers cross-encoder and a local trained Cognix pair MLP.

Implemented status: deterministic re-ranking is implemented. Optional neural and local trained pair-model paths are implemented but require configured dependencies or trained artifacts.

#### Cross-Encoder Reranking

Plain English: instead of comparing a question and a chunk separately, a cross-encoder reads both together and judges whether the chunk actually answers the question.

Technical version: Cognix can call a SentenceTransformers `CrossEncoder` model when `COGNIX_RERANKER_BACKEND=cross-encoder`. This is a stronger relevance model than embedding similarity because it scores `(query, chunk)` pairs directly.

Implemented status: optional external SentenceTransformers integration exists. Cognix also now has a local trained tiny cross-encoder artifact path through `COGNIX_RERANKER_BACKEND=cognix-cross-encoder` and `data/models/cognix-reranker-cross-encoder.json`. The local artifact is not a transformer, but it is a trained neural pairwise scorer that reads query and chunk together.

#### Local Tiny Cross-Encoder

Plain English: Cognix now has a small trained model that reads two texts together, such as a question and a candidate source chunk, then predicts whether the pair belongs together.

Technical version: the model builds learned token and segment embeddings for the left and right text, combines mean pooled left/right vectors with absolute-difference, product, overlap, length, and negation features, then trains a one-hidden-layer softmax classifier with cross-entropy. It saves JSON artifacts for reranking and NLI.

Implemented status: implemented for local reranking and NLI with artifacts:

- `data/models/cognix-reranker-cross-encoder.json`
- `data/models/cognix-nli-cross-encoder.json`

This is not a transformer-class cross-encoder. It is a dependency-free local neural cross-encoder baseline.

#### Cognix Pair MLP

Plain English: Cognix now has a small trainable local model that learns from pairs of texts. It can be trained to say whether a chunk is relevant to a question or whether two claims contradict each other.

Technical version: this is a one-hidden-layer neural classifier. It converts `(left_text, right_text)` into hashed pair features, passes them through a tanh hidden layer, then uses a softmax output trained with cross-entropy SGD. It saves a JSON artifact under `data/models/`.

Implemented status: implemented for local reranker and local NLI-style contradiction classification. This is not a transformer and not a foundation model.

#### Cognix Transformer LoRA NLI

Plain English: Cognix now has a verified transformer LoRA fine-tune path for its contradiction model. This model does one job: compare two claims and classify them as `contradiction`, `related`, or `unrelated`.

Technical version: the model uses a frozen tiny transformer encoder over `[claim A] [claim B]` token pairs, then fine-tunes trainable low-rank LoRA adapter matrices on the NLI classifier projection. The base transformer features stay frozen; only `lora_a`, `lora_b`, and classifier bias are updated with cross-entropy. The trained artifact is stored at `data/models/cognix-nli-transformer-lora.json`.

Implemented status: implemented and product-eval verified for Cognix NLI. This is a real transformer+LoRA fine-tune for contradiction judgment, not a general chat model and not a large external foundation model.

#### Cognix Micro-Synthesis Model

Plain English: Cognix can now train a small local answer-synthesis policy from reviewed Cognix Q&A examples. It learns the structure of good Cognix answers, then writes new answers from retrieved evidence.

Technical version: the model consumes Cognix SFT JSONL, learns markdown answer headings, citation/evidence section conventions, bullet style, answer length, and term salience from assistant examples, then saves a JSON artifact at `data/models/cognix-micro-synthesis.json`. At runtime, `COGNIX_SYNTHESIS_BACKEND=cognix-micro` makes answer synthesis use this local trained policy when the artifact exists.

Implemented status: implemented as a local trained synthesis policy. This is not a generative transformer, not a LoRA adapter, and not a custom foundation model. It is an owned trainable artifact for grounded extractive synthesis.

#### Cognix SFT Adapter

Plain English: Cognix can train a small local behavior adapter from reviewed question-and-answer examples. It learns which answer patterns, headings, citation style, and source-grounded structure match Cognix's preferred response style.

Technical version: the adapter consumes supervised fine-tuning JSONL, converts prompt/answer pairs into hashed lexical, overlap, structure, citation, and length features, then trains a softmax prototype selector with cross-entropy. At runtime, `COGNIX_SYNTHESIS_BACKEND=cognix-sft-adapter` loads `data/models/cognix-sft-adapter.json` and uses the selected prototype to structure a grounded answer from retrieved chunks.

Implemented status: implemented as a dependency-free local SFT-style adapter path. This is real local training over Cognix examples, but it is not a transformer LoRA adapter, not a foundation model, and not a model that stores private facts in neural weights.

#### Claim Extraction

Plain English: Cognix breaks documents into factual statements it can audit later.

Technical version: chunks are passed through a structured claim extractor. If an LLM provider is available, Cognix asks for JSON claims. If not, it uses deterministic extraction for definition-like and factual/numeric statements. Claims are embedded and stored in SQLite.

Implemented status: implemented.

#### NLI-Style Contradiction Detection

Plain English: Cognix tries to detect when two claims cannot both be true.

Technical version: NLI means natural language inference. Cognix uses claim similarity to find candidate pairs, then classifies them through an LLM judge, verified local transformer LoRA NLI backend, optional trained NLI backend, optional local pair MLP, or a conservative heuristic fallback. Without a reliable judge, Cognix labels items as candidates instead of pretending they are proven contradictions.

Implemented status: candidate detection, local fallback, local pair NLI, local cross-encoder NLI, and transformer LoRA NLI are implemented. Real quality still depends on customer-specific reviewed claim-pair data.

#### Knowledge Gap Detection

Plain English: if Cognix sees a concept repeatedly but no wiki page explains it, it flags a gap.

Technical version: the intelligence pass extracts concept mentions, canonicalizes aliases, counts frequencies, compares them against existing wiki concept pages, and writes gap findings.

Implemented status: implemented as a transparent heuristic baseline, not neural NER yet.

#### Knowledge Graph

Plain English: Cognix connects concepts that appear together so the library becomes a navigable map, not just a pile of documents.

Technical version: concept co-occurrence produces `RELATED_TO` graph edges stored in SQLite. Neighbor traversal is done with SQLite graph queries. This is an explainable knowledge graph, not a graph neural network.

Implemented status: implemented as a lightweight graph layer.

#### Confidence Scoring And Calibration

Plain English: Cognix estimates how much you should trust an answer based on the evidence behind it.

Technical version: raw confidence combines source count, source diversity, retrieval score, source recency, and contradiction penalties. Calibration then learns how raw scores map to observed correctness once enough reviewed/eval outcomes exist. The trainable calibrator is a one-feature logistic probability model.

Implemented status: heuristic confidence is implemented. Empirical and logistic calibration infrastructure is implemented, but calibrated probabilities need enough reviewed/eval outcomes to become meaningful.

#### OCR And Vision

Plain English: Cognix tries to read images and scanned PDFs instead of ignoring them.

Technical version: local OCR uses Tesseract. Scanned PDFs can be rendered into page images with `pdftoppm` before OCR. OpenAI vision can optionally extract visible text and context from images when explicitly configured. Extraction artifacts store method, confidence estimate, warnings, page count, and text length.

Implemented status: implemented with local OCR/scanned-PDF OCR and optional OpenAI vision.

#### Evaluation Harness

Plain English: Cognix has tests that check whether ingestion, retrieval, and intelligence behavior actually work instead of relying on vibes.

Technical version: `backend/evals/run_evals.py` builds temporary fixture corpora, runs ingest, checks retrieval recall against expected sources, runs intelligence checks, and can run a synthetic large-file benchmark with throughput, memory, latency, chunk parity, and recall metrics.

Implemented status: implemented for smoke and synthetic large-file evals. Real-world benchmark suites still need customer-like corpora.

#### LoRA Fine-Tuning

Plain English: Cognix can fine-tune a local transformer LoRA adapter for NLI now, and can prepare future assistant-model LoRA jobs from reviewed examples.

Technical version: the verified NLI path trains a local frozen-transformer classifier with low-rank LoRA adapter matrices for claim-pair classification. Separately, Cognix exports reviewed Q&A into supervised fine-tuning JSONL, validates the dataset, supports a dry-run manifest, and has a Hugging Face PEFT/LoRA training entrypoint for optional future assistant-model training dependencies.

Implemented status: transformer LoRA fine-tune is implemented and verified for the NLI contradiction model. General assistant/foundation-model LoRA remains a future path.

### Implemented Now

#### Vector Embeddings

Cognix converts chunks and extracted claims into numeric vectors so the system can compare text by approximate meaning instead of only exact words.

Current implementation:

- Local default: `cognix-hash-embedding-v1`, a deterministic hashed bag-of-words embedding.
- Optional local neural path: SentenceTransformers through `COGNIX_LOCAL_EMBEDDING_BACKEND=sentence-transformers` and `COGNIX_SENTENCE_TRANSFORMER_MODEL`.
- Optional cloud path: OpenAI `text-embedding-3-small` through the embedding router when cloud embeddings are explicitly enabled and a valid OpenAI key is configured.
- Embedding metadata is stored with provider, model, dimensionality, and embedding source so incompatible vector spaces are not silently mixed.
- Chunk vectors are stored in SQLite and ChromaDB.
- Claim vectors are stored in SQLite.

What this means technically: the hash path is fast, private, reproducible, and dependency-light. The SentenceTransformers path is the stronger local neural option when the optional ML dependency is installed. The architecture supports reindexing by embedding backend and model version.

#### Cosine Similarity

Cognix uses cosine similarity to compare normalized vectors.

Current use:

- semantic chunk retrieval
- claim-to-claim similarity during contradiction candidate detection
- evidence relevance scoring

Cosine similarity measures whether two vectors point in a similar direction. In this system, similar direction means the two texts have similar token patterns or semantic embeddings, depending on the embedding backend used.

#### Hybrid Retrieval

Cognix does not rely on one retrieval method. It combines:

- keyword search for exact phrases, filenames, file types, and named entities
- semantic vector search through ChromaDB and stored embeddings
- metadata and file-type filtering, such as HTML/PDF/code/document filters
- deterministic relevance filtering to avoid answering from unrelated evidence

This matters because exact search catches things semantic search can miss, while semantic search catches related language that does not share the same keywords.

#### Reciprocal Rank Fusion

Cognix implements Reciprocal Rank Fusion, or RRF, to merge ranked lists from keyword and semantic search.

RRF rewards chunks that appear near the top of multiple retrieval lists. It is a practical retrieval method because it improves result stability without needing a trained ranking model.

#### Query Decomposition

Cognix can split complex questions into smaller retrieval subqueries.

Current implementation:

- Uses the configured LLM provider when available.
- Falls back to deterministic query splitting when no provider is available.
- Merges and deduplicates evidence from subqueries before synthesis.

This helps when a question asks about multiple ideas at once, for example comparing a concept across several documents.

#### Re-ranking And Evidence Filtering

Cognix applies deterministic re-ranking after retrieval.

Current signals:

- query-token coverage
- keyword group matches
- file-type match
- source diversity
- vector score

This deterministic layer remains the default no-training baseline. Cognix also has trained reranking paths described below.

#### Retrieval Diagnostics

Cognix now returns retrieval diagnostics with every Ask response.

Current diagnostics:

- chunk count
- unique source count
- unique source paths
- max and mean retrieval scores
- active file-type filters
- subquery count
- keyword coverage
- caution notes, such as single-source evidence or weak keyword coverage

These diagnostics are shown in the UI as an evidence check so users can see when an answer is narrow or weak.

#### Trained Pair Reranking

Cognix now has two optional trained reranking paths.

Current implementation:

- default: deterministic reranking
- optional: SentenceTransformers `CrossEncoder` through `COGNIX_RERANKER_BACKEND=cross-encoder`
- model setting: `COGNIX_CROSS_ENCODER_MODEL`
- local trained Cognix tiny cross-encoder through `COGNIX_RERANKER_BACKEND=cognix-cross-encoder`
- local cross-encoder artifact path: `COGNIX_LOCAL_CROSS_ENCODER_RERANKER_MODEL_PATH` or `data/models/cognix-reranker-cross-encoder.json`
- local cross-encoder training command: `.venv/bin/python backend/training/train_cross_encoder_model.py --task reranker`
- local trained Cognix pair MLP through `COGNIX_RERANKER_BACKEND=cognix-pair`
- local artifact path: `COGNIX_PAIR_RERANKER_MODEL_PATH` or `data/models/cognix-reranker-pair.json`
- training command: `.venv/bin/python backend/training/train_pair_model.py --task reranker`
- safe fallback: if the model or dependency is unavailable, Cognix keeps deterministic ranking

Why this matters: a cross-encoder reads the query and candidate chunk together, so it can judge relevance more directly than vector similarity alone.

The local Cognix pair model is not a transformer. It is a small hashed-feature neural MLP trained with cross-entropy over `(query, chunk)` pairs. It gives Cognix an owned, trainable reranker artifact path that works without downloading external model weights.

#### Trained Micro-Synthesis Policy

Cognix now has a local trained synthesis artifact path:

- training service: `backend/app/services/cognix_micro_model.py`
- training command: `.venv/bin/python backend/training/train_cognix_micro_model.py --dataset data/exports/training/qa_citation-reviewed-sft.jsonl --register-artifact`
- bootstrap command for local baseline artifacts: `.venv/bin/python backend/training/bootstrap_local_models.py --register-artifacts`
- runtime setting: `COGNIX_SYNTHESIS_BACKEND=cognix-micro`
- artifact path: `COGNIX_COGNIX_MICRO_MODEL_PATH` or `data/models/cognix-micro-synthesis.json`
- readiness capability: `custom_cognix_micro_synthesis_model`

The model learns answer structure and term salience from Cognix SFT JSONL. It still answers from retrieved chunks and preserves source citations. This gives Cognix a real local trained behavior layer for synthesis without claiming to be a full language model.

#### Local SFT Adapter

Cognix also has a local SFT-style adapter artifact path:

- training service: `backend/app/services/sft_adapter.py`
- training command: `.venv/bin/python backend/training/train_sft_adapter.py --dataset data/exports/training/qa_citation-reviewed-sft.jsonl --register-artifact`
- bootstrap command for local baseline artifacts: `.venv/bin/python backend/training/bootstrap_local_models.py --register-artifacts`
- runtime setting: `COGNIX_SYNTHESIS_BACKEND=cognix-sft-adapter`
- artifact path: `COGNIX_COGNIX_SFT_ADAPTER_PATH` or `data/models/cognix-sft-adapter.json`
- readiness capability: `custom_cognix_sft_adapter`

This adapter learns from supervised Cognix Q&A examples. The objective is cross-entropy over prototype selection: for a given query/evidence prompt, the adapter learns which reviewed answer prototype best matches the desired Cognix style. The output still remains evidence-grounded: it does not invent from weights, and it does not replace retrieval.

The workspace currently has bootstrap-trained local artifacts for:

- `data/models/cognix-reranker-pair.json`
- `data/models/cognix-reranker-cross-encoder.json`
- `data/models/cognix-nli-pair.json`
- `data/models/cognix-nli-cross-encoder.json`
- `data/models/cognix-nli-transformer-lora.json`
- `data/models/cognix-micro-synthesis.json`
- `data/models/cognix-sft-adapter.json`

These are seed baseline artifacts. They prove the local training/runtime path and make Cognix usable without downloading external model weights. The NLI transformer LoRA artifact is a verified transformer LoRA fine-tune for contradiction judgment, but it is still a small local NLI model, not a general foundation assistant.

#### Claim Extraction

Cognix extracts factual claims from chunks and stores them as auditable units.

Current implementation:

- LLM structured JSON extraction through the configured provider route when enabled.
- Deterministic fallback for definition-like and factual/numeric statements.
- Stored fields include claim text, claim type, source chunk, source file, confidence, status, source date, ingest date, and claim embedding.

Claims are the foundation for contradiction detection, staleness checks, confidence penalties, and future fine-tuning datasets.

#### LLM-As-Judge For Contradictions

Cognix uses a two-stage contradiction process:

1. Generate candidate pairs using claim similarity and lexical tension.
2. Ask the configured LLM to judge whether the pair is a real contradiction when LLM judgment is enabled.

If no LLM judge is available, Cognix marks likely pairs as contradiction candidates instead of pretending they are confirmed contradictions.

When LLM judgment is disabled, Cognix uses a local NLI-style fallback that checks content-term overlap, bounded negation, and opposing terms. Cognix also has:

- an optional external trained NLI cross-encoder path through `COGNIX_NLI_BACKEND=cross-encoder` and `COGNIX_NLI_MODEL`
- a local trained Cognix tiny cross-encoder through `COGNIX_NLI_BACKEND=cognix-cross-encoder`
- local cross-encoder artifact path: `COGNIX_LOCAL_CROSS_ENCODER_NLI_MODEL_PATH` or `data/models/cognix-nli-cross-encoder.json`
- local cross-encoder training command: `.venv/bin/python backend/training/train_cross_encoder_model.py --task nli`
- a verified local transformer LoRA NLI backend through `COGNIX_NLI_BACKEND=cognix-transformer-lora`
- transformer LoRA NLI artifact path: `COGNIX_TRANSFORMER_LORA_NLI_MODEL_PATH` or `data/models/cognix-nli-transformer-lora.json`
- transformer LoRA NLI training command: `.venv/bin/python backend/training/train_transformer_lora_nli.py --register-artifact`
- a local trained Cognix pair MLP through `COGNIX_NLI_BACKEND=cognix-pair`
- local artifact path: `COGNIX_PAIR_NLI_MODEL_PATH` or `data/models/cognix-nli-pair.json`
- training command: `.venv/bin/python backend/training/train_pair_model.py --task nli`

If a selected model artifact or dependency is unavailable, Cognix falls back to the local heuristic.

#### Knowledge Gap Detection

Cognix detects gaps by extracting concept mentions, counting frequency, and checking whether a structured wiki concept page exists.

Current implementation:

- heuristic concept extraction from chunks
- canonical concept linking for common aliases such as `AI` -> `artificial intelligence`, `ML` -> `machine learning`, and `large language model` -> `llm`
- concept mention counts in SQLite
- configurable threshold for repeated mentions
- gap findings in the Intelligence page
- gap compilation endpoint that writes `wiki/concepts/{slug}.md` from retrieved evidence

This is not transformer-based named-entity recognition yet. It is a transparent local baseline designed to be easy to inspect and improve.

#### Concept Graph

Cognix builds a lightweight concept graph.

Current implementation:

- graph edges stored in SQLite
- `RELATED_TO` edges built from concept co-occurrence
- neighbor traversal through SQLite recursive queries
- frontend graph pulse for browsing concepts

This is a knowledge graph, not a graph neural network. The value is explainable traversal over source-backed concepts, not learned graph embeddings.

#### Evidence Confidence Scoring

Cognix computes an evidence-confidence score for answers.

Current components:

- source count
- source diversity
- retrieval score
- source recency
- contradiction penalty

The score is stored with a JSON breakdown so the UI can show not only the label, but why the answer was rated high, medium, or low confidence.

This is a heuristic trust score, not a calibrated probability that the answer is true.

#### Temporal And Staleness Detection

Cognix tracks source and ingest dates where available and uses them to flag potentially stale claims.

Current implementation:

- claim dates and ingest dates stored in SQLite
- stale-claim findings generated for older claims with related newer evidence
- staleness results included in Intelligence Briefs

#### Proactive Intelligence Loop

Cognix can run an intelligence pass without the user asking a question.

Current pass:

- extracts missing claims
- updates concept mentions
- rebuilds graph edges
- detects knowledge gaps
- detects contradiction candidates
- detects stale claims
- writes findings to SQLite
- generates an Intelligence Brief markdown file in `wiki/_intelligence/`

This is the core shift from "chat with documents" toward "knowledge auditor."

#### Product Evaluation Harness

Cognix now includes a formal smoke evaluation runner at `backend/evals/run_evals.py`.

The current eval builds a temporary library, writes a labeled fixture corpus, runs ingest, checks retrieval recall against expected sources, runs the intelligence pass for the smoke suite, and exits nonzero on failure.

Current eval coverage:

- ingest coverage
- image and scanned-PDF OCR extraction artifact coverage
- persisted logistic calibration model training and application
- local neural cross-encoder reranker training, reload, and relevance-separation check
- local trained NLI cross-encoder training, reload, and contradiction/related/unrelated classification check
- transformer LoRA NLI fine-tune, reload, and contradiction/related/unrelated classification check
- local SFT adapter training, reload, prototype selection, and grounded synthesis check
- malformed/text-recoverable PDF fallback
- HTML retrieval
- PDF-type retrieval
- semantic-search retrieval
- intelligence finding and brief generation

This is the first customer-facing evaluation gate. It is still synthetic, but it now checks product behavior instead of only unit-level functions.

The eval runner also supports:

```bash
.venv/bin/python backend/evals/run_evals.py --suite large
```

The large suite checks larger ingest/retrieval behavior without running the full intelligence audit over the larger corpus.

Current large benchmark families:

- large HTML page
- large CSV table
- large JSON export
- large Python/code file
- large log/text file
- large markdown research file
- text-recoverable PDF fallback fixture
- large email export
- image OCR artifact fixture
- scanned-PDF OCR artifact fixture

The suite reports:

- total generated large-file bytes
- supported file coverage
- failed large-file count
- OCR/vision artifact coverage for image and scanned-PDF inputs
- calibrated probability model training/application from labeled eval outcomes
- trained local cross-encoder reranker gate
- trained local NLI cross-encoder gate
- transformer LoRA NLI fine-tune gate
- local SFT adapter training/runtime gate
- total chunk count
- chunk/embedding parity
- ingest elapsed time
- ingest throughput in MB/s
- peak RSS memory
- per-family retrieval latency
- per-family source hit/miss data
- large-family recall@10
- optional JSON report output through `--report`

#### Calibration And Fine-Tuning Data Substrate

Cognix now records prediction and training-example data for future calibration and model work.

Current implementation:

- `model_predictions`: stores task, predicted label, score, model, and metadata
- `prediction_outcomes`: stores reviewed or eval-provided ground truth labels
- `training_examples`: stores future fine-tuning/evaluation examples
- `calibration_models`: stores fitted probability calibrators per task
- `export_training_jsonl`: exports task-specific examples to JSONL
- empirical calibration: maps raw answer-confidence scores to observed correctness when enough reviewed/eval outcomes exist
- logistic calibration: fits `P(prediction is correct | raw confidence score)` from reviewed/eval outcomes

Current calibration behavior:

- If there are fewer than five reviewed/eval outcomes for a task, Cognix uses identity calibration and exposes that calibration was not applied.
- If enough outcomes exist but no trained calibrator has been saved, Cognix computes binned empirical accuracy with light smoothing and uses that calibrated probability for the displayed confidence label.
- If at least twenty reviewed/eval outcomes with both correct and incorrect examples exist, Cognix can train and persist a one-feature logistic calibration model.
- Training command: `.venv/bin/python backend/training/train_calibrator.py --task answer_confidence`
- The fitted model stores weight, bias, example count, positive/negative counts, Brier score, and log loss.
- When an active trained calibrator exists, answer confidence uses that persisted probability model instead of the binned fallback.
- The confidence breakdown records raw score, calibrated score, calibration method, and calibration example count.

This does not train a full Cognix language model yet. It does implement a real calibrated probability model for answer-confidence once reviewed/eval outcomes exist.

The product eval now includes a `calibrated_probability_model` gate. It writes labeled eval outcomes, trains the persisted logistic calibrator, reloads it through the normal calibration path, and verifies that high raw confidence maps above low raw confidence. This proves the calibration mechanism end to end. Customer-specific production calibration still requires real reviewed outcomes from the deployed library.

#### LoRA Fine-Tuning Pipeline

Cognix now includes a real fine-tuning entrypoint:

```bash
.venv/bin/python backend/training/train_lora.py \
  --dataset data/exports/training/qa_citation-reviewed-sft.jsonl \
  --output-dir data/models/cognix-lora \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --adapter-name cognix-lora
```

Current implementation:

- `export_sft_jsonl` converts reviewed Cognix Q&A examples into chat-style SFT JSONL.
- `validate_sft_jsonl` validates message structure, assistant responses, dataset size, and dataset hash.
- `train_lora.py --dry-run` validates data and writes `training_manifest.json` without heavy dependencies.
- Non-dry-run mode uses Hugging Face `transformers`, `datasets`, `peft`, and `trl` when optional training dependencies are installed.
- `model_artifacts` records planned/trained adapter metadata, including base model, path, manifest, status, and metrics.
- `write_ollama_modelfile` writes an Ollama-compatible Modelfile using the base model plus trained adapter path.
- `build_model_package_manifest` records how to load the local Cognix adapter as a named runtime model.

This means the fine-tuning pipeline exists. It does not mean this repository already contains a trained Cognix LoRA adapter.

#### Advanced ML Readiness Audit

Cognix now exposes machine-verifiable readiness for advanced ML features:

```text
GET /api/ml/readiness
```

The readiness audit reports:

- local neural embedding status
- neural cross-encoder reranker status
- trained NLI contradiction model status
- LoRA training stack status
- custom Cognix adapter artifact status
- local SFT adapter artifact status
- OCR/vision status
- empirical calibration status
- large-file benchmark suite status

Readiness states:

- `ready`: the feature is available and has the required local evidence/dependencies.
- `configured`: settings or records exist, but runtime verification still depends on model files, API keys, or a completed run.
- `fallback`: a deterministic/local fallback is active instead of the advanced model path.
- `missing`: the feature was requested but required dependencies/artifacts are missing.

#### OCR And Vision Pipeline

Cognix now supports structured OCR/vision extraction for images and scanned PDFs.

Current path:

1. Images use local Tesseract OCR when available.
2. Images can optionally use OpenAI vision when `COGNIX_VISION_BACKEND=openai` and an OpenAI key are configured.
3. Scanned/image-only PDFs use local `pdftoppm` page rendering plus Tesseract OCR when both tools are installed.
4. Selectable-text PDFs still use direct PDF text extraction first.
5. Every extraction writes an `extraction_artifacts` row with method, confidence, page count, text length, warnings, and metadata.

The OpenAI vision path sends the local image as a base64 data URL with an `input_image` content item and asks the model to extract visible text plus searchable visual context. If cloud vision is unavailable, Cognix falls back to local OCR/status text instead of failing ingest.

The current OCR confidence score is a lightweight extraction-quality estimate based on text shape. It is not a replacement for engine-native OCR confidence, but it gives health checks and future UI a stable signal for weak or missing OCR.

The product eval now includes `ocr_vision_artifact_coverage`. It verifies that image inputs and scanned/no-selectable-text PDFs are processed, produce extraction artifact rows, store OCR/vision methods, preserve confidence values, record page/text length, and keep warnings inspectable. The eval accepts low-confidence or empty OCR as long as Cognix records the state truthfully instead of silently pretending the image/PDF was understood.

### Implemented As Hooks Or Partial Infrastructure

- Cloud/local provider routing exists for LLM calls and optional OpenAI embeddings.
- Optional local neural embeddings exist through SentenceTransformers when installed and enabled.
- Optional trained cross-encoder reranking exists when installed and enabled.
- Optional trained NLI classification exists when installed and enabled.
- Local trainable Cognix pair MLP artifacts exist for reranking and NLI.
- Chroma can be disabled for smoke evals while SQLite vector retrieval remains testable.
- Prediction/outcome logging exists for confidence calibration.
- Empirical calibration is applied when enough reviewed/eval outcomes exist.
- A persisted logistic calibration model can be trained and reused once enough mixed correctness outcomes exist.
- Verified transformer LoRA fine-tuning exists for the Cognix NLI contradiction model.
- Fine-tuning-ready JSONL export exists for reviewed training examples.
- LoRA fine-tuning CLI exists with dry-run validation and optional real training dependencies.
- Local SFT adapter training/runtime exists as a dependency-free trained Cognix behavior artifact.
- Model artifact registry exists for planned/trained adapters.
- Advanced ML readiness endpoint exists at `/api/ml/readiness`.
- OpenAI image-bytes vision extraction exists behind explicit config.
- Scheduled background intelligence has service support, with a fallback scheduler if APScheduler is unavailable.
- ChromaDB-backed chunk search exists, while SQLite remains the durable operational store.
- Output saving feeds generated analyses back into `wiki/outputs/analysis/`.
- Fine-tuning-ready data begins to accumulate through claims, outputs, citations, and promoted wiki pages.

### Not Implemented Yet

- No completed general assistant/foundation-model LoRA run has been verified in this environment.
- No custom foundation model; Cognix now has a verified transformer LoRA fine-tune for the NLI contradiction model, while the general foundation-model path remains future work.
- No mandatory bundled transformer cross-encoder reranker; optional external and local Cognix pair-model paths exist.
- No graph neural network.
- No bundled pre-trained calibrator yet; the calibrated probability model exists but needs project-specific reviewed/eval outcomes before it becomes active.
- No mandatory bundled local neural embedding model; SentenceTransformers is optional and must be installed/enabled.
- No complete vision support for every provider; OpenAI image vision plus local image/PDF OCR paths exist.
- No full source-specific adapters for every possible export family.
- No verified trained adapter package loaded into Ollama yet; packaging helpers exist.
- No real-world public benchmark corpus is bundled yet; the large-file benchmark suite is synthetic but now measures size, coverage, chunks, embeddings, throughput, memory, latency, and recall across multiple file families.

The current system is best described as a local-first agentic retrieval and knowledge-auditing system with proactive intelligence features. It is not yet a proprietary foundation model.

## Intelligence Tab Product Model

The Intelligence tab is not another chat page. It is the proactive audit surface for the library.

The Ask page answers a question the user already has. The Intelligence tab tries to surface issues the user did not know to ask about yet.

### Run Intelligence

The `Run intelligence` button starts an audit pass over the already indexed library.

It does not ingest raw files directly. The correct flow is:

1. Put files in `data/raw/`.
2. Run ingest.
3. Run intelligence.

The intelligence pass currently:

- extracts missing claims from chunks
- updates concept mention counts
- rebuilds graph edges
- detects knowledge gaps
- detects contradiction candidates
- detects stale claims
- writes findings to SQLite
- generates an Intelligence Brief in `wiki/_intelligence/`

If the indexed database is empty or stale, the Intelligence tab will also be empty or stale. It depends on successful ingest.

### Knowledge Gaps

A knowledge gap means Cognix has seen a concept repeatedly but cannot find a structured wiki page for it.

Example:

- Many chunks mention `semantic search`.
- No `wiki/concepts/semantic-search.md` exists.
- Cognix flags `semantic search` as a gap.

This does not mean the source material is missing. It means the library has raw evidence but has not yet compiled that idea into a durable concept article.

### Concept Graph Pulse

The Concept Graph Pulse is a lightweight visual map of currently active gap concepts.

It is not the full graph database UI yet. It is a quick pulse view showing which concepts are surfacing from the current findings.

Clicking a concept asks Cognix:

```text
What does my library say about [concept]?
```

The purpose is to turn an audit finding into a research question quickly.

### Contradictions

Contradictions show claims that may disagree.

The current system has two levels:

- `contradiction_candidate`: Cognix found two claims that look related and possibly opposed.
- `contradiction`: an LLM judge confirmed the claims conflict.

If provider-backed LLM judgment is disabled, Cognix should be conservative and mark likely conflicts as candidates, not confirmed truth.

### Morning / Intelligence Brief

The brief is the audit report written back into the wiki.

It summarizes:

- library pulse
- new or open gaps
- contradiction candidates
- stale claims
- suggested review actions

The brief is generated from SQLite findings, not free-floating LLM prose. That keeps it auditable.

### Metrics

The top metric cards are status counters:

- `Knowledge gaps`: open gap findings.
- `Contradictions`: open contradiction or contradiction-candidate findings.
- `Latest brief`: date of the latest generated intelligence brief.

They are not quality scores by themselves. They are operational signals.

## Competitive Positioning

Cognix should not compete as a generic "chat with documents" clone. That market is already crowded and several competitors are more mature.

The intended wedge is:

```text
customer-owned knowledge system + proactive knowledge auditing
```

In plain terms: Cognix should become the system that watches a growing corpus, finds gaps, flags contradictions, tracks stale claims, builds wiki memory, and shows evidence confidence over time.

### Where Cognix Can Be Strong

- Proactive auditing rather than only reactive chat.
- Durable markdown wiki output, not only transient chat answers.
- Claim-level memory for contradiction and staleness checks.
- Evidence-confidence scoring attached to answers.
- Local/customer-owned architecture with optional provider routing.
- Research-library workflow that improves as the corpus grows.

### Where Cognix Is Currently Weak

- Not as polished as mature commercial products.
- Local embeddings are still deterministic hash embeddings by default, not high-quality neural local embeddings.
- PDF and large-file handling still need hardening.
- Source connectors are early.
- Evaluation benchmarks are not yet built.
- No custom trained Cognix model exists.
- No polished audio/video/study output layer exists.
- No enterprise-grade permissions/connectors/compliance layer exists.

### Honest Verdict

Cognix is not currently better than NotebookLM, AnythingLLM, Open WebUI, or Onyx as a broad general-purpose AI document product.

Cognix can become meaningfully different if it goes deeper on the proactive intelligence layer:

- find what is missing
- find what conflicts
- find what is stale
- explain confidence
- compile durable wiki memory
- keep improving the knowledge base without waiting for a direct question

That is the product category to defend. If Cognix only becomes another RAG chat interface, it will lose.

## Product Hardening ML Roadmap

This section defines the ML and AI concepts required to move Cognix from a promising v2 into a product customers can rely on.

The goal is not to add impressive-sounding models. The goal is to improve five measurable product qualities:

1. ingestion reliability
2. embedding and retrieval quality
3. UI clarity and explainability
4. evaluation discipline
5. proactive intelligence quality

Every upgrade below needs an evaluation gate before it is considered shippable.

### First Product-Hardening Tranche Implemented

Implemented on 2026-06-17:

- Optional SentenceTransformers local neural embedding backend with hash fallback.
- Safer ingest flow that computes chunks and embeddings before replacing indexed rows.
- PDF parser fallback for malformed but text-recoverable `.pdf` files.
- Ask API retrieval diagnostics and UI evidence check.
- Local NLI-style contradiction fallback for offline judgment.
- Canonical concept linking for common aliases.
- More useful Intelligence Brief priority queue and finding triggers.
- Formal smoke evaluation runner at `backend/evals/run_evals.py`.
- Optional trained cross-encoder reranker hook.
- Optional trained NLI classifier hook.
- Verified transformer LoRA fine-tune for the NLI contradiction model.
- Prediction/outcome logging for calibration.
- Empirical probability calibration for answer confidence.
- Training-example storage and JSONL export for future fine-tuning.
- SFT JSONL export, LoRA training CLI, dry-run validation, and model artifact tracking.
- Local SFT adapter training/runtime artifact for Cognix answer behavior.
- Large-file eval suite for ingest/retrieval stress.
- OpenAI image-bytes vision extraction with local image OCR and scanned-PDF OCR fallback.

Verified gates for this tranche:

- backend unit tests passed
- frontend production build passed
- smoke evaluation runner passed
- large ingest/retrieval evaluation passed

Remaining roadmap items below still matter; this tranche is not the full product-grade endpoint.

### 1. Better Ingestion Reliability

Large-file reliability is partly ML and partly systems engineering.

Plain English: before Cognix can be intelligent, it must first be able to read the file correctly, finish the job, and prove what it did.

Technical objective:

```text
For each supported file type, maximize extraction coverage while minimizing silent failures.
```

Required concepts:

- **Streaming parsing**: process large PDFs, HTML files, logs, CSVs, and exports in pieces instead of loading the entire file into memory.
- **Resumable ETL**: extraction jobs should checkpoint progress, so one failed page or chunk does not throw away the whole ingest.
- **Document layout analysis**: detect headings, tables, columns, captions, page breaks, and reading order instead of treating every file as plain text.
- **OCR quality scoring**: scanned pages should produce a confidence score so Cognix can warn when text extraction is weak.
- **Deduplication / near-duplicate detection**: identify repeated files, repeated pages, copied articles, and repeated chunks.
- **Content-type classification**: classify files as book, article, code, chat log, table, image, transcript, or export so the right parser and chunker are used.
- **Adaptive chunking**: chunk by semantic boundaries such as sections, headings, paragraphs, code blocks, and tables instead of only fixed length.

Implementation direction:

- Keep raw files immutable.
- Add parser-specific progress records.
- Store page/section/table-level extraction metadata.
- Store skipped-file reasons explicitly.
- Separate extraction failures from indexing failures.
- Do not delete old chunks until the replacement extraction/indexing succeeds.

Evaluation gates:

- Ingest coverage: at least 99% of supported files produce either indexed chunks or a clear typed error.
- Silent failure rate: 0 known silent skips in the evaluation corpus.
- Large-file test: PDFs, HTML, CSV, text logs, and code repos above 100MB complete without database corruption.
- Resume test: interrupted ingest resumes without duplicate chunks or lost files.
- Extraction spot check: sampled extracted text must preserve title, headings, page references, and enough surrounding context for citations.

### 2. Stronger Embeddings And Retrieval

Plain English: embeddings are how Cognix develops a sense of "this text is about the same thing as that text." Better embeddings mean better memory.

Technical objective:

```text
Maximize recall@k and citation precision for source-grounded questions while keeping latency and cost acceptable.
```

Required concepts:

- **Dense neural embeddings**: replace or supplement hash embeddings with SentenceTransformers, OpenAI embeddings, or another high-quality embedding backend.
- **Sparse retrieval / BM25**: keep exact lexical search for names, acronyms, filenames, code symbols, formulas, and rare terms.
- **Hybrid retrieval**: combine dense semantic retrieval with sparse keyword retrieval.
- **Reciprocal Rank Fusion**: merge retrieval lists without overtrusting one search method.
- **Cross-encoder reranking**: score query-document pairs together so the model judges actual relevance, not just vector closeness.
- **Multi-vector retrieval**: represent long documents with multiple vectors, such as section vectors, table vectors, claim vectors, and summary vectors.
- **Query decomposition**: split complex questions into smaller retrieval tasks.
- **Hard-negative mining**: collect confusing but wrong chunks so future rerankers learn what not to retrieve.

Implementation direction:

- Add a production local embedding backend such as SentenceTransformers.
- Keep provider routing: local, OpenAI, Anthropic-compatible routes where applicable, and Ollama/local LLM for synthesis.
- Store embedding backend, model, dimension, and index version for every vector.
- Support full reindex by model version without corrupting the existing index.
- Add a neural reranker behind a feature flag before making it default.

Evaluation gates:

- Retrieval recall@5: known-answer questions should retrieve the correct source in the top 5.
- Retrieval recall@10: correct source should appear in the top 10 for harder multi-hop questions.
- Citation precision: cited chunks must actually support the answer.
- Negative retrieval test: unrelated keyword traps should not dominate results.
- Latency target: typical query should return useful evidence within an acceptable interactive window on the target hardware.

### 3. Better UI Clarity And Explainability

This is not only design. It is also an ML trust problem.

Plain English: users should understand why Cognix answered, why it is uncertain, and what evidence it used.

Technical objective:

```text
Expose retrieval, confidence, uncertainty, and source grounding in a way non-technical users can inspect.
```

Required concepts:

- **Evidence attribution**: every answer section should point to supporting chunks.
- **Confidence calibration**: confidence labels should track actual correctness over time, not just look reassuring.
- **Uncertainty decomposition**: separate "not enough sources", "sources disagree", "source is old", and "retrieval is weak".
- **Evidence clustering**: group sources by theme or position so users can scan why evidence matters.
- **Abstention policy**: Cognix should say "I cannot answer from the library" when retrieval evidence is weak.
- **Counter-evidence display**: show sources that disagree, not only sources that support the generated answer.

Implementation direction:

- Make Ask output show evidence groups, confidence breakdown, and missing-evidence warnings.
- Make Intelligence findings explain the trigger: mention count, source count, contradiction pair, stale date, or weak retrieval score.
- Keep "deep mode" meaningfully different by using broader retrieval, decomposition, reranking, and structured synthesis.

Evaluation gates:

- User can identify the source of each major claim in an answer.
- Low-evidence questions trigger abstention instead of confident filler.
- Confidence labels correlate with answer correctness in the evaluation set.
- UI copy distinguishes "candidate" from "confirmed" findings.

### 4. Serious Evaluation Tests

No major retrieval, ingestion, or intelligence change ships without evaluation.

Plain English: we need a test lab for Cognix, not just "it worked on my files once."

Technical objective:

```text
Create reproducible benchmarks that measure ingestion, retrieval, answer faithfulness, contradiction detection, gap detection, and latency.
```

Required concepts:

- **Golden datasets**: small fixed corpora with known correct answers, known contradictions, known gaps, and known stale claims.
- **Retrieval metrics**: recall@k, precision@k, MRR, and nDCG.
- **Answer faithfulness**: answer claims must be supported by retrieved evidence.
- **Citation precision and citation recall**: citations should be both correct and sufficient.
- **Hallucination rate**: generated claims not supported by evidence must be counted.
- **Contradiction detection precision/recall**: measure real conflicts vs false alarms.
- **Gap detection precision**: measure whether suggested gaps are useful, not random noun phrases.
- **Latency and memory benchmarks**: track time and memory for large files and large indexes.
- **Regression testing**: a change that fixes one query must not break the evaluation set.

Implementation direction:

- Add `backend/evals/` with fixture corpora, labeled questions, and expected source paths.
- Add an eval runner that prints a pass/fail summary and writes JSON results.
- Add separate eval suites:
  - `ingest`
  - `retrieval`
  - `ask`
  - `intelligence`
  - `large_files`
- Treat eval failures as release blockers for customer-facing builds.

Minimum shipping gates:

- Unit tests pass.
- Frontend build passes.
- Ingest eval passes.
- Retrieval eval meets target recall.
- Ask eval meets citation precision target.
- Intelligence eval produces no critical false positives on the golden corpus.
- Large-file eval completes within the target resource budget.

### 5. More Polished Proactive Intelligence

Plain English: this is the feature that can make Cognix different. It should not merely list nouns. It should produce useful research judgment.

Technical objective:

```text
Turn indexed evidence into actionable findings: gaps, contradictions, stale claims, weak claims, emerging themes, and review priorities.
```

Required concepts:

- **Claim extraction**: convert source text into factual units.
- **Natural Language Inference**: classify claim pairs as entailment, contradiction, or neutral.
- **Temporal information extraction**: detect when a claim was true, observed, published, or superseded.
- **Entity and concept linking**: connect different names for the same concept.
- **Topic modeling / clustering**: group related chunks and claims into themes.
- **Graph centrality**: identify concepts that connect many parts of the library.
- **Novelty detection**: flag newly emerging ideas that were not present before.
- **Active learning**: use user review decisions to improve future gap/contradiction suggestions.
- **Finding prioritization**: rank findings by severity, evidence strength, novelty, and likely usefulness.

Implementation direction:

- Move from heuristic-only concept extraction toward model-assisted entity/concept extraction.
- Use NLI-style contradiction checking before calling a general LLM judge when possible.
- Store user feedback on findings: useful, not useful, resolved, merged, ignored.
- Use feedback to suppress repeated low-value findings.
- Make the morning brief prioritize what matters, not just what was detected.

Evaluation gates:

- Gap suggestions should be useful in human review at a defined minimum rate.
- Contradiction candidates should have acceptable precision before appearing as high-severity.
- Morning brief should be stable: same corpus produces same major findings.
- Review actions should update future findings.
- Findings must cite source chunks and explain their trigger.

### Upgrade Order

The correct build order is:

1. Ingestion reliability and large-file safety.
2. Evaluation harness and golden datasets.
3. Stronger embeddings and retrieval reranking.
4. Ask UI explainability and confidence clarity.
5. Proactive intelligence polish.

Reason: if ingest and evals are weak, every later ML feature becomes impossible to trust.

### Product Rule

No customer-facing release ships on vibes.

Before shipping:

- run unit tests
- run frontend build
- run ingestion evals
- run retrieval evals
- run answer faithfulness evals
- run intelligence evals
- inspect failure cases manually
- update architecture and user guide if behavior changed

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
