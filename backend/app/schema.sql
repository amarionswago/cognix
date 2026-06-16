CREATE TABLE IF NOT EXISTS raw_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    extension TEXT NOT NULL,
    source_type TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    status TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    processed_at TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(path, sha256)
);

CREATE INDEX IF NOT EXISTS idx_raw_files_sha256 ON raw_files(sha256);
CREATE INDEX IF NOT EXISTS idx_raw_files_status ON raw_files(status);
CREATE INDEX IF NOT EXISTS idx_raw_files_sensitivity ON raw_files(sensitivity);

CREATE TABLE IF NOT EXISTS extracted_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    text_path TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    extraction_method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES raw_files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES raw_files(id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES extracted_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_sensitivity ON chunks(sensitivity);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind);

CREATE TABLE IF NOT EXISTS ingest_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    path TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    query TEXT NOT NULL,
    answer_preview TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    retrieval_json TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outputs_status ON outputs(status);
CREATE INDEX IF NOT EXISTS idx_outputs_type ON outputs(type);

CREATE TABLE IF NOT EXISTS health_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    path TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_health_status ON health_findings(status);

CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    task TEXT NOT NULL,
    source TEXT,
    estimated_cost REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    username TEXT NOT NULL DEFAULT 'cognix-user',
    display_name TEXT NOT NULL DEFAULT 'Cognix User',
    theme TEXT NOT NULL DEFAULT 'light',
    default_answer_style TEXT NOT NULL DEFAULT 'memo',
    raw_data_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_settings (
    provider TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    api_key TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    last_status TEXT NOT NULL DEFAULT 'untested',
    last_message TEXT NOT NULL DEFAULT '',
    last_checked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_services (
    name TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    interval_seconds INTEGER NOT NULL DEFAULT 300,
    last_run_at TEXT,
    last_message TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
