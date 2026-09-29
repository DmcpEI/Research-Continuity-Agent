CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (source, target, kind)
);

CREATE TABLE IF NOT EXISTS source_revisions (
    revision_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    ingest_status TEXT NOT NULL,
    path TEXT NOT NULL,
    ingest_name TEXT NOT NULL,
    title TEXT NOT NULL,
    file_sha256 TEXT,
    content_sha256 TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (source_id, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_source_revisions_source_id ON source_revisions(source_id);
CREATE INDEX IF NOT EXISTS idx_source_revisions_file_sha256 ON source_revisions(file_sha256);
CREATE INDEX IF NOT EXISTS idx_source_revisions_content_sha256 ON source_revisions(content_sha256);
CREATE INDEX IF NOT EXISTS idx_source_revisions_ingest_name ON source_revisions(ingest_name);
CREATE INDEX IF NOT EXISTS idx_source_revisions_path ON source_revisions(path);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    title,
    text,
    content='nodes',
    content_rowid='rowid',
    tokenize = "unicode61 tokenchars '._-+'"
);

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, title, text)
    VALUES (new.rowid, new.title, coalesce(new.text, ''));
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, coalesce(old.text, ''));
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, coalesce(old.text, ''));
    INSERT INTO nodes_fts(rowid, title, text)
    VALUES (new.rowid, new.title, coalesce(new.text, ''));
END;
