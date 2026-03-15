# Data Model

All contracts are defined in `rca/contracts/`.

---

## ID scheme

Source: `rca/contracts/ids.py`

### Patterns

| Type | Pattern | Example |
|---|---|---|
| Source node | `src:<namespace>/<slug>` | `src:pdf/jampacker_an_efficient_and_reliable_robotic_bin_packing_system_for_cuboid_objects` |
| Chunk node | `chk:<namespace>/<slug>:<NNNN>` | `chk:pdf/jampacker_an_efficient_and_reliable_robotic_bin_packing_system_for_cuboid_objects:0034` |

- `namespace` — typically `pdf` for ingested documents, `papers` for manually registered entries
- `slug` — lowercase; underscores in the original filename are preserved; spaces and other non-alphanumeric characters (except `.`, `-`, `_`, `/`) are replaced with hyphens
- `NNNN` — 4-digit zero-padded chunk index within the source document

### Helper functions

```python
from rca.contracts.ids import make_source_id, make_chunk_id

source_id = make_source_id("pdf", "my_paper")
# → "src:pdf/my_paper"

chunk_id = make_chunk_id(source_id, 3)
# → "chk:pdf/my_paper:0003"
```

The ingest layer strips file extensions before calling `make_source_id`, so `"my_paper.pdf"` is passed as `"my_paper"`. Passing a name with an extension (e.g. `"my_paper.pdf"`) would produce `src:pdf/my_paper.pdf` — the `.` is a valid slug character and is not stripped automatically.

Validation regexes (`SOURCE_ID_PATTERN`, `CHUNK_ID_PATTERN`) are defined in `ids.py` and used by the ingestion and retrieval layers to reject malformed IDs at the boundary.

---

## Node kinds

Source: `rca/contracts/nodes.py` — `NodeKind` enum

| Kind | Description |
|---|---|
| `source` | Top-level document (PDF, note, or other ingested file) |
| `chunk` | Text segment extracted from a source document |
| `note` | Manually authored note or annotation |
| `paper` | Research paper entry (used for PDF-ingested sources post-ingest) |
| `experiment` | Recorded experiment or run result |
| `digest` | Generated summary or digest document |

In practice the graph currently contains `paper` nodes (source-level) and `chunk` nodes. The `note`, `experiment`, and `digest` kinds are reserved for planned features.

### Node fields

```python
class Node(BaseModel):
    id: str           # unique node ID (src: or chk: scheme)
    kind: NodeKind    # one of the kinds above
    title: str        # human-readable label
    text: str | None  # full extracted text for source nodes; chunk text for chunk nodes
    metadata: dict    # arbitrary key-value pairs (authors, year, venue, etc.)
    created_at: str   # ISO 8601 timestamp
```

---

## Edge kinds

Source: `rca/contracts/nodes.py` — `EdgeKind` enum

| Kind | Direction | Description |
|---|---|---|
| `contains` | source → chunk | A source document contains a chunk |
| `derived_from` | chunk → source | A chunk was derived from a source |
| `references` | paper → paper | A paper references another paper |
| `cites` | chunk → chunk | A chunk cites content from another chunk |
| `related_to` | any → any | General semantic relation used for graph expansion |
| `produced_by` | experiment/digest → source | An output was produced from a source |

The `contains` edges are created at ingest time for every chunk. `derived_from`, `references`, `cites`, `related_to`, and `produced_by` are available in the contract but are not emitted by the current ingest flow. The parent-expansion relation is `contains`, and `RetrieveFlow._expand_to_sources()` now filters to that edge kind explicitly.

### Edge fields

```python
class Edge(BaseModel):
    source: str     # source node ID
    target: str     # target node ID
    kind: EdgeKind  # one of the kinds above
    weight: float   # relevance weight (default 1.0)
```

---

## Storage layout

```
.rca/
├── graph.sqlite3       # GraphStore: nodes, edges, metadata
├── vectors/            # VectorStore: ChromaDB persistent embeddings
├── events.jsonl        # Event log (append-only ingest events)
├── telemetry.jsonl     # Telemetry log
└── experiments.sqlite3 # Experiment tracking (planned)
```

### SQLite schema (graph.sqlite3)

Two tables plus indices:

```sql
CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    text        TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',   -- JSON stored as TEXT
    created_at  TEXT NOT NULL
);

CREATE TABLE edges (
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    metadata    TEXT NOT NULL DEFAULT '{}',   -- JSON stored as TEXT
    created_at  TEXT NOT NULL,
    PRIMARY KEY (source, target, kind)
);

CREATE INDEX idx_nodes_kind    ON nodes(kind);
CREATE INDEX idx_edges_source  ON edges(source);
CREATE INDEX idx_edges_target  ON edges(target);

CREATE VIRTUAL TABLE nodes_fts USING fts5(
    title,
    text,
    content='nodes',
    content_rowid='rowid',
    tokenize = "unicode61 tokenchars '._-+'"
);

CREATE TRIGGER nodes_ai AFTER INSERT ON nodes ...;
CREATE TRIGGER nodes_ad AFTER DELETE ON nodes ...;
CREATE TRIGGER nodes_au AFTER UPDATE ON nodes ...;
```

The production lexical path now uses the FTS5 virtual table (`nodes_fts`) with BM25 ranking. The original token-wise `LIKE` query across `lower(title)` and `lower(coalesce(text, ''))` is still retained in `GraphStore.search_nodes_like()` for reference and regression testing because it documents the earlier production implementation. `RetrieveFlow` still applies exact word-token reranking on the returned lexical candidates to avoid partial-word false positives.

---

## Normalization

`normalize_identifier(name)` in `ids.py` produces consistent slugs from arbitrary filenames or titles:

- Lowercased
- Spaces → hyphen (`-`)
- Non-alphanumeric characters (except `.`, `-`, `_`, `/`) → hyphen
- Multiple consecutive separators collapsed to single hyphen
- Leading and trailing separators stripped

Underscores in the original input are preserved as-is (they are valid slug characters). This is why PDF filenames like `stable_bin_packing_of_non-convex...` retain their underscores while any spaces in a title would become hyphens.
