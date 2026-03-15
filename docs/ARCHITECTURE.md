# Architecture

## Overview

RCA is a local-first research knowledge system. Given a corpus of technical PDFs, it ingests documents into a dual-store, rewrites queries before retrieval, performs hybrid search, and generates grounded answers with source citations.

```
PDF / Notes
    │
    ▼
IngestFlow
    ├── chunking (size=1200 chars, overlap=150)
    ├── metadata extraction (title, authors, source ID)
    ├── VectorStore  ← dense embeddings (nomic-embed-text via Ollama)
    └── GraphStore   ← document registry + keyword search index (SQLite)
    │
    ▼
RetrieveFlow
    ├── vector search  ← approximate nearest-neighbour (ChromaDB)
    ├── graph keyword  ← LIKE candidate query over title/text (SQLite)
    ├── exact-word lexical rerank + dedup → ranked RetrievalBundle
    └── source expansion ← chunk → parent src: nodes
    │
    ▼
GenerateFlow
    ├── query rewrite (inside GenerateFlow, optional)
    ├── retrieval via RetrieveFlow
    ├── prompt assembly (system + context chunks + query)
    ├── LLM generation (qwen2.5:14b via Ollama)
    ├── citation extraction (_extract_citations)
    ├── source-ID resolution (chunk ID → parent src: node)
    └── grounding check → GenerateResult(answer, citations, grounded)
    │
    ▼
Direct flow orchestration  (stateless per query)
```

Orchestration is handled directly via `RetrieveFlow` and `GenerateFlow`. A LangGraph-based agent workflow is on the roadmap.

---

## Layers

### IngestFlow (`rca/flows/ingest_flow.py`)

Entry point for adding documents to the knowledge base. Accepts a PDF path or folder.

1. **Parse** — extracts text and metadata from PDF
2. **Chunk** — splits into overlapping text windows (`chunk_size`, `chunk_overlap` from settings)
3. **Embed** — sends chunks to `nomic-embed-text` via Ollama for dense vector embeddings
4. **Store (vector)** — writes embeddings to ChromaDB with chunk ID as document ID
5. **Store (graph)** — writes `paper` source node and `chunk` nodes to SQLite; registers edges

Node IDs follow the scheme defined in `rca/contracts/ids.py` — see [DATA_MODEL.md](DATA_MODEL.md).

### RetrieveFlow (`rca/flows/retrieve_flow.py`)

Hybrid retrieval over the dual-store. Called with a query string, returns a `RetrievalBundle`.

1. **Vector search** — top-k approximate nearest-neighbour over embeddings
2. **Graph keyword** — SQLite `LIKE` query over node title and text to fetch lexical candidates
3. **Lexical rerank** — exact word-token overlap over title and text removes partial-word false positives before merge
4. **Merge** — deduplicates by node ID, scores merged by max, sorted descending
5. **Source expansion** — follows chunk → source edges and appends parent `src:` nodes
6. **Edge collection** — gathers related edges for the returned nodes

**Ablation results** (hit@5, n=30 golden pairs):

| Config | hit@5 | hit@10 |
|---|---|---|
| vector-only | 80.0% | 96.7% |
| vector + keyword | 80.0% | 96.7% |
| vector + keyword + expansion | 96.7% | 100.0% |
| full + query rewrite | 100.0% | 100.0% |

Exact-word lexical rescoring plus source expansion materially improve retrieval on edge cases; query rewriting closes the remaining top-5 miss on the current 30-case set.

### GenerateFlow (`rca/flows/generate_flow.py`)

Grounded answer generation with citation enforcement.

1. **Rewrite query** — uses the LLM to derive a dense keyword query before retrieval
2. **Retrieve** — calls `RetrieveFlow.retrieve()`
3. **Assemble context** — formats retrieved hits into the prompt context block
4. **Prompt** — system prompt enforces citation format `[[src:paper_name]]` or `[[chk:paper:NNNN]]`
5. **Generate** — calls `qwen2.5:14b` via Ollama
6. **Extract citations** — parses inline citation markers from generated text
7. **Resolve source IDs** — strips `:NNNN` suffix and resolves chunk IDs to parent `src:` nodes unconditionally
8. **Ground check** — `grounded=True` if at least one valid citation was resolved to a returned hit

**QueryTrace observability.** Each query builds a single in-memory `QueryTrace` that records the six pipeline stages: `llm_rewrite`, `vector_search`, `graph_search`, `score_merge`, `expand_sources`, and `llm_generate`. The trace is attached to the returned `GeneratedAnswer` (and nested `RetrievalBundle` during retrieval), while persistence stays outside the core flows; the evaluation harness writes per-query trace files under `eval/results/traces/`.

### GraphStore (`rca/store/graph_store.py`)

SQLite-backed structured store. Three responsibilities:

- **Document registry** — tracks all ingested sources with metadata and provenance
- **Keyword search** — `LIKE` query over `lower(title)` and `lower(text)` for lexical candidate retrieval
- **Entity graph** — nodes (papers, chunks, notes, experiments) and typed edges (contains, references, cites, related_to)

See [DATA_MODEL.md](DATA_MODEL.md) for node kinds, edge kinds, and ID patterns.

### VectorStore (`rca/store/vector_store.py`)

ChromaDB-backed dense retrieval index. Stores chunk embeddings with chunk IDs as document keys. Queried by the RetrieveFlow for approximate nearest-neighbour search.

In v2, this will be replaced by `pgvector` in Postgres for a unified store with better production operability.

---

## Why two stores?

| Store | Role | Why not the other |
|---|---|---|
| ChromaDB | Semantic retrieval — approximate nearest-neighbour over dense embeddings | SQLite LIKE can't do semantic similarity |
| SQLite | Document registry, keyword search, graph traversal, provenance, eval results | ChromaDB has no structured query, no graph, no foreign keys |

Each store has a distinct function. The graph is not decorative — it enables chunk-to-source resolution, structured traversal for related content, and provenance tracking that vector search cannot provide.

---

## Data flow for a query

```
User query
    │
    ├─ GenerateFlow._rewrite_query(query) → rewritten_query
    │
    ├─ VectorStore.query(rewritten_query, limit=10)        → vector_hits
    ├─ GraphStore.search_nodes(rewritten_query, limit=10)  → keyword_hits
    ├─ exact-word lexical rerank over title/text           → rescored_hits
    ├─ RetrieveFlow._expand_to_sources(...)                → source_hits
    │
    ├─ merge(vector_hits, rescored_hits, source_hits) → RetrievalBundle
    │
    ├─ GenerateFlow.generate_answer(query)
    │       ├─ format context from bundle.hits
    │       ├─ LLM(system_prompt + context + query) → raw_answer
    │       ├─ _extract_citations(raw_answer, bundle) → citations
    │       └─ GeneratedAnswer(answer, citations, grounded)
    │
    └─ UI / API response with inline citations
```

---

## Settings

All configuration via environment variables with `RCA_` prefix. See `.env.example` for full list.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `RCA_CHUNK_SIZE` | 1200 | Characters per chunk |
| `RCA_CHUNK_OVERLAP` | 150 | Overlap between chunks in characters |
| `RCA_EMBEDDING_MODEL` | nomic-embed-text | Ollama embedding model |
| `RCA_GENERATION_MODEL` | qwen2.5:14b | Ollama generation model |
| `RCA_EMBEDDING_BASE_URL` | http://localhost:11434 | Ollama base URL |
| `RCA_GRAPH_DB_PATH` | .rca/graph.sqlite3 | SQLite graph store path |
| `RCA_VECTOR_DIR` | .rca/vectors | ChromaDB persist directory |
