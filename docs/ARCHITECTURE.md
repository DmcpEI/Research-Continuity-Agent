# Architecture

## Overview

RCA is a local-first research knowledge system. Given a corpus of technical PDFs, it ingests documents into a dual-store, rewrites queries before retrieval, performs hybrid search, and generates grounded answers with source citations. The same application can now run against local Ollama or an OpenAI-compatible backend for both generation and embeddings.

```
PDF / Notes
    │
    ▼
IngestFlow
    ├── chunking (size=1200 chars, overlap=150)
    ├── metadata extraction (title, authors, source ID)
    ├── VectorStore  ← dense embeddings (via configurable client-backed embedding function)
    └── GraphStore   ← document registry + lexical indices (SQLite)
    │
    ▼
RetrieveFlow
    ├── vector search  ← approximate nearest-neighbour (ChromaDB)
    ├── graph keyword  ← FTS5 / BM25 over title/text (production)
    ├── like fallback  ← token-wise LIKE kept for reference/testing
    ├── exact-word lexical rerank + dedup → ranked RetrievalBundle
    └── source expansion ← chunk → parent src: nodes
    ├── cross-encoder rerank ← reorder full merged candidate pool
    └── reserve new-source slot ← last of top-k goes to an unseen paper
    │
    ▼
GenerateFlow
    ├── query rewrite (inside GenerateFlow, optional)
    ├── retrieval via RetrieveFlow
    ├── prompt assembly (system + context chunks + query)
    ├── LLM generation (qwen3.5:4b locally or OpenAI-compatible in deployment)
    ├── citation extraction (_extract_citations)
    ├── source-ID resolution (chunk ID → parent src: node)
    ├── rerank gate (max rerank_score <= -3.618 → abstain before LLM call)
    ├── facts check (hedged replies only: a second LLM call lists what the reply gives)
    ├── abstention check (facts check or hedge rule + citation validity + retrieval confidence)
    └── grounding check → GeneratedAnswer(answer, citations, outcome, failure_labels)
    │
    ▼
Direct flow orchestration  (stateless per query)
```

Grounded chat is handled directly via `RetrieveFlow` and `GenerateFlow`. Multi-turn tool use is a separate agent loop (see "Agent loop" below).

---

## Layers

### IngestFlow (`rca/flows/ingest_flow.py`)

Entry point for adding documents to the knowledge base. Accepts a PDF path or folder.

1. **Parse** — extracts text and metadata from PDF
2. **Chunk** — splits into overlapping text windows (`chunk_size`, `chunk_overlap` from settings)
3. **Embed** — sends chunks through the configured embedding backend for dense vector embeddings
4. **Store (vector)** — writes embeddings to ChromaDB with chunk ID as document ID
5. **Store (graph)** — writes `paper` source node and `chunk` nodes to SQLite; registers edges

Node IDs follow the scheme defined in `rca/contracts/ids.py` — see [DATA_MODEL.md](DATA_MODEL.md).

**Revisions and idempotent re-ingest.** Before writing, `IngestFlow` computes `file_sha256` and `content_sha256` (normalized extracted text) and resolves the logical source in this order: file hash → content hash → prior path → prior normalized ingest name → legacy source ID → new source. The result is one of three `IngestStatus` values:

- `unchanged` — same content; no chunk or vector writes. A renamed same-content file still records a revision row so alias lookups keep working.
- `updated` — same logical `source_id`, new content; `GraphStore.replace_source_chunks_atomic()` swaps the chunk set in one transaction, then stale vectors are deleted and new ones upserted.
- `created` — new logical source.

Every non-`unchanged` ingest appends a row to `source_revisions`. Retrieval, citations, and eval still operate only on logical `src:` IDs and the latest active chunks. Vector deletes fail closed on Chroma errors so stale vectors are never silently served.

### RetrieveFlow (`rca/flows/retrieve_flow.py`)

Hybrid retrieval over the dual-store. Called with a query string, returns a `RetrievalBundle`.

1. **Vector search** — top-k approximate nearest-neighbour over embeddings
2. **Graph keyword** — SQLite FTS5/BM25 query over node title and text to fetch lexical candidates on the production path
3. **Lexical rerank** — exact word-token overlap over title and text removes partial-word false positives before merge
4. **Merge** — deduplicates by node ID, scores merged by max, sorted descending
5. **Source expansion** — follows chunk → source edges and appends parent `src:` nodes
6. **Cross-encoder rerank** — reranks the merged candidate set with `cross-encoder/ms-marco-MiniLM-L-6-v2`, preserving the original retrieval score scale for downstream context selection. The API loads the model at startup. If it cannot load or score, retrieval returns the hits unranked, `RetrievalBundle.rerank_error` is set, the answer carries the label `rerank_unavailable`, and `/status` warns that results are not reranked and the no-match gate is off; a failed load is retried at most once a minute, and the next successful rerank clears the warning
6b. **Reserved new-source slot** — truncation to `limit` keeps the top `limit-1` hits as ranked and, when hit `limit` would repeat a paper already shown, gives the last slot to the best-ranked unseen paper with rerank score above `retrieval_reserve_min_rerank_score` (`-6.0`); otherwise plain truncation (`reserve_new_source()`, setting `retrieval_reserve_new_source`). The top hit is never replaced, so the rerank gate's `max rerank_score` is unchanged; the post-generation fused-score check can move up or down if the swapped-in or swapped-out hit holds the fused-score maximum (rare). With the reranker off or failing, hits carry no rerank score, so the floor does not apply, and the candidate pool is small enough that the slot rarely changes anything.
7. **Edge collection** — gathers related edges for the returned nodes

**Ablation results** (hit@5 / hit@10, n=90 answerable pairs):

| Config | hit@5 | hit@10 |
|---|---|---|
| fts5-only (BM25 baseline) | 95.6% | 98.9% |
| vector-only (dense baseline) | 84.4% | 91.1% |
| vector + keyword (FTS5) | 84.4% | 91.1% |
| vector + keyword + expansion | 95.6% | 96.7% |
| full + query rewrite | 95.6% | 98.9% |
| production path (query-type aware) | 98.9% | 98.9% |

The important current result is that FTS5/BM25 outperformed the original production `LIKE` lexical path strongly enough that the lexical backbone was migrated. On the 2026-10-02 run, the production path (rewrite skipped for proper nouns, query type passed to retrieval) is the best configuration; expansion still provides most of the hybrid lift over dense retrieval alone.

### GenerateFlow (`rca/flows/generate_flow.py`)

Grounded answer generation with citation enforcement.

1. **Classify + rewrite query** — a lightweight classifier can skip rewrite for proper-noun queries; conceptual and hybrid queries still derive a dense keyword query before retrieval
2. **Retrieve** — calls `RetrieveFlow.retrieve()`
3. **Assemble context** — formats retrieved hits into the prompt context block. Each block carries the hit's full node text (`RetrievalHit.text`, up to one chunk); the 240-character `excerpt` is only for display and for the cross-encoder, whose rerank-gate threshold was calibrated on it. The post-generation support check reads the same full text.
4. **Prompt** — system prompt enforces citation format `[[src:paper_name]]` or `[[chk:paper:NNNN]]`
5. **Generate** — calls the configured generation backend
6. **Extract citations** — parses inline citation markers from generated text
7. **Resolve source IDs** — strips `:NNNN` suffix and resolves chunk IDs to parent `src:` nodes unconditionally
8. **Abstention gates** — before generation, a calibrated cross-encoder score gate abstains when no hit is a strong match (label `low_rerank_score`); after generation, a reply that hedges ("the context does not contain ...", `rca/flows/facts_check.is_hedged`) goes to the facts check: a second call with the chat model lists the statements in it that give what the question asks for, and none means RCA declines (`facts_check_no_answer`). If the check is off (`RCA_FACTS_CHECK_ENABLED=false`) or fails (`facts_check_failed`), a hedged reply declines only when its citation fails or retrieval is weak. Independently, an invalid or unsupported citation, or a missing citation with weak retrieval, also declines. Prompt and measurements: docs/EVAL.md, "Answer classifiers"
9. **Outcome** — every reply carries exactly one `AnswerOutcome`; `failure_labels` say why:
   - `answered`: the answer cites retrieved context and passes the citation check (the cited ID resolves to a retrieved hit and the answer shares content words with the cited text). This is a citation check, not proof that each claim is supported.
   - `unverified`: an answer is shown but no citation passes the check (`missing_citation`, `invalid_citation` or `unsupported_claim`).
   - `not_found`: RCA declined, either at a gate before generation (`empty_retrieval`, `empty_context`, `low_rerank_score`) or after it (step 8).
   - `conversational`: the question was about the chat itself, so retrieval was skipped.

   `grounded` and `abstained` are derived from the outcome (`answered` and `not_found`) and are still present in API payloads and eval artifacts. `/chat` returns `outcome`; the React badge reads it and shows a refusal as "No evidence found" instead of "Unverified". A failed request is marked `error` by the client only.

**QueryTrace observability.** Each query builds a single in-memory `QueryTrace` that records the seven pipeline stages: `llm_rewrite`, `vector_search`, `graph_search`, `score_merge`, `expand_sources`, `cross_encoder_rerank`, and `llm_generate`, plus `facts_check` when a hedged reply is checked. The trace is attached to the returned `GeneratedAnswer` (and nested `RetrievalBundle` during retrieval), while persistence stays outside the core flows; the evaluation harness writes per-query trace files under `eval/results/traces/`.

### GraphStore (`rca/store/graph_store.py`)

SQLite-backed structured store. Three responsibilities:

- **Document registry** — tracks all ingested sources with metadata and provenance
- **Keyword search** — BM25 over an SQLite FTS5 virtual table (`nodes_fts`) for the production lexical path
- **Historical baseline** — token-wise `LIKE` search retained as `search_nodes_like()` for reference/testing and comparison against the earlier implementation
- **Entity graph** — nodes (papers, chunks, notes, experiments) and typed edges (contains, references, cites, related_to)

See [DATA_MODEL.md](DATA_MODEL.md) for node kinds, edge kinds, and ID patterns.

### VectorStore (`rca/store/vector_store.py`)

ChromaDB-backed dense retrieval index. Stores chunk embeddings with chunk IDs as document keys. Queried by the RetrieveFlow for approximate nearest-neighbour search. Embeddings now flow through a small configurable wrapper, so the same store boundary can use the default local Ollama models or an OpenAI-compatible embedding endpoint.

Failure and recovery: every call tries Chroma. If a call fails (usually the embedding server is down), that call is served from a JSON bag-of-words store and the store reports `json` with a warning in `/status` and the UI; the next successful call clears it, with no restart. Chunks written while degraded go to the JSON file, which is on disk and survives a restart, and are copied into Chroma 64 per successful call (`/status` reports how many are waiting); a chunk Chroma rejects as invalid (`ValueError`/`TypeError`) is moved to `<collection>.rejected.json`, while any other error keeps it pending. Deletes drop pending copies first and then go to Chroma (failing closed), under the same lock as the copy. Without `chromadb` installed, the JSON file is the whole index. `fallback_events` counts degraded calls; eval provenance uses it to flag runs that degraded and recovered.

---

### Agent loop (`rca/agent/loop.py`, `rca/agent/tools.py`)

`AgentLoop` runs a multi-turn tool-calling conversation (up to 10 LLM turns). `ToolRegistry` exposes a native `search_knowledge_base` adapter, which shares chat's `RetrieveFlow` and vector store, plus read-only MCP tools: filesystem (`list_directory`, `read_text_file`, `search_text`) sandboxed to `RCA_FILESYSTEM_ROOT`, and experiments (`list_runs`, `get_run`).

Path guards in `ToolRegistry.call` redirect filesystem calls that cannot work without touching the MCP server: a knowledge-base source ID used as a path (`src:...`/`chk:...`, also in `[[...]]` form) gets guidance to answer from the search excerpts and cite the source, and a `.pdf` path is pointed at `search_knowledge_base`. Both report status `error`, so the agent benchmark counts them as invalid calls; the model only sees the guidance text. Stop reasons: `final_answer`, `empty_final_answer`, `unparsed_tool_call`, `clarification_needed`, `max_iterations`, `fallback_no_tools`, `error`.

### Model picker

Models are read live from the backend (`/api/tags` for Ollama), so each machine sees its own list. Embedding-only names (`embed`, `bge`, `e5`, `gte`, `nomic-embed`) are excluded, the rest are ranked for text retrieval and synthesis (instruction-tuned and text families first; multimodal models stay as compatible candidates), and `RCA_MODEL_RECOMMEND_INCLUDE`, `RCA_MODEL_RECOMMEND_EXCLUDE` and `RCA_MODEL_RECOMMEND_PREFERRED` override the ranking. The picker groups them into "Recommended for RCA" and "Other compatible".

## Why two stores?

| Store | Role | Why not the other |
|---|---|---|
| ChromaDB | Semantic retrieval — approximate nearest-neighbour over dense embeddings | SQLite lexical indices can't do semantic similarity |
| SQLite | Document registry, keyword search, graph traversal, provenance | ChromaDB has no structured query, no graph, no foreign keys |

Each store has a distinct function. The graph is not decorative — it enables chunk-to-source resolution, structured traversal for related content, and provenance tracking that vector search cannot provide.

---

## Data flow for a query

```
User query
    │
    ├─ GenerateFlow._rewrite_query(query) → rewritten_query
    │
    ├─ VectorStore.query(rewritten_query, limit=10)        → vector_hits
    ├─ GraphStore.search_nodes(rewritten_query, limit=10)  → keyword_hits (FTS5 production)
    ├─ GraphStore.search_nodes_like(query, limit=10)       → like_hits (reference only)
    ├─ exact-word lexical rerank over title/text           → rescored_hits
    ├─ RetrieveFlow._expand_to_sources(...)                → source_hits
    │
    ├─ merge(vector_hits, rescored_hits, source_hits) → RetrievalBundle
    │
    ├─ GenerateFlow.generate_answer(query)
    │       ├─ format context from bundle.hits
    │       ├─ LLM(system_prompt + context + query) → raw_answer
    │       ├─ _extract_citations(raw_answer, bundle) → citations
    │       ├─ abstention gate → optional no-answer result
    │       └─ GeneratedAnswer(answer, citations, outcome)
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
| `RCA_LLM_BACKEND` | ollama | Backend selector for chat, tool use, and embeddings |
| `RCA_ENABLE_FILESYSTEM_TOOLS` | true | Enable or disable filesystem MCP tools in the agent loop |
| `RCA_EMBEDDING_MODEL` | nomic-embed-text | Ollama embedding model |
| `RCA_GENERATION_MODEL` | qwen3.5:4b | Ollama generation model |
| `RCA_AGENT_MODEL` | qwen3.5:4b | Ollama model for the tool-using agent loop |
| `RCA_EMBEDDING_BASE_URL` | http://localhost:11434 | Ollama embedding base URL |
| `RCA_LLM_BASE_URL` | http://localhost:11434 | Ollama chat base URL |
| `RCA_OPENAI_BASE_URL` | https://api.openai.com/v1 | OpenAI-compatible base URL |
| `RCA_FILESYSTEM_ROOT` | current directory | Root exposed to filesystem MCP tools when enabled |
| `RCA_GRAPH_DB_PATH` | .rca/graph.sqlite3 | SQLite graph store path |
| `RCA_VECTOR_DIR` | .rca/vectors | ChromaDB persist directory |
