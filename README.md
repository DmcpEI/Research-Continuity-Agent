# Research Continuity Agent (RCA)

A **local-first research knowledge system** that ingests technical PDFs, performs hybrid retrieval over semantic and structured links, and generates grounded answers with source citations — built to support a Master's thesis on structured visual perception for robotic inventory generation.

---

## What it does

Given a corpus of research PDFs, RCA:

1. **Ingests** documents into a dual-store (vector embeddings + structured graph)
2. **Rewrites** queries before retrieval to improve keyword density and recall
3. **Retrieves** via hybrid search: semantic vector similarity + graph keyword FTS + graph neighbourhood expansion
4. **Generates** grounded answers with inline citations enforced at the prompt level
5. **Evaluates** answer quality against a golden Q&A set with retrieval, citation, and keyword metrics

---

## Architecture

```
PDF / Notes
    │
    ▼
IngestFlow
    ├── chunking + metadata extraction
    ├── VectorStore  (ChromaDB — semantic retrieval index)
    └── GraphStore   (SQLite — document registry, provenance, entity links)
    │
    ▼
RetrieveFlow
    ├── query rewriter  (LLM — dense keyword expansion before retrieval)
    ├── vector search   (nomic-embed-text via Ollama)
    ├── graph FTS       (SQLite full-text search)
    └── graph expansion (neighbour traversal for related chunks)
    │
    ▼
GenerateFlow
    ├── grounded answer generation  (qwen2.5:14b via Ollama)
    └── citation enforcement        (prompt-level + fallback injection)
    │
    ▼
Orchestrator  (LangGraph-ready, stateless per query)
```

**Why three stores?**

| Store | Role | Justification |
|---|---|---|
| ChromaDB | Semantic retrieval index | Fast approximate nearest-neighbour over dense embeddings |
| SQLite (graph) | Document registry, provenance, entity links | Structured traversal, explainable expansion, zero-dependency deployment |
| SQLite (eval) | Run results, golden pairs | Reproducible evaluation without external services |

Each store has a distinct function. The graph is not decorative — it enables chunk-to-source resolution, neighbour expansion for related content, and provenance tracking that vector search cannot provide alone.

---

## Component details

**Knowledge store.** Two persistent stores back the system. `GraphStore` holds typed nodes and directed edges in a local SQLite database. Node kinds are `source`, `chunk`, `note`, `paper`, `experiment`, and `digest`; edge kinds are `contains`, `derived_from`, `references`, `cites`, `related_to`, and `produced_by`. `VectorStore` wraps ChromaDB with an Ollama embedding function using `nomic-embed-text` (768 dimensions, cosine similarity). When ChromaDB is unavailable, it falls back to a JSON file with bag-of-words cosine scoring.

**ID system.** All identifiers are stable and deterministic. Source nodes follow the pattern `src:namespace/name` (e.g., `src:pdf/attention-is-all-you-need`). Chunk nodes derive from their source: `chk:namespace/name:0000`. Identifiers are validated with regular expressions at the contract boundary so invalid IDs cannot enter the stores.

**MCP servers.** Two MCP servers expose tools over stdio. The filesystem server sandboxes all path resolution to a configured root directory and delegates text search to ripgrep. The experiments server provides full CRUD for experiment runs (record, list, update, get) with a status lifecycle of `pending → running → complete / failed`, backed by a separate SQLite database.

**Ingest flow.** `IngestFlow.ingest_path` dispatches on file type: `.pdf` → `PDFExtractor`, `.md`/`.txt` → `NoteExtractor`, `.json`/`.yaml` → `ExperimentExtractor`, directory → `GitExtractor`. Extracted text is split into boundary-aware chunks (default 1200 characters, 150 overlap) that prefer paragraph breaks, then newlines, then word boundaries before hard-cutting. Each chunk becomes a graph node linked to its source via a `contains` edge, and all chunks are upserted into the vector store in a single batch call.

**Retrieve flow.** `RetrieveFlow.retrieve` composes vector similarity search with lexical graph search (`LIKE` on title/text). Results are merged by node ID, then `_expand_to_sources` follows `contains` edges from chunk hits to surface parent source nodes. The final bundle contains up to 10 ranked hits (chunks + expanded sources) and the associated graph edges.

**Generate flow.** `GenerateFlow.generate_answer` is a three-step pipeline:
1. **Query rewriting** — the user's natural-language question is sent to the LLM to produce a dense 8–12 keyword technical search query, improving vector recall over conversational phrasing.
2. **Grounded context** — the rewritten query is passed to `RetrieveFlow`; hits scoring above 0.55 (or any `src:` node) are formatted into a bracketed context block. If the rewritten query yields no context, the pipeline retries with the original raw query.
3. **Citation-enforced generation** — the LLM is instructed to follow every factual claim with `[[source_id]]` using the exact IDs from the context block. After generation, `_extract_citations` resolves cited IDs against the hit map, normalising chunk-style IDs (e.g. `chk:pdf/paper:0009`) to their parent source when needed. If the LLM produces no citations despite having context, the top-scoring source hit is injected as a fallback citation.

**LLM client.** `OllamaLLMClient` calls the Ollama `/api/chat` endpoint locally — no external API key needed. The generation model is `qwen2.5:14b`; embeddings use `nomic-embed-text`. `EchoLLMClient` is a deterministic stub for tests.

**Contracts layer.** `rca/contracts/` defines the identifier rules, node/edge models, and other shared DTOs that every other layer imports. No layer other than `store` performs persistence; no persistence layer makes model calls.

**Streamlit UI.** `app.py` provides a browser interface with two modes. *Research Chat* accepts natural-language questions and streams grounded answers with inline citation cards and a `✓ grounded` / `⚠ unverified` badge. *Research Workspace* has three tabs: Ingest (drag-and-drop PDF upload, single or batch), Knowledge Map (interactive Plotly/NetworkX graph of source nodes and edges), and Store (searchable list of all ingested items). The sidebar shows live paper/chunk counts and supports dark/light theme toggling.

---

## Storage layout

```
.rca/
├── vectors/          # ChromaDB persistent store
└── graph.sqlite3     # Document graph, FTS index, metadata
```

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| Embeddings | nomic-embed-text (Ollama) | Local, fast, strong retrieval quality |
| Generation | qwen2.5:14b (Ollama) | Best local model for structured grounded generation |
| Vector DB | ChromaDB | Persistent, zero-infrastructure prototype |
| Graph/metadata | SQLite | Zero-dependency, portable, easy to audit |
| UI | Streamlit | Prototype interface — FastAPI migration planned |
| Orchestration | Custom flows + LangGraph-ready | Modular, testable, evolvable |

---

## Evaluation

RCA is evaluated against **30 golden Q&A pairs** spanning 6 source papers, covering easy, medium, and hard questions across schema knowledge, method detail, quantitative results, error analysis, and cross-paper reasoning.

### Current results (run 2026-03-13)

| Metric | Value |
|---|---|
| Grounded rate | 100% |
| Citation precision | 83.3% |
| Avg keyword hit rate | 0.154 |
| Avg latency | 17.1 s |

**Breakdown by difficulty:**

| Difficulty | Count | Grounded | Citation precision | Keyword hit rate |
|---|---|---|---|---|
| Easy | 6 | 100% | 83.3% | 0.447 |
| Medium | 15 | 100% | 86.7% | 0.083 |
| Hard | 9 | 100% | 77.8% | 0.076 |

**Retrieval ablations — hit@5 / hit@10 (n=30):**

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + FTS | 80.0% | 96.7% |
| 3. vector + FTS + expansion | 80.0% | 96.7% |
| 4. full pipeline (+ rewrite) | 90.0% | 93.3% |

### What the numbers mean

- **83.3% citation precision** — two fixes applied: (1) `_extract_citations` code bug where chunk IDs bypassed parent resolution (26.7% → 73.3%); (2) golden.json typo where `vision_language` should be `vision-language` in 3 sgvl expected_source fields (73.3% → 83.3%).
- **96.7% hit@10 on plain vector search** — the correct source is almost always in the index and retrievable; the bottleneck is rank position, not embedding quality.
- **Query rewriting improves hit@5 (+10pp)** but slightly hurts hit@10 (-3.3%) by drifting on queries with specific named entities.
- **1 irreducible miss: jampacker-001** — "two main components of JamPacker" fails at top-10 under all strategies. Chunks describe components by function, not by name; the rewriter produces generic terms.
- **Low keyword hit rate** — answers cite the right source but LLM paraphrases rather than quoting, missing specific technical terms. This is an answer quality issue, not a retrieval issue.

### Known failure modes

| Failure class | Description |
|---|---|
| Named-entity rewrite drift | Rewriter replaces specific system/scene names with generic terms |
| Rank position sensitivity | Correct chunk exists in top-10 but scores below top-5 cutoff |
| Function-vs-name chunk mismatch | Paper describes components by function; query asks by name (jampacker-001) |
| LLM paraphrase | Answer cites correctly but misses specific numerical/keyword content |

---

## Roadmap

### v1 — Local system (current)

- [x] PDF ingest → chunking → dual-store (vector + graph)
- [x] Hybrid retrieval (vector + FTS + graph expansion)
- [x] Query rewriting before retrieval
- [x] Grounded answer generation with citation enforcement
- [x] Streamlit UI (chat + ingest + knowledge map + store)
- [x] Integration tests
- [x] Evaluation harness with golden Q&A pairs
- [ ] **Fix citation precision** — source-ID resolution bug (priority)
- [ ] **Retrieval ablations** — vector-only vs hybrid vs graph vs rewrite
- [ ] **Expand golden set** — 30 → 100+ pairs
- [ ] Observability — per-stage latency, token usage, retrieval provenance
- [ ] arxiv MCP server
- [ ] Zotero MCP server
- [ ] Docker + one-command local boot
- [ ] Weekly digest generator

### v2 — Production-shaped deployment

- FastAPI backend (replace Streamlit)
- Next.js frontend
- Background ingest worker (async)
- Object storage for raw PDFs (S3 or equivalent)
- Postgres + pgvector (replace ChromaDB in cloud deployment)
- Structured logs + metrics dashboard
- GitHub Actions CI (lint + tests + smoke deploy)
- Docker Compose
- Optional auth / multi-user namespaces

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), [ripgrep](https://github.com/BurntSushi/ripgrep)

```bash
brew install ripgrep

# Pull models
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull qwen2.5:14b        # answer generation and query rewriting

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `RCA_WORKSPACE_ROOT` | current directory | Root of the research workspace exposed to the filesystem server |
| `RCA_DATA_DIR` | `.rca` | Directory where all runtime data is stored |
| `RCA_CHUNK_SIZE` | `1200` | Target chunk size in characters |
| `RCA_CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `RCA_GENERATION_MODEL` | `qwen2.5:14b` | Ollama model used for answer generation and query rewriting |
| `RCA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model used for vector embeddings |
| `ANONYMIZED_TELEMETRY` | `False` | Disable ChromaDB telemetry |

Runtime directories under `.rca/` are created automatically on first use.

> **Always use `uv run python`.** Never bare `python` — the system Python lacks ChromaDB and silently falls back to the JSON backend, returning 0 documents.

---

## Usage

**Launch the UI**

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Use *Research Chat* to ask questions about ingested papers. Use *Workspace → Ingest* to drag-and-drop PDFs directly from Finder.

**Ingest documents**

```bash
uv run python -m rca.cli ingest path/to/papers/
```

Or programmatically:

```python
from rca.flows.ingest_flow import IngestFlow

flow = IngestFlow()
result = flow.ingest_path("papers/some-robotics-paper.pdf")

print(f"source_id : {result.source_id}")
print(f"chunks    : {len(result.chunk_ids)}")
print(f"nodes     : {result.node_count}")
print(f"edges     : {result.edge_count}")
```

**Ask a question and get a grounded answer**

```python
from rca.flows.generate_flow import GenerateFlow

g = GenerateFlow()
result = g.generate_answer("What methods are used for robotic bin packing?")

print("Grounded:", result.grounded)
for c in result.citations:
    print(f"  [{c.source_id}] {c.title}")
print(result.answer)
```

**Query the retrieval layer directly**

```python
from rca.flows.retrieve_flow import RetrieveFlow

r = RetrieveFlow()
bundle = r.retrieve("structured perception inventory generation pipeline", limit=10)

for hit in bundle.hits:
    print(f"[{hit.score:.3f}] {hit.node_id}")
    print(hit.excerpt[:200])
    print()
```

**Run tests**

```bash
uv run pytest                                          # all tests
uv run pytest tests/integration/test_generate_flow.py # generate pipeline only
uv run pytest tests/integration/test_ingest_flow.py   # ingest pipeline only
```

**Run evaluation harness**

```bash
uv run python eval/harness.py
```

---

## Project structure

```
.
├── app.py                      # Streamlit UI — Research Chat and Workspace
├── .streamlit/
│   └── config.toml             # Theme configuration (base: dark)
├── rca/                        # Main package
│   ├── config/                 # Settings (pydantic-settings, RCA_ env prefix) and tool policies
│   ├── contracts/              # Shared data models: node/edge types, ID rules, citations, artifacts
│   ├── store/                  # Persistence only: GraphStore (SQLite), VectorStore (ChromaDB), EventLog
│   │   └── migrations/         # SQL schema applied on GraphStore init
│   ├── extractors/             # Turn files into text payloads: PDF, Markdown, Git, Experiment
│   ├── flows/                  # Compose extractors and stores into workflows
│   │   ├── ingest_flow.py      # File → chunks → graph + vector store
│   │   ├── retrieve_flow.py    # Vector + lexical search, graph expansion, scored bundles
│   │   └── generate_flow.py    # Query rewriting → retrieval → grounded LLM answer
│   ├── llm/                    # LLM client interface: OllamaLLMClient, EchoLLMClient
│   ├── orchestrator/           # LangGraph state machine and routing (skeletal)
│   ├── telemetry/              # Tracing and metrics instrumentation (skeletal)
│   └── mcp_servers/
│       ├── filesystem/         # MCP server: sandboxed file access and ripgrep search
│       ├── experiments/        # MCP server: experiment run CRUD over SQLite
│       ├── arxiv/              # Reserved — not implemented
│       ├── zotero/             # Reserved — not implemented
│       └── git/                # Reserved — not implemented
├── cli/                        # Entry points: rca-ingest, rca-query, rca-digest
├── eval/                       # Golden question set, evaluation harness, and run results
├── tests/
│   ├── unit/                   # Contract and store unit tests
│   └── integration/            # End-to-end ingest and generate flow tests
└── docs/                       # Architecture, data model, and evaluation documentation
```

---

## Project context

Built alongside a Master's thesis at Instituto Superior Técnico on *Structured Perception for Packing-Relevant Inventory Generation* — a system that generates machine-readable grocery inventories from RGB images for robotic bagging. RCA serves as the research memory layer: ingesting related papers, tracking design decisions, and enabling grounded retrieval over the full literature corpus.
