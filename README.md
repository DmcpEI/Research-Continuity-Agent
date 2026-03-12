# Research Continuity Agent

A personal research intelligence system built for a thesis on structured perception for robotic grocery bagging. It ingests research papers, experiments, code, and notes into a queryable knowledge graph, connecting them with semantic relationships so the researcher can query their own research the way you'd query a codebase. The system is retrieval-first: every piece of content receives a stable deterministic identifier, a graph node, and a vector embedding before any generation step is considered.

---

## Architecture

The system is organized into four functional layers.

```
┌─────────────────────────────────────────────┐
│             Streamlit UI (app.py)           │
│   Research Chat  │  Workspace  │  Ingest    │
└─────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────┐
│              MCP Tool Layer                 │
│   filesystem server  │  experiments server  │
│   (ripgrep search)   │  (SQLite CRUD)       │
└─────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────┐
│            Agent / Flow Layer               │
│   IngestFlow  │  RetrieveFlow  │  GenerateFlow  │
│   Extractors: PDF, Note, Git, Experiment    │
└─────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────┐
│           Knowledge Store Layer             │
│   GraphStore (SQLite)  │  VectorStore       │
│   nodes + edges        │  (ChromaDB/Ollama) │
└─────────────────────────────────────────────┘
```

**Knowledge store.** Two persistent stores back the system. `GraphStore` holds typed nodes and directed edges in a local SQLite database. Node kinds are `source`, `chunk`, `note`, `paper`, `experiment`, and `digest`; edge kinds are `contains`, `derived_from`, `references`, `cites`, `related_to`, and `produced_by`. `VectorStore` wraps ChromaDB with an Ollama embedding function using `nomic-embed-text` (768 dimensions, cosine similarity). When ChromaDB is unavailable, it falls back to a JSON file with bag-of-words cosine scoring.

**ID system.** All identifiers are stable and deterministic. Source nodes follow the pattern `src:namespace/name` (e.g., `src:pdf/attention-is-all-you-need`). Chunk nodes derive from their source: `chk:namespace/name:0000`. Identifiers are validated with regular expressions at the contract boundary so invalid IDs cannot enter the stores.

**MCP servers.** Two MCP servers expose tools over stdio. The filesystem server sandboxes all path resolution to a configured root directory and delegates text search to ripgrep. The experiments server provides full CRUD for experiment runs (record, list, update, get) with a status lifecycle of `pending → running → complete / failed`, backed by a separate SQLite database.

**Ingest flow.** `IngestFlow.ingest_path` dispatches on file type: `.pdf` → `PDFExtractor`, `.md`/`.txt` → `NoteExtractor`, `.json`/`.yaml` → `ExperimentExtractor`, directory → `GitExtractor`. Extracted text is split into boundary-aware chunks (default 1200 characters, 150 overlap) that prefer paragraph breaks, then newlines, then word boundaries before hard-cutting. Each chunk becomes a graph node linked to its source via a `contains` edge, and all chunks are upserted into the vector store in a single batch call.

**Retrieve flow.** `RetrieveFlow.retrieve` composes vector similarity search with lexical graph search (`LIKE` on title/text). Results are merged by node ID, then `_expand_to_sources` follows `contains` edges from chunk hits to surface parent source nodes. The final bundle contains up to 10 ranked hits (chunks + expanded sources) and the associated graph edges. Default retrieval limit is 10.

**Generate flow.** `GenerateFlow.generate_answer` is a three-step pipeline:
1. **Query rewriting** — the user's natural-language question is sent to the LLM to produce a dense 8–12 keyword technical search query, improving vector recall over conversational phrasing.
2. **Grounded context** — the rewritten query is passed to `RetrieveFlow`; hits scoring above 0.55 (or any `src:` node) are formatted into a bracketed context block. If the rewritten query yields no context, the pipeline retries with the original raw query.
3. **Citation-enforced generation** — the LLM is instructed to follow every factual claim with `[[source_id]]` using the exact IDs from the context block. After generation, `_extract_citations` resolves cited IDs against the hit map, normalising chunk-style IDs (e.g. `chk:pdf/paper:0009`) to their parent source when needed. If the LLM produces no citations despite having context, the top-scoring source hit is injected as a fallback citation.

**LLM client.** `OllamaLLMClient` calls the Ollama `/api/chat` endpoint locally — no external API key needed. The generation model is `qwen2.5:14b`; embeddings use `nomic-embed-text`. `EchoLLMClient` is a deterministic stub for tests.

**Contracts layer.** `rca/contracts/` defines the identifier rules, node/edge models, and other shared DTOs that every other layer imports. No layer other than `store` performs persistence; no persistence layer makes model calls.

**Streamlit UI.** `app.py` provides a browser interface with two modes. *Research Chat* accepts natural-language questions and streams grounded answers with inline citation cards and a `✓ grounded` / `⚠ unverified` badge. *Research Workspace* has three tabs: Ingest (drag-and-drop PDF upload, single or batch), Knowledge Map (interactive Plotly/NetworkX graph of source nodes and edges), and Store (searchable list of all ingested items). The sidebar shows live paper/chunk counts and supports dark/light theme toggling.

---

## Status

The following components are implemented and have been exercised against real inputs.

| Component | State |
|---|---|
| Filesystem MCP server — sandboxed path resolution, `list_directory`, `read_text_file` (1 MB limit), ripgrep `search_text` | Working |
| Experiments MCP server — `record_run`, `list_runs`, `update_run`, `get_run`, status lifecycle | Working |
| `GraphStore` — `upsert_node`, `upsert_edge`, `get_node`, `list_edges`, `search_nodes` (LIKE on title/text) | Working |
| `VectorStore` — ChromaDB backend with Ollama `nomic-embed-text` embeddings, cosine distance, JSON fallback | Working |
| `IngestFlow` — PDF → text extraction → boundary-aware chunking → graph nodes + vector embeddings | Working |
| `RetrieveFlow` — vector + lexical fusion, graph expansion to parent source nodes, scored hit bundles | Working |
| `GenerateFlow` — LLM query rewriting, grounded answer generation, citation extraction and enforcement, chunk→source ID resolution, no-citation fallback | Working |
| `OllamaLLMClient` — local generation via `qwen2.5:14b`, no API key | Working |
| Streamlit UI — Research Chat, Workspace (Ingest/Map/Store tabs), dark/light theme, drag-and-drop PDF upload | Working |
| Semantic retrieval returning similarity scores above 0.8 on real robotics-domain queries | Working |
| Integration tests — ingest flow and generate flow end-to-end | Working |

The following are scaffolded (files exist, interfaces are defined) but not yet integrated end-to-end:

- Note and experiment extractors are wired into `IngestFlow` dispatch but have not been tested on real inputs at the same level as the PDF path
- `orchestrator/` — LangGraph graph and routing logic are skeletal
- `telemetry/` — tracing and metrics modules exist but are not instrumented across flows
- MCP servers for `git`, `zotero`, and `arxiv` are reserved directories with no implementation

---

## Setup

**Requirements**

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — used for environment and dependency management
- [ripgrep](https://github.com/BurntSushi/ripgrep) — required by the filesystem MCP server
- [Ollama](https://ollama.com) — required for vector embeddings and answer generation

```bash
brew install ripgrep
```

Install Ollama from [ollama.com](https://ollama.com), then pull both models:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull qwen2.5:14b        # answer generation and query rewriting
```

**Install dependencies**

```bash
uv venv && uv sync
```

**Configure environment**

```bash
cp .env.example .env
```

Edit `.env` as needed. The key variables are:

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

> **Important:** always run Python through `uv run python` (or `uv run pytest`), not the system `python`. The system interpreter may lack `chromadb` and will silently fall back to the JSON vector backend, producing 0 vector hits.

---

## Usage

**Launch the UI**

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Use *Research Chat* to ask questions about ingested papers. Use *Workspace → Ingest* to drag-and-drop PDFs directly from Finder.

**Ingest a PDF programmatically**

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

**Query the vector store directly**

```python
from rca.store.vector_store import VectorStore
from rca.config.settings import get_settings

settings = get_settings()
store = VectorStore(settings.vector_dir, settings.default_collection)

results = store.query("structured perception for robotic manipulation", limit=5)
for r in results:
    print(f"[{r.score:.3f}] {r.id}")
    print(r.document[:300])
    print()
```

**Run tests**

```bash
uv run pytest                                          # all tests
uv run pytest tests/integration/test_generate_flow.py # generate pipeline only
uv run pytest tests/integration/test_ingest_flow.py   # ingest pipeline only
```

---

## Project Structure

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
├── eval/                       # Golden question set and evaluation harness
├── tests/
│   ├── unit/                   # Contract and store unit tests
│   └── integration/            # End-to-end ingest and generate flow tests
└── docs/                       # Architecture, data model, and evaluation documentation
```

---

## Roadmap

| Phase | Status |
|---|---|
| Retrieve flow with graph expansion — semantic chunk retrieval + traversal to parent source nodes | ✅ Done |
| GenerateFlow — LLM query rewriting, grounded generation, citation enforcement | ✅ Done |
| Streamlit UI — Research Chat and Workspace with drag-and-drop ingest | ✅ Done |
| Note and experiment extractors validated at the same level as the PDF path | Pending |
| arxiv and Zotero MCP servers — pull papers directly and sync a Zotero library | Pending |
| LangGraph orchestrator — replace skeletal graph with working state machine | Pending |
| Weekly digest generator — structured summary of recent ingest activity with grounded citations | Pending |
| Evaluation harness — run golden question set against retrieval pipeline, track precision/recall | Pending |
