# Research Continuity Agent

A personal research intelligence system built for a thesis on structured perception for robotic grocery bagging. It ingests research papers, experiments, code, and notes into a queryable knowledge graph, connecting them with semantic relationships so the researcher can query their own research the way you'd query a codebase. The system is retrieval-first: every piece of content receives a stable deterministic identifier, a graph node, and a vector embedding before any generation step is considered.

---

## Architecture

The system is organized into three functional layers.

```
┌─────────────────────────────────────────────┐
│              MCP Tool Layer                 │
│   filesystem server  │  experiments server  │
│   (ripgrep search)   │  (SQLite CRUD)       │
└─────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────┐
│            Agent / Flow Layer               │
│   IngestFlow  │  RetrieveFlow  │  Flows...  │
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

**Contracts layer.** `rca/contracts/` defines the identifier rules, node/edge models, and other shared DTOs that every other layer imports. No layer other than `store` performs persistence; no persistence layer makes model calls.

---

## Status

The following components are implemented and have been exercised against real inputs.

| Component | State |
|---|---|
| Filesystem MCP server — sandboxed path resolution, `list_directory`, `read_text_file` (1 MB limit), ripgrep `search_text` | Working |
| Experiments MCP server — `record_run`, `list_runs`, `update_run`, `get_run`, status lifecycle | Working |
| `GraphStore` — `upsert_node`, `upsert_edge`, `get_node`, `list_edges`, `search_nodes` (LIKE on title/text) | Working |
| `VectorStore` — ChromaDB backend with Ollama embeddings, cosine distance, JSON fallback | Working |
| `IngestFlow` — PDF → text extraction → boundary-aware chunking → graph nodes + vector embeddings | Working |
| Semantic retrieval returning similarity scores above 0.8 on real robotics-domain queries | Working |

The following are scaffolded (files exist, interfaces are defined) but not yet integrated end-to-end:

- `RetrieveFlow` — chunk results are not yet expanded to parent source nodes or connected experiments
- Note and experiment extractors are wired into `IngestFlow` dispatch but have not been tested on real inputs at the same level as the PDF path
- `orchestrator/` — LangGraph graph and routing logic are skeletal
- `llm/` — client and tool wrappers are present but not connected to a live model call
- `telemetry/` — tracing and metrics modules exist but are not instrumented across flows
- MCP servers for `git`, `zotero`, and `arxiv` are reserved directories with no implementation

---

## Setup

**Requirements**

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — used for environment and dependency management
- [ripgrep](https://github.com/BurntSushi/ripgrep) — required by the filesystem MCP server
- [Ollama](https://ollama.com) — required for vector embeddings

```bash
brew install ripgrep
```

Install Ollama from [ollama.com](https://ollama.com), then pull the embedding model:

```bash
ollama pull nomic-embed-text
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
| `ANONYMIZED_TELEMETRY` | `False` | Disable ChromaDB telemetry |

Runtime directories under `.rca/` are created automatically on first use.

---

## Usage

**Ingest a PDF**

```python
from rca.flows.ingest_flow import IngestFlow

flow = IngestFlow()
result = flow.ingest_path("papers/some-robotics-paper.pdf")

print(f"source_id : {result.source_id}")
print(f"chunks    : {len(result.chunk_ids)}")
print(f"nodes     : {result.node_count}")
print(f"edges     : {result.edge_count}")
```

**Query the vector store**

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

The CLI entry points (`rca-ingest`, `rca-query`, `rca-digest`) are registered in `pyproject.toml` but are thin wrappers that have not been independently validated beyond the Python API above.

---

## Project Structure

```
.
├── rca/                        # Main package
│   ├── config/                 # Settings (pydantic-settings, RCA_ env prefix) and tool policies
│   ├── contracts/              # Shared data models: node/edge types, ID rules, citations, artifacts
│   ├── store/                  # Persistence only: GraphStore (SQLite), VectorStore (ChromaDB), EventLog
│   │   └── migrations/         # SQL schema applied on GraphStore init
│   ├── extractors/             # Turn files into text payloads: PDF, Markdown, Git, Experiment
│   ├── flows/                  # Compose extractors and stores into ingest, retrieve, generate workflows
│   ├── orchestrator/           # LangGraph state machine and routing (skeletal)
│   ├── llm/                    # LLM client interface and tool definitions (skeletal)
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
│   └── integration/            # End-to-end ingest flow tests
└── docs/                       # Architecture, data model, and evaluation documentation
```

---

## Roadmap

Planned phases in order:

1. **Retrieve flow with graph expansion** — semantic chunk retrieval followed by traversal to parent source nodes and connected experiment nodes, returning a structured context bundle rather than raw chunks.
2. **Note and experiment extractors validated** — wire Markdown notes and JSON/YAML experiment records into the ingest flow with the same test coverage as the PDF path.
3. **arxiv and Zotero MCP servers** — pull papers directly from arxiv by ID and sync a Zotero library into the graph.
4. **LangGraph orchestrator** — replace the skeletal graph with a working state machine that routes queries through retrieve → generate with enforced tool policy.
5. **Weekly digest generator** — produce a structured summary of recent ingest activity with citations grounded in graph node IDs, not free-form text.
6. **Evaluation harness** — run the golden question set against the retrieval pipeline and track precision/recall across commits.
