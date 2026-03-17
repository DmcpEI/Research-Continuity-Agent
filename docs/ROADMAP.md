# RCA Roadmap

## North star

Build a measured, testable, production-shaped research knowledge system that demonstrates:
- evaluated retrieval quality
- grounded generation with citation precision
- strong observability and failure analysis
- practical agent/tool orchestration
- clear deployment and architectural tradeoffs

Target audience: AI/ML, orchestration, and agent-systems roles in robotics-adjacent teams. The project is about the reasoning and memory layer, not robot control.

---

## Current state (v1.2.2 — 2026-03-17)

RCA today is a local-first research knowledge system with:
- dual-store ingest into SQLite graph data plus ChromaDB embeddings
- hybrid retrieval using FTS5/BM25, dense search, source expansion, and cross-encoder reranking
- grounded answer generation with citation enforcement and QueryTrace observability
- Streamlit chat/workspace UI plus MCP servers for filesystem and experiment logs
- a 100-question evaluation corpus and a local CI baseline of Ruff plus pytest

The current implementation still orchestrates queries directly through `RetrieveFlow` and `GenerateFlow`. An explicit agent loop is planned, not shipped.

---

## Current metrics

| Metric | Value |
|---|---|
| Eval corpus | `100` questions (`90` answerable, `10` negative) |
| Retrieval, full pipeline | `96.7%` hit@5 / `98.9%` hit@10 |
| Citation precision | `92.1%` over `89` answerable, non-abstained cases |
| Negative abstention recall | `2/10` (`20.0%`) |
| Average latency | `11.3 s` |
| Tests | `52` passing |

Detailed methodology, artifacts, and caveats live in [EVAL.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/docs/EVAL.md). System structure lives in [ARCHITECTURE.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/docs/ARCHITECTURE.md). The user-facing overview stays in [README.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/README.md).

---

## Now

These are the highest-priority milestones for reliability and correctness.

- **Confidence-calibrated abstention** — Replace the current threshold heuristic with a lightweight classifier over retrieval features such as max score, score spread, query length, and hits above threshold. Rationale: `2/10` abstention recall is the clearest current correctness gap.
- **Human-authored external eval subset** — Add a small independently written subset to the golden set. Rationale: it reduces self-bias in external reporting better than adding more self-authored questions.
- **Retrieval auditability in eval** — Expose retrieved hit lists and per-chunk provenance in harness output, then add a compact failure taxonomy. Rationale: it will make ranking-versus-generation errors easier to explain and defend.

---

## Next

These are the next capability-building milestones once the correctness gaps above are addressed.

- **MCP agent loop** — Build a planning agent that receives a research question, decides which MCP tools to call across the filesystem, knowledge base, and experiment logs, executes them, and synthesizes a response with a full observable trace in the UI. Rationale: it is the biggest missing agent-systems capability in the current architecture and has strong portfolio value.
- **Document versioning and idempotent re-ingest** — Track document revisions and make re-ingest safe for already indexed sources. Rationale: it improves operational discipline and makes the system more realistic for long-running research use.
- **Research connectors** — Add `arxiv` and `Zotero` MCP ingestion paths. Rationale: they reduce manual ingest work and make the eventual agent loop materially more useful.

---

## Later

These are worthwhile, but they should follow the correctness and workflow milestones above.

- **AWS deployment** — Containerize and deploy RCA on AWS, most likely via ECS first and possibly Lambda + API Gateway for narrower serving paths. Rationale: it demonstrates cloud deployment competence and removes the assumption of a fully local setup.
- **API backend parity** — Extend the current OpenAI-compatible chat backend support into full backend parity, including embedding-side and deployment-facing configuration. Rationale: local-first should stay a deliberate tradeoff rather than an Ollama-only constraint.
- **Production-shaped backend split** — Move from the current Streamlit-first packaging toward a cleaner FastAPI plus frontend split with a cloud-friendly store story. Rationale: it is the right path if RCA needs to become a more deployable multi-surface system rather than a local-only research tool.

---

## Phase 2 platform path

This preserves the earlier next-phase direction in a shorter form.

- **FastAPI backend** — Split the current UI-first packaging into a cleaner API service boundary.
- **Next.js frontend** — Replace the Streamlit-only surface if RCA evolves into a more polished multi-page application.
- **Background ingest worker** — Move long-running ingest and indexing off the request path.
- **Object storage for raw documents** — Store source PDFs outside the local filesystem when deploying remotely.
- **Postgres + pgvector** — Replace the local deployment store story with a more production-friendly unified backend.
- **Structured logs and metrics** — Add service-level observability beyond the current per-query traces.
- **Optional auth and namespaces** — Support multi-user or shared deployment scenarios without making them a local-default requirement.

---

## Shipped milestones

### Core system

- [x] Project scaffold with stable IDs and settings management
- [x] PDF, note, and experiment ingest flows with boundary-aware chunking
- [x] SQLite graph store and ChromaDB vector store
- [x] Direct retrieval and generation flows
- [x] Filesystem and experiment MCP servers
- [x] Streamlit chat and workspace UI

### Retrieval and grounding

- [x] FTS5/BM25 lexical migration
- [x] Exact-word lexical rescoring to remove partial-word false positives
- [x] Source expansion hardening
- [x] Cross-encoder reranking
- [x] Query-type-aware rewrite gating
- [x] Append-only query rewrite expansion instead of query replacement
- [x] Citation resolution from chunk IDs to source IDs

### Evaluation and observability

- [x] 100-question golden set with `90/10` answerable-negative split
- [x] Stratified dev/test splits
- [x] Retrieval ablations and coefficient sweep
- [x] QueryTrace stage timings, token usage, and retrieval provenance
- [x] Per-query trace export under `eval/results/traces/`

### Tooling and delivery

- [x] Docker, docker-compose, and Makefile-based local boot
- [x] Configurable OpenAI-compatible chat backend support in the LLM client
- [x] GitHub Actions CI workflow added; local Ruff and pytest baseline verified

---

## Guardrails

- RCA is currently a research memory and grounded QA system, not a robot controller.
- Orchestration claims should stay honest: direct flow composition is shipped; an agent loop is future work.
- Local-first remains the default. Cloud deployment and API backends are optional extensions, not the core identity of the project.
