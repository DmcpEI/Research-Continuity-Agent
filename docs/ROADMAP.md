# RCA Roadmap

## North star

Build a **measured, testable, production-shaped research knowledge system** that demonstrates:
- Evaluated retrieval quality (not vibes)
- Robust ingestion with failure handling
- Grounded generation with citation precision
- Observability and deployment discipline
- Clear architectural tradeoffs — not fashionable component assembly

Target audience: **agent/orchestration/AI systems roles** in DACH robotics and ML companies (Munich/Zurich, 2026). Not hardware robotics — the brain layer, not the arm.

---

## Current state (v1.1.0 — 2026-03-14)

### What works
- [x] Project scaffold (uv, pydantic settings, stable ID system)
- [x] MCP servers — filesystem + experiments
- [x] Knowledge store — SQLite graph + ChromaDB vector store
- [x] PDF extractor + boundary-aware chunker
- [x] Ingest flow — PDF → chunks → graph nodes + vector embeddings
- [x] Retrieve flow — hybrid vector + keyword search + graph expansion
- [x] Generate flow — grounded answer generation with citation enforcement
- [x] Query rewriter — LLM rewrites natural language to dense keywords before retrieval
- [x] Citation resolution — chunk IDs resolved to source IDs
- [x] Streamlit UI — chat + workspace (ingest, knowledge map, store)
- [x] Integration tests (pytest, 0.90s)
- [x] Evaluation harness with 30 golden Q&A pairs

### Measured performance

| Metric | Value |
|---|---|
| Grounded rate | 100% |
| Citation precision | 86.7% |
| Avg keyword hit rate | 0.189 |
| Avg latency | 16.8 s |

Retrieval ablations — hit@5 / hit@10 (n=30):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + keyword search | 80.0% | 96.7% |
| 3. vector + keyword + expansion | 96.7% | 100.0% |
| 4. full pipeline (+ query rewrite) | 100.0% | 100.0% |

### Known failure modes
- Citation selection drift on a small set of cases (`pic2-010`, `review-002`, `review-003`, `stablebinpacking-002`)
- Vector-only and keyword-only retrieval still under-rank a few questions that source expansion or rewrite recover
- LLM generating plausible but wrong answers from off-topic chunks (keyword hit rate remains low)
- Query rewriter drifts on queries containing specific named entities (scene names, system names)

---

## Post-v1.1.0 backlog

Priority order:

### v1.1.0 shipped
- [x] Debug source-ID resolution — strip `:NNNN` suffix, verify parent `src:` lookup
- [x] Re-run harness — citation precision 73.3% after code fix
- [x] Fix golden.json typo — sgvl expected_source had `vision_language` (underscore) vs stored `vision-language` (hyphen). All 3 sgvl cases were false negatives. Citation precision 73.3% → 83.3%.
- [x] Harden retrieval/citation ranking — lexical scoring + source expansion + rewrite sanitizer lifted precision to 86.7%.
- [x] Remove partial-word false positives in title/text rescoring — exact word-token matching restored stable citation precision and pushed full-pipeline retrieval to 100.0% hit@5.
- [x] QueryTrace observability — per-stage query latency, hit provenance, token usage, and eval trace export.

### P0 — Remaining citation/observability work
- [ ] Add per-chunk provenance logging to retrieval stage
- [ ] Expose retrieved hit lists in harness results for easier ranking-vs-generation audits
- [ ] Add deterministic fallback for no-answer / low-confidence citation cases

### P1 — Retrieval ablations ✅ completed 2026-03-14

hit@5 / hit@10 across 30 golden pairs (current):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + keyword search | 80.0% | 96.7% |
| 3. vector + keyword + expansion | 96.7% | 100.0% |
| 4. full pipeline (+ query rewrite) | 100.0% | 100.0% |

**Findings:**
- Keyword search alone adds no lift over vector-only — same cases miss at all k. The bottleneck is embedding quality for specific queries when lexical terms don't appear verbatim in retrieved chunks.
- Exact-word lexical rescoring + source expansion lift hit@5 by +16.7pp (80% → 96.7%) and achieve 100% hit@10. Removing partial-word false positives mattered as much as adding source expansion.
- Query rewriting adds the final +3.3pp to hit@5 (96.7% → 100.0%) while preserving 100% hit@10.
- The remaining harness misses are generation-side citation-selection problems, not full-pipeline retrieval misses.
- Full results in `eval/results/ablations.json`.

Remaining:
- [ ] Measure: chunk size sensitivity (256 / 512 / 1024 tokens)
- [ ] Measure: embedding model sensitivity (nomic vs alternatives)

### P2 — Expand evaluation
- [ ] Grow golden set: 30 → 100+ pairs across more papers
- [ ] Add failure taxonomy to `docs/EVAL.md`:
  - query classes that fail
  - chunking failures
  - citation mismatch patterns
  - rewrite-induced drift
  - hallucination modes
- [ ] Per-category breakdown in harness output

### P3 — Observability follow-ups
- [x] Per-stage query latency logging (rewrite / retrieve / generate)
- [x] Token usage tracking
- [x] Retrieval hit provenance (vector / lexical / expansion)
- [x] Export per-query traces to `eval/results/traces/`
- [ ] Extend traces to ingest / embed stages
- [ ] Answer rejection rate (grounded=False cases)

### Open GitHub issues
- [ ] `#1` `embed()` NotImplementedError — broken interface contract
- [ ] `#3` Full document text stored on source node — performance / storage tradeoff
- [ ] `#4` `_expand_to_sources` should filter by `contains` edge kind — correctness bug
- [ ] `#5` `VectorStore` silent degradation when Chroma fails — operational hazard

### P4 — Ingestion robustness
- [ ] Handle malformed / scanned PDFs gracefully
- [ ] Idempotent re-ingest (skip already-indexed docs)
- [ ] Document versioning (re-ingest updated paper)
- [ ] Metadata normalization (author, year, venue)

### P5 — Packaging
- [x] Docker + docker-compose for one-command local boot (`Dockerfile`, `docker-compose.yml`)
- [x] Makefile / task runner — `make run`, `make eval`, `make ablations`, `make test`
- [ ] Seeded demo dataset (5-10 papers pre-ingested)
- [ ] GitHub Actions: lint + tests on push
- [ ] SQLite schema migrations
- [ ] arxiv MCP server — pull papers by ID or search directly into RCA
- [ ] Zotero MCP server — sync Zotero library into RCA automatically

### Tag v1.1.0 when:
- Citation precision > 70% ✅ (86.7%)
- Retrieval ablation table published ✅
- Docker one-command boot working ✅
- All integration tests passing

---

## v2 — Production-shaped deployment

**Framing: productionization, not migration. Same core `rca/` package.**

### Architecture changes

| Component | v1 | v2 |
|---|---|---|
| API | Streamlit (monolith) | FastAPI (decoupled) |
| Frontend | Streamlit | Next.js |
| Vector store | ChromaDB | Postgres + pgvector |
| Ingest | Synchronous | Background worker (async queue) |
| Document storage | Local filesystem | Object storage (S3 / R2) |
| Metadata store | SQLite | Postgres |
| Logs / metrics | stdout | Structured logs + dashboard |
| Auth | None | Optional JWT / API key |
| Deployment | Local | Dockerized, cloud-deployable |
| CI | None | GitHub Actions |

### What NOT to add
- SageMaker — no model training/serving here, Ollama is the inference layer
- Lambda — no clear event-driven use case yet
- Full AWS sprawl — one cloud path with cost rationale, not a service catalog

### Good cloud story for v2
- FastAPI + Dockerized services
- Object storage for raw PDFs
- Postgres + pgvector as the unified store
- Structured logs → CloudWatch or Grafana
- CI with tests + lint + smoke deploy
- Infrastructure diagram with cost-conscious rationale

---

## Branching strategy

```
main              ← stable, always working
v2/migrate        ← v2 active development
```

```bash
# Tag v1.1.0 before starting v2
git tag v1.1.0 -m "v1.1.0: retrieval ranking hardening and eval-backed docs"
git push origin v1.1.0

# Start v2
git checkout -b v2/migrate
```

---

## What this project is and is not

### Is
- A research memory and retrieval system for an agent
- A testbed for hybrid retrieval + grounded generation
- Evidence of: orchestration, memory systems, LLM failure mode understanding, evaluation discipline, full-stack ownership

### Is not
- A perception system
- A robot controller
- A generic PDF chatbot
- A service catalog exercise

### How to describe it

**Bad:**
> Built a RAG system with ChromaDB, SQLite, Ollama, Streamlit, LangGraph, MCP.

**Good:**
> Built a local-first research knowledge system that ingests technical PDFs, performs hybrid retrieval over semantic and structured links, and generates grounded answers with source citations. Designed an evaluation harness with golden queries, retrieval ablations, and failure analysis for citation errors and hallucination modes.

**With numbers (v1 current):**
> Citation precision improved from 26.7% → 86.7% across the citation-resolution fix, golden-set correction, and later retrieval/ranking hardening. Retrieval ablations (n=30): hit@5 80.0% vector-only, 96.7% with graph expansion, and 100.0% with the full pipeline; hit@10 reaches 100.0% once source expansion is enabled. The remaining failures are citation-selection cases, not full-pipeline retrieval misses. Average latency is ~16.8s. 25 papers / 1826 chunks indexed.

---

## Decision log

| Decision | Rationale |
|---|---|
| Local-first (Ollama) | Privacy, cost control, offline use, research-environment fit |
| ChromaDB v1 | Zero-infrastructure prototype, persistent, sufficient for <10k chunks |
| Postgres+pgvector v2 | Production-friendly, single unified store, better for cloud deployment |
| SQLite graph | Structured traversal, provenance, zero external dependency |
| Skip note extractor | User doesn't take structured notes; Notion logs are mentor-facing, not knowledge |
| Skip SageMaker | No model training; Ollama covers inference entirely |
| LangGraph-ready (not LangGraph) | Premature to commit; modular flows already support it when needed |
