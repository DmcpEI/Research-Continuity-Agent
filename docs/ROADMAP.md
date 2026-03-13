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

## Current state (v1 — 2026-03-13)

### What works
- [x] Project scaffold (uv, pydantic settings, stable ID system)
- [x] MCP servers — filesystem + experiments
- [x] Knowledge store — SQLite graph + ChromaDB vector store
- [x] PDF extractor + boundary-aware chunker
- [x] Ingest flow — PDF → chunks → graph nodes + vector embeddings
- [x] Retrieve flow — hybrid vector + FTS + graph expansion
- [x] Generate flow — grounded answer generation with citation enforcement
- [x] Query rewriter — LLM rewrites natural language to dense keywords before retrieval
- [x] Citation resolution — chunk IDs resolved to source IDs
- [x] Streamlit UI — chat + workspace (ingest, knowledge map, store)
- [x] Integration tests (pytest, 1.48s)
- [x] Evaluation harness with 30 golden Q&A pairs

### Measured performance

| Metric | Value |
|---|---|
| Grounded rate | 100% |
| Citation precision | 73.3% |
| Avg keyword hit rate | 0.164 |
| Avg latency | 16.2 s |

### Known failure modes
- Wrong-source retrieval on hard / cross-paper queries → citation precision ceiling at ~55% on hard
- Query rewriter producing generic keywords on some question types
- Score threshold (0.55) excluding valid chunks in hard retrievals
- LLM generating plausible but wrong answers from off-topic chunks

---

## v1 remaining — complete before tagging v1.0.0

Priority order:

### P0 — Fix citation precision ✅ 26.7% → 73.3%
- [x] Debug source-ID resolution — strip `:NNNN` suffix, verify parent `src:` lookup
- [x] Re-run harness — citation precision now 73.3% (target was >70%)
- [ ] Add per-chunk provenance logging to retrieval stage

### P1 — Retrieval ablations ✅ completed 2026-03-13

hit@5 across 30 golden pairs:

| Configuration | hit@5 |
|---|---|
| 1. vector-only | 73.3% |
| 2. vector + FTS | 73.3% |
| 3. vector + FTS + expansion | 73.3% |
| 4. full pipeline (+ rewrite) | 70.0% |

**Findings:**
- FTS and graph expansion add zero marginal hit@5 over vector-only. All three miss on the same cases (jampacker, sgvl papers).
- Query rewriting slightly *hurts* (-3.3%). The rewriter drifts on specific queries (e.g. "SelfCheckoutMedium4 scene" becomes generic robotics keywords), dropping the jampacker-002 and pic2-007 hits.
- **Retrieval ceiling is index quality, not search strategy.** jampacker and sgvl papers fail under all configs — their chunks are not close to these queries in embedding space. Root cause: chunking or low embedding density for those papers.
- Full results in `eval/results/ablations.json`.

Remaining:
- [ ] Measure: chunk size sensitivity (256 / 512 / 1024 tokens)
- [ ] Measure: embedding model sensitivity (nomic vs alternatives)
- [ ] Document results in `docs/EVAL.md` with real numbers

### P2 — Expand evaluation
- [ ] Grow golden set: 30 → 100+ pairs across more papers
- [ ] Add failure taxonomy to `docs/EVAL.md`:
  - query classes that fail
  - chunking failures
  - citation mismatch patterns
  - rewrite-induced drift
  - hallucination modes
- [ ] Per-category breakdown in harness output

### P3 — Observability
- [ ] Per-stage latency logging (ingest / embed / retrieve / rewrite / generate)
- [ ] Token usage tracking
- [ ] Retrieval hit provenance (which store returned each chunk)
- [ ] Answer rejection rate (grounded=False cases)
- [ ] Export to structured logs in `eval/results/`

### P4 — Ingestion robustness
- [ ] Handle malformed / scanned PDFs gracefully
- [ ] Idempotent re-ingest (skip already-indexed docs)
- [ ] Document versioning (re-ingest updated paper)
- [ ] Metadata normalization (author, year, venue)

### P5 — Packaging
- [ ] Docker + docker-compose for one-command local boot
- [ ] Seeded demo dataset (5-10 papers pre-ingested)
- [ ] Makefile / task runner
- [ ] GitHub Actions: lint + tests on push
- [ ] SQLite schema migrations
- [ ] arxiv MCP server — pull papers by ID or search directly into RCA
- [ ] Zotero MCP server — sync Zotero library into RCA automatically

### Tag v1.0.0 when:
- Citation precision > 70%
- Retrieval ablation table published
- Docker one-command boot working
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
# Tag v1 before starting v2
git tag v1.0.0 -m "v1: Streamlit + local Ollama + SQLite + ChromaDB"
git push origin v1.0.0

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

**With numbers (target for v1.0.0):**
> Citation precision improved from 26.7% → 73.3% after fixing chunk-to-source ID resolution. Retrieval ablations show hybrid search outperforms vector-only by Y% on keyword hit rate. Median latency ~16s. N documents / M chunks indexed.

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
