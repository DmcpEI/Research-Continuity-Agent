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
| Citation precision | 83.3% |
| Avg keyword hit rate | 0.154 |
| Avg latency | 17.1 s |

Retrieval ablations — hit@5 / hit@10 (n=30):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + FTS | 80.0% | 96.7% |
| 3. vector + FTS + expansion | 80.0% | 96.7% |
| 4. full pipeline (+ rewrite) | 90.0% | 93.3% |

### Known failure modes
- **jampacker-001**: 1 irreducible semantic miss — chunks for "two main components of JamPacker" don't surface even at top-10. Query rewriter produces generic terms. Needs re-chunking or title-biased retrieval.
- LLM generating plausible but wrong answers from off-topic chunks (keyword hit rate remains low)
- Query rewriter drifts on queries containing specific named entities (scene names, system names)

---

## v1 remaining — complete before tagging v1.0.0

Priority order:

### P0 — Fix citation precision ✅ 26.7% → 83.3%
- [x] Debug source-ID resolution — strip `:NNNN` suffix, verify parent `src:` lookup
- [x] Re-run harness — citation precision 73.3% after code fix
- [x] Fix golden.json typo — sgvl expected_source had `vision_language` (underscore) vs stored `vision-language` (hyphen). All 3 sgvl cases were false negatives. Citation precision 73.3% → 83.3%.
- [ ] Add per-chunk provenance logging to retrieval stage

### P1 — Retrieval ablations ✅ completed 2026-03-13

hit@5 / hit@10 across 30 golden pairs (after all fixes):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + FTS | 80.0% | 96.7% |
| 3. vector + FTS + expansion | 80.0% | 96.7% |
| 4. full pipeline (+ rewrite) | 90.0% | 93.3% |

**Findings:**
- FTS and graph expansion add zero marginal hit@k over vector-only at any k. All strategies miss the same cases.
- Query rewriting improves hit@5 (+10pp: 80% → 90%) by promoting relevant chunks in ranking. It slightly hurts hit@10 (-3.3%: 96.7% → 93.3%) by drifting on 1 query.
- **96.7% hit@10** with plain vector search means the correct source exists in the index and is retrievable — the bottleneck is rank position, not embedding quality. Increasing retrieval window from 5 → 10 recovers most failures.
- **1 irreducible miss: jampacker-001.** "What are the two main components of JamPacker?" fails at top-10 under all strategies. Rewriter produces `"JamPacker architecture components problem solving techniques optimization performance enhancement"` — generic, loses specificity. Root cause: those chunks describe the components by function, not by name — the phrasing mismatch persists even with keyword expansion.
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
- [x] Docker + docker-compose for one-command local boot (`Dockerfile`, `docker-compose.yml`)
- [x] Makefile / task runner — `make run`, `make eval`, `make ablations`, `make test`
- [ ] Seeded demo dataset (5-10 papers pre-ingested)
- [ ] GitHub Actions: lint + tests on push
- [ ] SQLite schema migrations
- [ ] arxiv MCP server — pull papers by ID or search directly into RCA
- [ ] Zotero MCP server — sync Zotero library into RCA automatically

### Tag v1.0.0 when:
- Citation precision > 70% ✅ (83.3%)
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

**With numbers (v1 current):**
> Citation precision improved from 26.7% → 83.3% across two fixes: chunk-to-source ID resolution bug and golden.json expected_source typo. Retrieval ablations (n=30): hit@5 80% vector-only, 90% with query rewrite; hit@10 96.7% across all strategies. One irreducible semantic miss (jampacker-001). Median latency ~17s. 25 papers / 1826 chunks indexed.

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
