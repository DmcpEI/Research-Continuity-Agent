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

## Current state (v1.2.1 — 2026-03-16)

### What works
- [x] Project scaffold (uv, pydantic settings, stable ID system)
- [x] MCP servers — filesystem + experiments
- [x] Knowledge store — SQLite graph + ChromaDB vector store
- [x] PDF extractor + boundary-aware chunker
- [x] Ingest flow — PDF → chunks → graph nodes + vector embeddings
- [x] Retrieve flow — hybrid vector + FTS5 lexical search + graph expansion
- [x] Generate flow — grounded answer generation with citation enforcement
- [x] Query rewriter — LLM rewrites natural language to dense keywords before retrieval
- [x] Citation resolution — chunk IDs resolved to source IDs
- [x] Streamlit UI — chat + workspace (ingest, knowledge map, store)
- [x] Integration tests (pytest, 52 passing)
- [x] Evaluation harness with 100 golden Q&A pairs

### Measured performance

| Metric | Value |
|---|---|
| Citation precision (answerable, non-abstained) | 92.1% over 89 cases |
| Negative abstention recall | 2/10 (20.0%) |
| Overall grounded rate | 97.0% |
| Avg keyword hit rate | 0.259 |
| Avg latency | 11.3 s |

Retrieval ablations — hit@5 / hit@10 (n=90 answerable):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 0. fts5-only (BM25 baseline) | 95.6% | 98.9% |
| 1. vector-only (dense baseline) | 76.7% | 88.9% |
| 2. vector + keyword (FTS5) | 76.7% | 88.9% |
| 3. vector + keyword + expansion | 94.4% | 96.7% |
| 4. full pipeline (+ query rewrite) | 96.7% | 98.9% |

### Known failure modes
- Citation selection drift on a small set of cases (`jampacker-003`, `pic2-010`, `review-002`, `review-003`, `stablebinpacking-002`, `vilain-001`)
- Vector-only retrieval still under-rank a few questions that source expansion recovers
- LLM generating plausible but wrong answers from off-topic chunks (keyword hit rate remains low)
- Cross-paper comparison questions remain the hardest retrieval slice even after rewrite improvements
- Abstention remains under-calibrated even after reranking: answerable citation precision improved, but the system still only rejects `2/10` negative questions

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

### P1 — Retrieval baselines and lexical migration ✅ completed 2026-03-15

hit@5 / hit@10 across 60 answerable golden pairs (historical 65-question snapshot):

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 0. fts5-only (BM25 baseline) | 95.0% | 98.3% |
| 1. vector-only (dense baseline) | 76.7% | 91.7% |
| 2. vector + keyword (FTS5) | 76.7% | 91.7% |
| 3. vector + keyword + expansion | 95.0% | 96.7% |
| 4. full pipeline (+ query rewrite) | 95.0% | 98.3% |

**Findings:**
- FTS5/BM25 was strong enough in ablation that it replaced the earlier token-wise `LIKE` lexical stage on the production path.
- A held-out coefficient sweep then tuned the reranker from `(title=0.12, text=0.04)` to `(title=0.12, text=0.05)`, lifting config 3 from `93.3%` to `95.0%` at hit@5.
- The original `LIKE` implementation is still retained as `search_nodes_like()` for reference and regression testing because it documents the design history and provides a simple fallback baseline.
- Query rewriting is still mixed on the expanded set, but the final reranker now lets the full pipeline match the pure FTS5 baseline on both hit@5 and hit@10.
- The current retrieval story is stronger than before: graph expansion remains the main lift over dense retrieval, and cross-encoder reranking closes the remaining gap to the BM25-only baseline.
- Full results in `eval/results/ablations.json`.

Remaining:
- [ ] Diagnose why pure FTS5-only still beats the composed retrieval pipeline after the lexical migration
- [ ] Measure: chunk size sensitivity (256 / 512 / 1024 tokens)
- [ ] Measure: embedding model sensitivity (nomic vs alternatives)

### P2 — Expand evaluation
- [x] Grow golden set: 30 → 100 pairs across more papers
  Current validated set is 100 questions. This expansion did not require `OPENAI_API_KEY`; API access is optional and separate from eval-set design.
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

### Recently resolved GitHub issues
- [x] `#1` `embed()` NotImplementedError — fixed by implementing `OllamaLLMClient.embed()`
- [x] `#3` Source node stored full document text — fixed by storing a bounded preview instead
- [x] `#4` `_expand_to_sources` edge-kind bug — fixed by filtering to `contains`
- [x] `#5` `VectorStore` silent degradation — fixed by explicit warning logging on Chroma fallback

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

### v1.1.0 tag criteria (historical):
- Citation precision > 70% ✅ (88.5% over answerable, non-abstained cases)
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
> Built a RAG system with ChromaDB, SQLite, Ollama, Streamlit, MCP.

**Good:**
> Built a local-first research knowledge system that ingests technical PDFs, performs hybrid retrieval over semantic and structured links, and generates grounded answers with source citations. Designed an evaluation harness with golden queries, retrieval ablations, and failure analysis for citation errors and hallucination modes.

**With numbers (historical 65-question snapshot):**
> Citation precision is 92.9% over 56 answerable, non-abstained cases on the 65-question harness, with negative abstention recall at 3/5. Retrieval baselines (n=60 answerable): hit@5 95.0% FTS5/BM25, 76.7% dense-only, 76.7% vector + FTS5, 95.0% with graph expansion, and 95.0% with the full rewrite + rerank pipeline. The current codebase now uses a 100-question corpus, so these numbers should be treated as a dated benchmark snapshot.

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
| Direct flow orchestration | Orchestration is handled directly via `RetrieveFlow` and `GenerateFlow`; a LangGraph-based agent workflow stays on the roadmap until it earns its complexity |

---

## Strategic Priorities (from v1.1.0 review)

These priorities were identified through external review of v1.1.0. They are ordered by thesis credibility impact, not implementation complexity.

🔴 High priority

- Eval set expansion — current 30-question golden set is too small for thesis credibility. Target: 60–100 questions, stratified into buckets: factual lookup, proper noun/paper title lookup, paraphrased terminology, contribution/method questions, comparison, synthesis across chunks, negative/unanswerable queries, citation-sensitive questions, hard lexical mismatch cases. Report results per bucket, not just aggregate.
- Unanswerable query handling — system currently always answers. Need basic abstention/grounding check and negative queries in golden set. A research agent that confidently answers with insufficient evidence is a thesis credibility risk.

🟡 Medium priority

- Coefficient tuning justification — completed once for the current reranker, and any future coefficient changes should continue to use a held-out sweep rather than manual eyeballing so the choice stays defensible to a thesis examiner.
- Index/state versioning — eval output should record what ingest schema version, chunking version, embedding model, and graph build version produced the current store. Prevents ambiguity like "did this metric come from current code against stale graph state?"
- Chroma fallback visibility in eval — a warning log is not enough. Eval runs should detect and flag if retrieval ran on JSON fallback instead of Chroma, since metrics are not comparable across backends.

🟢 Later roadmap

- Reranking — shipped in the current pipeline as a cross-encoder final stage. Future work is now about improving calibration and whether section-aware or paper-aware reranking beats the current chunk-level setup.
- Reciprocal rank fusion — rank-based late fusion as a cleaner alternative to score-scale-dependent merging.
- Query-type-aware retrieval — lightweight routing: proper noun queries → stronger lexical/title bias; conceptual queries → stronger semantic retrieval.
- Section-aware chunk weighting — weight abstract/title/conclusion chunks differently based on question type.

⚠️ Orchestration wording to keep honest

- Orchestration is handled directly via `RetrieveFlow` and `GenerateFlow`. A LangGraph-based agent workflow is on the roadmap, not part of the current implementation.

---

## Expert Review — Action Items (post v1.1.0)

Action items from independent expert review of v1.1.0. Ordered by impact on thesis credibility and portfolio value. Items marked 🔴 should be resolved before thesis submission or adding this project to a CV.

🔴 Critical — before thesis defence / CV:

- [x] Abstention / grounding detection — a two-gate abstention check now exists, but it is still under-calibrated. The original v1.1.0 review snapshot was `1/5` negative recall with `8` answerable abstentions on the 65-question run; the next step is still calibrated confidence scoring or a dedicated abstention classifier.
- [x] Expand generation evaluation to the full 65-question set — this milestone is complete historically, and the corpus has since grown further to 100 questions.
- [x] Explicit baselines added to evaluation — config 0 is now an FTS5/BM25 lexical baseline and config 1 is labeled explicitly as the dense baseline. All five retrieval configs are reported in one table.
- [x] FTS5 investigation completed — FTS5/BM25 measured better than the earlier `LIKE` path and has now replaced it on the production lexical path.
- [x] Coefficient sweep completed — held-out validation justified raising `text_weight` from `0.04` to `0.05` while keeping `title_weight` at `0.12`.
- Keep the repository lean — placeholder-only files and dead helpers should be deleted or clearly deprecated. Every retained file should have a clear current role in the ingest, retrieval, generation, eval, or deployment path.
- Keep orchestration claims honest — orchestration is handled directly via `RetrieveFlow` and `GenerateFlow` today. A LangGraph-based agent workflow remains a roadmap option, not shipped functionality.

🟡 Important — before thesis submission:

- [x] Coefficient sweep — completed with a held-out split. The current lexical reranker moved from `(title=0.12, text=0.04)` to `(title=0.12, text=0.05)` after a `+5.3pp` held-out hit@5 gain.
- [x] README architecture diagram + one-command eval — the README now includes a Mermaid system diagram plus explicit `uv run python eval/run_ablations.py` and `uv run python eval/harness.py` reproduction commands.
- Question independence — still preferable to have at least 10 golden questions written by a human external reviewer (labmate, advisor, reviewer) to reduce self-bias. This is separate from whether an API key is available.
- [x] API backend config option — `rca/llm/client.py` now supports configurable `RCA_LLM_BASE_URL` / `RCA_LLM_API_KEY` for OpenAI-compatible endpoints while keeping the default Ollama path unchanged.
- [x] Confidence intervals on per-category metrics — per-category ablation output now reports 95% Wilson confidence intervals and flags small-sample categories with explicit caution.

🟢 Roadmap — after core gaps are closed:

- Query-type-aware routing — rewrite already helps some proper-noun queries and hurts some paraphrase queries. Add a lightweight classifier or rule-based router and measure the delta.
- Cross-encoder reranking — shipped. The next question is whether chunk-level reranking should be supplemented with source-level grouping or section-aware weighting to reduce repeated-paper dominance in the final bundle.
- Eval question independence at scale — grow to roughly 80–100 questions total, with at least 10 authored externally.
