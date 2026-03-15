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
> Built a RAG system with ChromaDB, SQLite, Ollama, Streamlit, MCP.

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
| Direct flow orchestration | Orchestration is handled directly via `RetrieveFlow` and `GenerateFlow`; a LangGraph-based agent workflow stays on the roadmap until it earns its complexity |

---

## Strategic Priorities (from v1.1.0 review)

These priorities were identified through external review of v1.1.0. They are ordered by thesis credibility impact, not implementation complexity.

🔴 High priority

- Eval set expansion — current 30-question golden set is too small for thesis credibility. Target: 60–100 questions, stratified into buckets: factual lookup, proper noun/paper title lookup, paraphrased terminology, contribution/method questions, comparison, synthesis across chunks, negative/unanswerable queries, citation-sensitive questions, hard lexical mismatch cases. Report results per bucket, not just aggregate.
- Unanswerable query handling — system currently always answers. Need basic abstention/grounding check and negative queries in golden set. A research agent that confidently answers with insufficient evidence is a thesis credibility risk.

🟡 Medium priority

- Coefficient tuning justification — future changes to scoring coefficients (e.g. title weight) must include a small sweep (e.g. 0.15, 0.18, 0.20, 0.25) rather than manual eyeballing, so the chosen value is defensible to a thesis examiner.
- Index/state versioning — eval output should record what ingest schema version, chunking version, embedding model, and graph build version produced the current store. Prevents ambiguity like "did this metric come from current code against stale graph state?"
- Chroma fallback visibility in eval — a warning log is not enough. Eval runs should detect and flag if retrieval ran on JSON fallback instead of Chroma, since metrics are not comparable across backends.

🟢 Later roadmap

- Reranking — highest retrieval leverage after observability is in place. Improves top-5 precision and reduces generic-token bleed better than coefficient tuning.
- Reciprocal rank fusion — rank-based late fusion as a cleaner alternative to score-scale-dependent merging.
- Query-type-aware retrieval — lightweight routing: proper noun queries → stronger lexical/title bias; conceptual queries → stronger semantic retrieval.
- Section-aware chunk weighting — weight abstract/title/conclusion chunks differently based on question type.

⚠️ Orchestration wording to keep honest

- Orchestration is handled directly via `RetrieveFlow` and `GenerateFlow`. A LangGraph-based agent workflow is on the roadmap, not part of the current implementation.

---

## Expert Review — Action Items (post v1.1.0)

Action items from independent expert review of v1.1.0. Ordered by impact on thesis credibility and portfolio value. Items marked 🔴 should be resolved before thesis submission or adding this project to a CV.

🔴 Critical — before thesis defence / CV:

- Abstention / grounding detection — the system currently defaults to answering whenever retrieval returns context, and `GenerateFlow` can inject a fallback citation when the model emits none. The current `100% grounded` figure therefore overstates evidence sufficiency. Add a retrieval-confidence check (for example, mean top-3 score) and return "insufficient evidence" with `grounded=False` below threshold. Tune on the 5 negative questions and report abstention precision / recall.
- Expand generation evaluation to the full 65-question set — `eval/golden.json` and retrieval ablations now cover 65 questions, but the last reported generation metrics in docs/results are still from the older 30-question run. Regenerate answer-level metrics on the full set so retrieval and generation sections are directly comparable.
- Add explicit baselines to evaluation — current hybrid hit@5 numbers on the expanded set lack a pure lexical baseline. Keep vector-only config 1 as an explicit baseline and add a lexical-only baseline (ideally FTS5/BM25 if implemented, otherwise a clearly documented current lexical baseline). Report all baselines and hybrid configs in one table.
- FTS5 investigation — the current lexical layer in `GraphStore.search_nodes()` is token-wise SQL `LIKE`, which is the most obvious retrieval-design question an examiner will ask about. Either implement FTS5 and compare it empirically, or document a concrete rationale for staying with `LIKE` and what FTS5 would and would not change.
- Keep the repository lean — placeholder-only files and dead helpers should be deleted or clearly deprecated. Every retained file should have a clear current role in the ingest, retrieval, generation, eval, or deployment path.
- Keep orchestration claims honest — orchestration is handled directly via `RetrieveFlow` and `GenerateFlow` today. A LangGraph-based agent workflow remains a roadmap option, not shipped functionality.

🟡 Important — before thesis submission:

- Coefficient sweep — lexical weights are still hand-tuned. Run a grid over title weight and text weight, evaluate on a held-out split, and show the selected values are near-optimal rather than ad hoc.
- README architecture diagram + one-command eval — add a compact system diagram and ensure `uv run python eval/run_ablations.py` reproduces the reported numbers from a clean state.
- Question independence — have at least 10 golden questions written by someone else (labmate, advisor, reviewer) to reduce self-bias in the eval set.
- API backend config option — add an OpenAI-compatible base-URL / model configuration path in `rca/llm/client.py` so local-first is clearly a deliberate deployment choice, not a hard product limitation.
- Confidence intervals on per-category metrics — several categories have very small `n`, so per-category hit rates should either include confidence intervals or be explicitly caveated as small-sample results.

🟢 Roadmap — after core gaps are closed:

- Query-type-aware routing — rewrite already helps some proper-noun queries and hurts some paraphrase queries. Add a lightweight classifier or rule-based router and measure the delta.
- Cross-encoder reranking — retrieve top-20, rerank, and take top-5. This is the most plausible next lever for improving contribution and cross-paper retrieval quality.
- Eval question independence at scale — grow to roughly 80–100 questions total, with at least 10 authored externally.
