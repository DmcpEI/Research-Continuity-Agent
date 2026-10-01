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

## Current state (v1.5.0 — 2026-09-29)

RCA today is a local-first research knowledge system with:
- dual-store ingest into SQLite graph data plus ChromaDB embeddings, with document revisions and idempotent re-ingest
- hybrid retrieval using FTS5/BM25, dense search, source expansion, and cross-encoder reranking
- grounded answer generation with citation enforcement and QueryTrace observability
- Streamlit local dev/eval UI plus MCP servers for filesystem and experiment logs
- an MCP-backed agent loop for filesystem and experiment inspection, plus a native knowledge-base search adapter
- configurable backend parity for local Ollama or OpenAI-compatible chat and embedding APIs
- a FastAPI bridge matching the React frontend API contract for status, sources, ingest, chat, agent calls, model selection, and PDF fetch
- a deployment-ready AWS demo package with baked data, ECS templates, and short-lived task scripts
- a 129-question evaluation corpus (39 typed negatives) and a local CI baseline of Ruff plus pytest

The current implementation now has two orchestration paths: direct `RetrieveFlow`/`GenerateFlow` for grounded chat, and a separate agent loop for multi-turn tool use.

UI architecture is now actively migrating from the Streamlit-first packaging toward a FastAPI backend plus a React (Vite) frontend, while keeping Streamlit available for local evaluation workflows.

---

## v1.x in progress — UI and API migration

All items in this phase are now shipped and tracked under Shipped milestones.

---

## Current metrics

| Metric | Value |
|---|---|
| Eval corpus | `129` questions (`90` answerable, `39` negative) |
| Retrieval, full pipeline | `93.3%` hit@5 / `96.7%` hit@10 |
| Citation precision | `90.8%` over `87` answerable, non-abstained cases |
| Negative abstention recall | `3/10` (`30.0%`) |
| Answerable abstentions | `3` |
| Keyword hit rate | `20.5%` |
| Grounded rate | `94.0%` |
| Average latency | `9.7 s` |
| Tests | `113` passing |

These numbers come from the 2026-04-21 `gemma3:12b` run on the original 100-question set (10 negatives), before the grounding check and failure labels landed; a fresh run is pending, together with a local model comparison (`gemma3:12b` / `gemma4:12b` / `qwen3.5:9b`).

Detailed methodology, artifacts, and caveats live in [EVAL.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/docs/EVAL.md). System structure lives in [ARCHITECTURE.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/docs/ARCHITECTURE.md). The user-facing overview stays in [README.md](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/README.md).

---

## Hardware constraints

- On an 18GB M3 Pro, `phi4:14b` sits at the practical memory limit and should be reserved for quality-first runs.
- `gemma3:12b` is the recommended daily driver for interactive speed.

---

## Now

These are the highest-priority milestones for reliability and correctness.

- **Keyword-hit improvement via rewrite prompt tuning** — Improve the rewrite prompt and retrieval-term alignment for the robotics corpus. Rationale: keyword hit rate (`20.5%`) is the clearest current retrieval weakness.
- **Confidence-calibrated abstention** — The calibrated pre-generation rerank gate (`rerank_max <= -3.618`, test `5/12` negatives at `1/28` false abstentions) is wired into `GenerateFlow`; confirm with a full harness rerun. Remaining gap: unreported-detail and false-premise questions about papers that are in the corpus, which retrieval signals cannot catch and need a generation-side check. See [EVAL.md](EVAL.md#abstention-calibration-retrieval-features).
- **Human-authored external eval subset** — Add a small independently written subset to the golden set. Rationale: it reduces self-bias in external reporting better than adding more self-authored questions.
- **Retrieval auditability in eval** — Expose retrieved hit lists and per-chunk provenance in harness output, then add a compact failure taxonomy. Rationale: it will make ranking-versus-generation errors easier to explain and defend.
- **React API verification** — Keep the new FastAPI bridge covered by contract tests while Streamlit remains the supported local demo surface. Rationale: the React frontend is now technically usable, but product polish should follow correctness work.
- **Response-structure conformance checks** — Add model-response shape checks (citation tuple style, markdown bullet noise, citation parsing reliability) to model onboarding and release gates. Rationale: multi-model support now introduces output-format variability that can degrade UI rendering and citation UX.
- **Conversation request isolation** — Proper per-conversation request isolation requires backend session management; the current UI uses a global lock to avoid cross-conversation races.

---

## Next

These are the next capability-building milestones once the correctness gaps above are addressed.

- **Research connectors** — Add `arxiv` and `Zotero` MCP ingestion paths. Rationale: they reduce manual ingest work and make the eventual agent loop materially more useful.

---

## Later

These are worthwhile, but they should follow the correctness and workflow milestones above.

- **Live AWS deployment** — Turn the current deployment-ready package into a maintained live deployment, most likely via ECS first. Rationale: it demonstrates cloud deployment competence beyond one-off demos, but it should stay optional because it carries real cost.
- **Production-shaped backend split** — Harden the FastAPI plus frontend split with a cloud-friendly store story. Rationale: the API bridge exists, but RCA still needs lifecycle, background ingest, and deployment discipline before this becomes the primary surface.

---

## Phase 2 platform path

- **Background ingest worker** — Move long-running ingest and indexing off the request path.
- **Object storage for raw documents** — Store source PDFs outside the local filesystem when deploying remotely.
- **Postgres + pgvector** — Replace the local deployment store story with a more production-friendly unified backend.
- **Structured logs and metrics** — Add service-level observability beyond the current per-query traces.
- **Optional auth and namespaces** — Support multi-user or shared deployment scenarios without making them a local-default requirement.
- **Optional SSR frontend** — Revisit Next.js only if multi-user/cloud deployment demands SSR-specific behavior.

---

## v2 execution tracks (current planning)

### Track A — Library workflow depth

- Multi-select source rows
- Bulk actions: open, tag, re-ingest, export metadata
- Library quality indicators: missing metadata, duplicate/near-duplicate hints, ingest health flags

### Track B — Model selection UX

- Pin favorite models
- Recent model history in picker
- Keep recommendation policy dynamic from each user environment with optional overrides

### Track C — Observability UX

- Query Trace improvements (latency share, clearer stage diagnostics, compact expert mode)
- Session timeline (conversation events, model switches, source opens, tool/eval events)
- Response-shape diagnostics in traces (citation parsing anomalies, markdown/structure warnings)

### Track D — Evaluation UX

- Lightweight in-app eval mode for rapid model and prompt comparison
- Exportable run summary for reproducible discussion and regression tracking

### Proposed order

1. Track A (workflow impact)
2. Track B (daily model operations)
3. Track C (debuggability)
4. Track D (measurement loop)

### v2 tag readiness criteria

- The seven planned features are shipped behind stable UX states
- Documentation updated (`README`, `ROADMAP`, architecture/eval notes)
- Baseline checks pass (lint, tests, frontend build, API smoke)
- One fresh benchmark/eval snapshot is captured and linked in release notes
- Response-structure checks pass for selected default model profile (`gemma3:12b`)

---

## Shipped milestones

### Core system

- [x] Project scaffold with stable IDs and settings management
- [x] PDF, note, and experiment ingest flows with boundary-aware chunking
- [x] Document revisions (`source_revisions`) and idempotent re-ingest keyed on file + content hashes
- [x] SQLite graph store and ChromaDB vector store
- [x] Direct retrieval and generation flows
- [x] Filesystem and experiment MCP servers
- [x] MCP agent loop with read-only filesystem and experiment inspection plus native knowledge-base search
- [x] Streamlit UI for local dev/eval workflows

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
- [x] Negative set expanded to `39` typed unanswerable questions
- [x] Stratified dev/test splits
- [x] Retrieval ablations and coefficient sweep
- [x] QueryTrace stage timings, token usage, and retrieval provenance
- [x] Per-query trace export under `eval/results/traces/`

### Tooling and delivery

- [x] Docker, docker-compose, and Makefile-based local boot
- [x] Configurable backend parity for chat, agent tool use, and embeddings across Ollama and OpenAI-compatible APIs
- [x] AWS deployment-ready demo package with baked image defaults, ECS templates, and demo/teardown scripts
- [x] GitHub Actions CI workflow added; local Ruff and pytest baseline verified

### UI and API migration

- [x] FastAPI backend for core flows (Ingest, Retrieve, Generate) exposed through stable HTTP endpoints
- [x] React (Vite) frontend migration to a local-first production UI
- [x] Information architecture consolidation (Chat default, Agent secondary, Library/Settings, right panel, status bar)

---

## Guardrails

- RCA is currently a research memory and grounded QA system, not a robot controller.
- Orchestration claims should stay honest: grounded chat uses direct flow composition; the agent loop is shipped separately for tool use.
- Local-first remains the default. Cloud deployment and API backends are optional extensions, not the core identity of the project.
- Split-model rewriting is retained for research only and is not a recommended production configuration based on current latency/quality tradeoffs.
