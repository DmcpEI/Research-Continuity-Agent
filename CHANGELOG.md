# Changelog

All notable changes to RCA. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow `pyproject.toml` and are tagged `vX.Y.Z` (1.1–1.4 were never tagged). Entries before 1.5.0 live in git history.

## [Unreleased]

### Security
- The filesystem MCP server's `search_text` passed the model-supplied pattern to ripgrep as a positional argument, so a pattern such as `--pre=CMD` became an rg flag and ran `CMD` on every file in the sandbox. A prompt-injected document could therefore trigger command execution through the agent. The pattern now follows `-e` and the path follows `--`. Upgrade recommended for anyone running agent mode.

### Added
- One-command public demo: `docker compose up --build demo` (or `make demo`) checks host Ollama, downloads five CC BY 4.0 arXiv papers listed in `demo/papers.json`, ingests them, and serves the React app plus API on `localhost:8000` (#15). New `create_site()` in `rca/api/main.py` mounts the API under `/api` and serves `frontend/dist` with an `index.html` fallback for client-side routes (`RCA_FRONTEND_DIST`). Demo data lives in Docker volumes, not `./.rca`. README gains "Results at a glance" and "Try the demo" sections. The `demo` service sits behind a compose profile, so a bare `docker compose up` still starts only Streamlit; stop it with `docker compose --profile demo down`. Also: the Streamlit `rca` compose service no longer fails when `.env` is missing, and `frontend/node_modules`, `.claude/` and `CLAUDE.md` are no longer sent to the Docker build context.
- Retrieval regression gate in the pytest job (CI and pre-push): a synthetic 12-note corpus and 27 questions run through GenerateFlow's retrieval call without Ollama (JSON vector backend, no reranker or rewrite), and the test fails if any question ranks worse than in `tests/fixtures/retrieval_gate/baseline.json`, printing hit@5, MRR, and each question whose rank moved (#8). It catches breakages of FTS5 candidates, lexical scoring, and ordering. Baseline hit@5 96.3%, MRR 0.717. Contributors: after a deliberate retrieval change, refresh with `RCA_UPDATE_RETRIEVAL_BASELINE=1 uv run pytest tests/integration/test_retrieval_gate.py` and commit the baseline diff. Does not cover rewrite, rerank, embeddings, or most query-type weights; see docs/EVAL.md.
- Agent benchmark `eval/run_agent_eval.py` with 20 sandboxed tasks (`eval/agent_tasks.json`, fixture `eval/agent_fixture/`) scoring task success, tool selection, tool precision, steps, invalid calls, and stop reasons (#13). First run with `qwen3.5:9b`: 12/20 tasks, tool selection 16/18. Eval-only; no runtime change.
- Run provenance in every eval artifact: corpus fingerprint, exact Ollama model digests and server version, reranker revision, result-affecting settings, git commit, and vector-backend state at start and end with an `index_degraded_during_run` flag (#10). Harness traces now fill `corpus_version`. Affects eval artifacts only.
- CI job for the frontend on Node 24: type-check of app and tests (`tsconfig.test.json`, tests were previously excluded from type-checking), production build, and vitest; the pre-push hook runs the same commands (`npm run typecheck`, `npm run build`, `npm test`) and stops with a clear message if `frontend/node_modules` is missing. Node version lives in `frontend/.nvmrc`. Contributors: run `npm ci` in `frontend/` once.
- Git hooks via the `pre-commit` framework: ruff check + format check on commit, the CI pytest command on push. Contributors: run `uv run pre-commit install` once per clone.
- Harness failure-stage taxonomy (`retrieval_miss`, `context_miss`, `false_abstention`, `citation_miss`, `false_answer`) with per-case retrieved/context sources and abstention per `negative_type` (4e87a4f). Affects eval artifacts only.
- Production-path retrieval ablation (config 5) mirroring `GenerateFlow`; refreshed metrics: 96.7% hit@5 / 98.9% hit@10 on 90 answerable questions (d609169).
- Retrieval-feature abstention calibration scripts `eval/collect_retrieval_features.py` and `eval/calibrate_abstention.py` (0058f36).
- 29 typed negative questions (`neg-011`–`neg-039`); golden set is now 129 questions with 39 negatives (639c60b).

### Changed
- When hit `k` would repeat a paper already shown, retrieval gives the last slot of the top `k` to the best-ranked unseen paper, so one paper's chunks are less likely to crowd out a second relevant paper (#9). Only candidates with rerank score above `-6.0` qualify (`RCA_RETRIEVAL_RESERVE_MIN_RERANK_SCORE`). Production hit@5 96.7% → 98.9% on 90 answerable questions; cross-paper questions 75% → 87.5%. Affects chat, agent search, and eval results; same-paper chunks in a 5-hit context drop from ~4.1 to ~3.8 on average. Disable with `RCA_RETRIEVAL_RESERVE_NEW_SOURCE=false`.
- Grounded chat abstains before the LLM call when the top cross-encoder rerank score is ≤ −3.618 (labels `low_rerank_score`, `low_retrieval_confidence`) (e9fd29d). Affects chat users and eval results: clearly out-of-corpus questions now return a refusal faster. The threshold is tied to `cross-encoder/ms-marco-MiniLM-L-6-v2`; recalibrate if the reranker changes.

### Fixed
- The agent passed knowledge-base source IDs (`src:...`) to filesystem tools as if they were files (#22). Filesystem tools now intercept `src:`/`chk:` paths (also in `[[...]]` form) and return guidance to answer from the search excerpts and cite the source; the knowledge-base tool description and system prompt say source IDs are citations. Redirected calls, and the existing `.pdf` redirect, now report status `error`, so the agent benchmark counts them as invalid calls (previously they looked successful). Tool-call arguments that are valid JSON but not an object are rejected with a warning instead of crashing the agent run.
- Thinking models (e.g. `qwen3.5:9b`) returned their whole answer in Ollama's `message.thinking` with empty `content`, so chat answers came back empty and the agent replied "I could not produce an answer..." even after fetching the right data (#21). Ollama requests now send `think: false` by default (`RCA_LLM_THINK`, only `true` for thinking-capable models). Agent benchmark with `qwen3.5:9b`: 12/20 → 19/20 tasks, mean latency 24 s → 11 s. Also: empty final turns stop as `empty_final_answer` and undecodable tool calls with no text as `unparsed_tool_call` (new values in agent traces and eval `stopped_reasons`); HTTP errors from the LLM client now include the server's message; a rejected `think: true` is no longer misreported as missing tool support.
- A Chroma → JSON vector-store fallback is now visible: `/status` returns `vector_backend` and `warnings`, the React status bar shows a "Vector search degraded" badge (status refreshes every 30 s), and the Streamlit sidebar shows a warning. Previously it was only logged, so retrieval could silently degrade (#17, #2). API consumers: two new additive fields, no migration.
- Agent knowledge-base search now uses the same retrieve flow and vector store as chat (API and Streamlit), so a vector fallback hit by the agent shows up in `/status` and the UI. Previously the agent built its own store. `ToolRegistry` and `AgentLoop` accept an optional `retrieve_flow` (passing it together with `knowledge_base_search` or `registry` raises `ValueError`); Streamlit clears the cached agent loop whenever it rebuilds the chat flows after an ingest. Existing callers are unaffected.
- Integration tests no longer load the real cross-encoder, which made Hugging Face Hub requests and could stall the pre-push hook for over a minute on a slow network. Contributors only; reranking stays covered by a stubbed unit test.

### Removed
- Empty `rca.mcp_servers.git` and `rca.mcp_servers.zotero` placeholder packages (#14). They held no code; nothing imported them. Zotero is no longer listed as planned work.

## [1.5.0] - 2026-09-29 (tag `v1.5.0`)

### Added
- Document revisions and idempotent re-ingest keyed on file and content hashes; additive `source_revisions` table, no migration needed for existing databases (446f9d5).
- FastAPI backend for the React frontend: sources with revisions, ingest, chat, agent, model selection with tool-capability probing, inline PDF (07be586).
- Generation failure labels and grounding check; harness `--model`, `--rewrite-model`, `--strict-model-check`, `--external-golden-path` (68716b5).
- Conversational queries (chit-chat) skip retrieval, and chat prompts include prior conversation turns (68716b5).

### Changed
- Uncited answers are no longer marked grounded by injecting the top source as a citation (68716b5).
- Default generation model is `gemma3:12b`; agent model `qwen2.5:14b` (07be586).

### Fixed
- Frontend tests on Node 25+, where the built-in `localStorage` shadowed jsdom's (3f3cc7f).
