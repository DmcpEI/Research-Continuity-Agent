# Changelog

All notable changes to RCA. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow `pyproject.toml`. Entries before 1.5.0 live in git history.

## [Unreleased]

### Added
- Harness failure-stage taxonomy (`retrieval_miss`, `context_miss`, `false_abstention`, `citation_miss`, `false_answer`) with per-case retrieved/context sources and abstention per `negative_type` (4e87a4f). Affects eval artifacts only.
- Production-path retrieval ablation (config 5) mirroring `GenerateFlow`; refreshed metrics: 96.7% hit@5 / 98.9% hit@10 on 90 answerable questions (d609169).
- Retrieval-feature abstention calibration scripts `eval/collect_retrieval_features.py` and `eval/calibrate_abstention.py` (0058f36).
- 29 typed negative questions (`neg-011`–`neg-039`); golden set is now 129 questions with 39 negatives (639c60b).

### Changed
- Grounded chat abstains before the LLM call when the top cross-encoder rerank score is ≤ −3.618 (label `low_rerank_score`) (e9fd29d). Affects chat users and eval results: clearly out-of-corpus questions now return a refusal faster. The threshold is tied to `cross-encoder/ms-marco-MiniLM-L-6-v2`; recalibrate if the reranker changes.

## [1.5.0] - 2026-09-29

### Added
- Document revisions and idempotent re-ingest keyed on file and content hashes; additive `source_revisions` table, no migration needed for existing databases (446f9d5).
- FastAPI backend for the React frontend: sources with revisions, ingest, chat, agent, model selection with tool-capability probing, inline PDF (07be586).
- Generation failure labels and grounding check; harness `--model`, `--rewrite-model`, `--strict-model-check`, `--external-golden-path` (68716b5).

### Changed
- Uncited answers are no longer marked grounded by injecting the top source as a citation (68716b5).
- Default generation model is `gemma3:12b`; agent model `qwen2.5:14b` (07be586).

### Fixed
- Frontend tests on Node 25+, where the built-in `localStorage` shadowed jsdom's (3f3cc7f).
