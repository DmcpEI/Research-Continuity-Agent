# Evaluation

RCA is evaluated against a golden Q&A set across two layers: retrieval quality (does the right paper surface?) and generation quality (does the answer contain correct, cited content?).

---

## Golden set

**30 Q&A pairs** across 6 source papers, covering 12 question categories and 3 difficulty levels.

| Paper | Questions | Categories |
|---|---|---|
| Robotic Grocery Bagging (PIC2) | 10 | pipeline, schema, results, error analysis, evaluation |
| Comprehensive Review of Robotized Freight Packing | 4 | contribution, findings, future work, method |
| JamPacker | 4 | contribution, method, results |
| Stable Bin Packing (Wang & Hauser) | 4 | contribution, method, results |
| SGVL (Scene Graph VLMs) | 3 | contribution, method, results |
| Multimodal Fusion Survey | 4 | scope, findings, challenges, future work |
| Cross-paper | 1 | cross-paper reasoning |

Difficulty distribution: 6 easy / 15 medium / 9 hard.

**Metric definitions:**
- **Citation precision** — fraction of cases where `expected_source` (exact string match on source ID) appears in the set of source IDs returned with the answer. Not precision@k — a binary per-question check.
- **Keyword hit rate** — fraction of `expected_keywords` found via case-insensitive substring match in the answer text.
- **hit@k** — fraction of cases where `expected_source` appears among the source IDs resolved from the top-k retrieved chunks.

---

## Generation harness

```bash
uv run python eval/harness.py
```

Measures per golden pair:
- `grounded` — whether the system flagged the answer as grounded
- `source_correct` — whether `expected_source` appears in returned citation IDs
- `keyword_hit_rate` — fraction of `expected_keywords` found (case-insensitive) in the answer
- `latency_ms` — end-to-end time including retrieval and generation

Writes full results to `eval/results/run_<timestamp>.json`.

### Results (2026-03-14, verified live run)

| Metric | Value |
|---|---|
| Grounded rate | 100% |
| **Citation precision** | **86.7%** |
| Avg keyword hit rate | 0.129 |
| Avg latency | 16.0 s |

**By difficulty:**

| Difficulty | Grounded | Citation precision | Keyword hit rate |
|---|---|---|---|
| Easy (n=6) | 100% | 100.0% | 0.331 |
| Medium (n=15) | 100% | 80.0% | 0.093 |
| Hard (n=9) | 100% | 88.9% | 0.054 |

### Progression

| Run | Change | Citation precision |
|---|---|---|
| Baseline | — | 26.7% |
| After P0 fix | Source-ID resolution bug fixed (1 line) | 73.3% |
| After golden fix | Typo in expected_source for sgvl papers | 83.3% |
| After retrieval hardening | Query rewrite cleanup + lexical/source scoring fixes | 86.7% |

---

## Retrieval ablations

```bash
uv run python eval/run_ablations.py
```

Tests 4 retrieval configurations on all 30 golden pairs. Measures hit@5 and hit@10 — whether `expected_source` appears among the top-k retrieved chunk source IDs.

### Results

| Configuration | hit@5 | hit@10 |
|---|---|---|
| 1. vector-only | 80.0% | 96.7% |
| 2. vector + keyword | 80.0% | 96.7% |
| 3. vector + keyword + expansion | 90.0% | 100.0% |
| 4. full pipeline (+ query rewrite) | 96.7% | 100.0% |

### Interpretation

**96.7% hit@10 on plain vector** — the corpus is still well-indexed, but one case remains a true semantic miss without help from graph-aware retrieval.

**Source expansion now matters** — once lexical hits are token-scored instead of flattened to a constant score, expanding chunk hits to parent `src:` nodes recovers the missing paper on several edge cases and lifts hit@5 from 80.0% to 90.0%.

**Query rewriting still adds lift** — after cleanup and salient-term retention, full rewrite raises hit@5 from 90.0% to 96.7% while preserving 100.0% hit@10.

---

## Failure analysis

### Class 1 — Source-ID resolution bug (fixed)

**Root cause:** In `generate_flow.py`, `_extract_citations` only attempted parent resolution when `hit is None`. When the LLM cited a chunk ID that was already in `hit_map`, the resolution was skipped and the raw chunk ID (`chk:pdf/paper:0042`) was stored as `Citation.source_id` instead of the parent source ID (`src:pdf/paper`). The harness comparison then always failed.

**Fix:** Remove the `hit is None` guard — parent resolution now runs unconditionally for any ID with a numeric suffix. Since `_expand_to_sources` always adds parent `src:` nodes to the retrieval bundle, the parent is reliably found.

**Impact:** 26.7% → 73.3% citation precision from a single-line change.

### Class 2 — Golden set typo (fixed)

**Root cause:** The `expected_source` for all three sgvl questions used `vision_language` (underscore) while the actual stored source ID uses `vision-language` (hyphen). All three were false negatives in the eval — the right chunks were being retrieved and cited, but the string comparison always failed.

**Fix:** Corrected the three entries in `eval/golden.json`.

**Impact:** 73.3% → 83.3% citation precision.

### Class 3 — Retrieval miss (active, 1 case)

**Affected case:** `stablebinpacking-002` — the heuristic question still misses at hit@5 under the full rewrite pipeline, although top-10 now recovers it.

### Class 4 — Citation selection miss (active, 4 cases)

**Affected cases:** `pic2-010`, `review-002`, `review-003`, `stablebinpacking-002`.

In these cases the correct paper is retrieved strongly enough for grounding, but generation still cites a different source. Retrieval is no longer the main bottleneck here; citation selection is.

### Class 5 — Answer quality miss (active)

Several questions are `source_correct=True` but `keyword_hits=0.0` — the right paper is retrieved and cited, but the answer paraphrases without including the exact technical terms tracked by `expected_keywords`.

---

## Known limitations

| Limitation | Status |
|---|---|
| 30 golden pairs is a small evaluation set | Planned expansion to 100+ |
| Keyword matching is case-insensitive substring — not semantic | Acceptable for technical terms, misses paraphrases |
| `jampacker-001` is a genuine hard retrieval failure at all depths | Documented, not yet fixed |
| Query rewriter degrades performance on precise proper-noun queries | Needs conditional rewriting logic |
| Generation latency ~16s | Acceptable for local Ollama; target <10s with top-5 retrieval |
