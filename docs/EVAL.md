# Evaluation

RCA is evaluated against a golden Q&A set across two layers: retrieval quality (does the right paper surface?) and generation quality (does the answer contain correct, cited content?).

---

## Golden set

**30 Q&A pairs** across 6 source papers, covering 5 question categories and 3 difficulty levels.

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

### Results (2026-03-13, after all fixes)

| Metric | Value |
|---|---|
| Grounded rate | 100% |
| **Citation precision** | **83.3%** |
| Avg keyword hit rate | 0.154 |
| Avg latency | 17.1 s |

**By difficulty:**

| Difficulty | Grounded | Citation precision | Keyword hit rate |
|---|---|---|---|
| Easy (n=6) | 100% | 83.3% | 0.447 |
| Medium (n=15) | 100% | 86.7% | 0.083 |
| Hard (n=9) | 100% | 77.8% | 0.076 |

### Progression

| Run | Change | Citation precision |
|---|---|---|
| Baseline | — | 26.7% |
| After P0 fix | Source-ID resolution bug fixed (1 line) | 73.3% |
| After golden fix | Typo in expected_source for sgvl papers | 83.3% |

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
| 2. vector + FTS | 80.0% | 96.7% |
| 3. vector + FTS + expansion | 80.0% | 96.7% |
| 4. full pipeline (+ query rewrite) | 90.0% | 93.3% |

### Interpretation

**96.7% hit@10 on plain vector** — the corpus is well-indexed. Coverage is not the problem; rank is. 3 of 30 questions don't find the right paper even at top-10.

**FTS and graph expansion contribute 0% lift at hit@5** — all three strategies miss on exactly the same cases. The bottleneck is embedding quality for specific queries, not the search strategy. When the right chunks are embedded well enough to rank, any strategy finds them.

**Query rewriting helps hit@5 (+10pp) but slightly hurts hit@10 (-3.3pp)** — the rewriter promotes relevant chunks higher in ranking for most queries, but drifts on queries with specific proper nouns (scene names, model names). Net effect is positive for a top-5 window, negative for a top-10 window.

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

### Class 3 — Retrieval rank miss (active)

**Affected cases:** `jampacker-003`, `jampacker-004`, `stablebinpacking-001`, `stablebinpacking-002`, `sgvl-003` — correct chunks exist in the index but rank 6–9, just below the top-5 cutoff.

**Fix:** Increasing retrieval window to top-10 recovers these. Current system uses top-5 for generation latency; top-10 is better for evaluation.

### Class 4 — Genuine semantic miss (active, 1 case)

**Affected case:** `jampacker-001` — *"What are the two main components of JamPacker?"*

The paper describes components as "Jampack" (the algorithm) and "FRM" (Fault Recovery Module) — internal brand names. The question uses "JamPacker" (the system name). The query rewriter output was: *"JamPacker architecture components problem solving techniques optimization performance enhancement"* — preserves the system name but loses the component names. No chunk in the index scores high enough under any retrieval strategy at any depth.

**Root cause:** Terminology mismatch between question vocabulary and chunk vocabulary. Not fixable with rewriting alone.

**Potential fixes:** Title-boosted retrieval (upweight chunks from papers whose title matches query terms), finer chunk granularity (abstract + section headers as separate retrievable units), or BM25 keyword matching as an additional signal.

### Class 5 — Answer quality miss (active)

Several questions are `source_correct=True` but `keyword_hits=0.0` — the right paper was retrieved and cited, but the answer paraphrases without including the specific technical terms in `expected_keywords`.

**Examples:** `pic2-001` retrieves the PIC2 report but answers "The context does not provide specific details about two local perception pipelines" despite relevant chunks being present. The LLM is being too conservative — it hedges rather than synthesising from the retrieved context.

**Root cause:** The retrieved chunk for that question (`:0035`) covers introduction-level content, not the methodology section where pipeline names appear. The specific content is in a different chunk that didn't rank in top-5.

**Fix:** Increase top-k for generation, or implement a second-pass retrieval that targets section-specific chunks when the first pass returns introductory material.

---

## Known limitations

| Limitation | Status |
|---|---|
| 30 golden pairs is a small evaluation set | Planned expansion to 100+ |
| Keyword matching is case-insensitive substring — not semantic | Acceptable for technical terms, misses paraphrases |
| `jampacker-001` is a genuine hard retrieval failure at all depths | Documented, not yet fixed |
| Query rewriter degrades performance on precise proper-noun queries | Needs conditional rewriting logic |
| Generation latency ~17s | Acceptable for local Ollama; target <10s with top-5 retrieval |
