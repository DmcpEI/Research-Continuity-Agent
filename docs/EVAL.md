# Evaluation

RCA is evaluated at two layers:
- retrieval quality: does the correct paper surface in the top-k bundle?
- generation quality: does the final answer cite the correct source, use retrieved evidence, and abstain when the corpus does not support the question?

This document reflects the current **100-question** evaluation corpus and the current eval scripts in the repo.

---

## Golden Set

The active golden set is [eval/golden.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/golden.json).

- Total questions: `100`
- Answerable: `90`
- Negative / unanswerable: `10`
- Difficulties: `9 easy`, `48 medium`, `43 hard`

Coverage includes:
- single-paper factual lookup
- method and contribution questions
- paraphrase / lexical mismatch cases
- cross-paper comparison
- multi-chunk synthesis
- explicit negative / unsupported queries

Metric definitions:
- `citation_precision`: fraction of answerable, non-abstained cases where every expected source appears in the returned citation IDs
- `abstention_recall`: fraction of negative questions where the system abstains correctly
- `grounded_rate`: fraction of all cases where the model returned a cited answer; for thesis reporting, the more meaningful view is answerable questions answered with citation
- `keyword_hit_rate`: fraction of `expected_keywords` matched by case-insensitive substring in the generated answer
- `hit@k`: retrieval-only metric indicating whether the expected source appears in the top-k resolved retrieval hits

---

## Generation Harness

```bash
uv run python eval/harness.py
```

The harness runs [GenerateFlow.generate_answer()](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/generate_flow.py) over all `100` questions and records:
- `grounded`
- `abstained`
- `citations`
- `source_correct`
- `keyword_hits`
- `latency_ms`

It writes a run artifact to `eval/results/run_<timestamp>.json` and per-question traces to `eval/results/traces/<run_id>/`.

Important caveat:
- harness results depend on the live local generation backend
- if Ollama or another configured endpoint is unavailable, the run is not comparable to a normal local run
- after changing the corpus, the right source of truth is a fresh local rerun, not an older checked-in artifact

Current local run:
- [run_20260316T191249Z.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/run_20260316T191249Z.json)

Headline metrics:

| Metric | Value |
|---|---|
| Overall harness coverage | `100` questions |
| Answerable questions | `90` |
| Negative questions | `10` |
| Citation precision (answerable, non-abstained) | `91.0%` over `89` cases |
| Negative abstention recall | `2/10` (`20.0%`) |
| Answerable abstentions | `1` |
| Average keyword hit rate | `0.265` |
| Average latency | `11.8 s` |

---

## Retrieval Ablations

```bash
uv run python eval/run_ablations.py
```

The ablation runner evaluates retrieval only on the `90` answerable questions and skips the `10` negatives.

It writes the aggregate artifact to `eval/results/ablations.json`.

Current local run:
- [ablations.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/ablations.json)

Aggregate retrieval results:

| Configuration | hit@5 | hit@10 |
|---|---:|---:|
| 0. fts5-only (BM25 baseline) | `95.6%` | `98.9%` |
| 1. vector-only (dense baseline) | `76.7%` | `88.9%` |
| 2. vector + keyword (FTS5) | `76.7%` | `88.9%` |
| 3. vector + keyword + expansion | `94.4%` | `96.7%` |
| 4. full pipeline (+ rewrite) | `87.8%` | `96.7%` |

Interpretation guidance:
- FTS5/BM25 is the production lexical backbone and should be treated as the main sparse baseline
- dense retrieval remains useful, but on this corpus the lexical signal is often very strong
- source expansion and the cross-encoder reranker are the main reasons the composed pipeline improves on simpler hybrids
- query rewrite remains mixed and should be reported honestly as such; on the current 100-question set it lowers hit@5 relative to the expansion-only variant

---

## Coefficient Sweep

```bash
uv run python eval/run_coefficient_sweep.py
```

The held-out coefficient sweep tunes the lexical reranker used before merge and expansion.

Current split behavior:
- split files live in [eval/splits/dev.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/splits/dev.json) and [eval/splits/test.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/splits/test.json)
- the split is stratified by category with `random.seed(42)`
- the script now scales the held-out size with the corpus instead of hardcoding the original `45 / 20`
- for the current 100-question corpus, the checked-in split is `69` dev / `31` test

As with the other eval scripts, rerun locally when the corpus or retrieval backend changes.

Note:
- the coefficient-sweep script itself was fixed to track the current retriever signature and current corpus size
- if the sweep is run in a sandboxed environment that forces JSON fallback, do not treat those numbers as comparable to the live Chroma/Ollama runs above

---

## Current Status

What is true today, independent of any single artifact:
- the golden corpus is now `100` questions
- the split files cover that full corpus exactly
- the eval schema tests pass against the updated corpus and split files
- abstention remains heuristic and is still one of the main open weaknesses
- a human-authored external subset would still be stronger for thesis bias reduction than self-authored or model-authored additions

Known active failure themes:
- citation selection drift on otherwise relevant retrieval bundles
- unsupported questions whose retrieved context still looks plausible enough to evade abstention
- detail-level abstentions on answerable questions
- rewrite-induced drift on some paraphrase and named-entity queries

---

## Practical Reporting Guidance

For thesis writeups and portfolio material:
- report corpus size, answerable/negative counts, and split sizes explicitly
- state which backend produced the metrics
- avoid comparing runs produced under different retrieval backends, especially JSON fallback versus Chroma
- treat human-external question authorship as a separate bias-reduction property from simple corpus size growth
