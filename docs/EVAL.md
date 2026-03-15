# Evaluation

RCA is evaluated at two layers:
- retrieval quality: does the correct paper surface in the top-k bundle?
- generation quality: does the final answer cite the correct source, use the retrieved evidence, and abstain when the corpus does not support the question?

This document reflects the current 65-question evaluation set and the live abstention-enabled generation harness.

---

## Golden Set

The active golden set is [eval/golden.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/golden.json).

- Total questions: `65`
- Answerable: `60`
- Negative / unanswerable: `5`
- Difficulties: `7 easy`, `28 medium`, `30 hard`

Coverage includes:
- single-paper factual lookup
- method and contribution questions
- paraphrase / lexical mismatch cases
- cross-paper comparison
- multi-chunk synthesis
- explicit negative / unsupported queries

Metric definitions:
- `citation_precision`: fraction of answerable, non-abstained cases where every `expected_source` appears in the returned citation IDs
- `abstention_recall`: fraction of negative questions where the system abstains correctly
- `grounded_rate`: for the current harness, fraction of all cases where the model returned a cited answer; for thesis reporting, the meaningful definition is answerable questions answered with citation
- `keyword_hit_rate`: fraction of `expected_keywords` matched by case-insensitive substring in the generated answer
- `hit@k`: retrieval-only metric indicating whether the expected source appears in the top-k resolved retrieval hits

---

## Generation Harness

```bash
uv run python eval/harness.py
```

The harness runs [GenerateFlow.generate_answer()](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/generate_flow.py) over all 65 questions and records:
- `grounded`
- `abstained`
- `citations`
- `source_correct`
- `keyword_hits`
- `latency_ms`

It writes a run artifact to `eval/results/run_<timestamp>.json` and per-question traces to `eval/results/traces/<run_id>/`.

### Current Generation Results

Live run artifact:
- [run_20260315T160649Z.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/run_20260315T160649Z.json)

Headline metrics:

| Metric | Value |
|---|---|
| Overall harness coverage | `65` questions |
| Answerable questions | `60` |
| Negative questions | `5` |
| Answerable questions answered with citation | `52/60` |
| Answerable abstentions or failures | `8/60` |
| Citation precision (answerable, non-abstained) | `88.5%` over `52` cases |
| Negative abstention recall | `1/5` (`20.0%`) |
| Meaningful grounded rate | `52/60 = 86.7%` |
| Average keyword hit rate | `0.181` |
| Average latency | `12.2 s` |

Notes:
- The harness summary field `overall_grounded_rate` is now lower than historical runs because abstentions correctly set `grounded=False`.
- For thesis reporting, `52/60 = 86.7%` is the more useful grounded-rate definition: answerable questions where the model produced a cited answer.
- Compared with the pre-migration abstention run, answerable citation precision improved but negative abstention recall fell. Stronger lexical retrieval helps the model find relevant-looking context even on unsupported questions.

### Difficulty Breakdown

| Difficulty | Grounded | Citation precision | Abstention recall | Answerable abstentions | Keyword hit rate |
|---|---|---|---|---:|---:|
| Easy | `85.7%` | `100.0%` | `0.0%` | `1` | `0.269` |
| Medium | `85.7%` | `90.9%` | `0.0%` | `4` | `0.121` |
| Hard | `86.7%` | `83.3%` | `33.3%` | `3` | `0.216` |

---

## Retrieval Ablations

```bash
uv run python eval/run_ablations.py
```

The ablation runner evaluates retrieval only on the `60` answerable questions and skips the `5` negatives.

Latest retrieval artifact:
- [ablations.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/ablations.json)

Current aggregate retrieval results:

| Configuration | hit@5 | hit@10 |
|---|---:|---:|
| 0. fts5-only (BM25 baseline) | `95.0%` | `98.3%` |
| 1. vector-only (dense baseline) | `76.7%` | `91.7%` |
| 2. vector + keyword (FTS5) | `76.7%` | `91.7%` |
| 3. vector + keyword + expansion | `93.3%` | `96.7%` |
| 4. full pipeline (+ rewrite) | `91.7%` | `96.7%` |

Interpretation:
- FTS5/BM25 is currently the strongest retrieval configuration on the expanded 65-question set
- config 2 shows that simply adding dense hits to the new FTS5 backbone does not help by itself
- graph expansion and rewrite now improve materially over the older LIKE-based pipeline, but still underperform pure BM25
- query rewrite remains mixed: it helps some sparse technical questions but still hurts paraphrase and some structured-planning queries

## Retrieval Baselines

The retrieval baseline question is now answered explicitly rather than by architectural preference.

| Configuration | hit@5 | hit@10 |
|---|---:|---:|
| 0. fts5-only (BM25 baseline) | `95.0%` | `98.3%` |
| 1. vector-only (dense baseline) | `76.7%` | `91.7%` |
| 2. vector + keyword (FTS5) | `76.7%` | `91.7%` |
| 3. vector + keyword + expansion | `93.3%` | `96.7%` |
| 4. full pipeline (+ rewrite) | `91.7%` | `96.7%` |

Examiner question: why did the system not use FTS5 earlier, and what changed?

Honest answer:
- the production retrieval path was originally built around transparent token-wise `LIKE` candidate generation plus vector composition
- after implementing an explicit BM25 baseline, FTS5 measured better than the older `LIKE` path
- on this 60-question answerable set, FTS5-only beats the old `LIKE`-based dense hybrid by `+18.3pp` at hit@5 (`95.0%` vs `76.7%`)
- it also beats the current expansion configuration by `+1.7pp` at hit@5 (`95.0%` vs `93.3%`)
- that result was strong enough that the production lexical backbone was migrated from `LIKE` to FTS5/BM25

So the current answer is:
- the repo did try the simpler `LIKE` design first because it was transparent and easy to audit
- FTS5 significantly outperformed it once measured directly
- production retrieval now uses `GraphStore.search_nodes()` backed by FTS5/BM25
- the earlier implementation remains available as `search_nodes_like()` for reference and regression testing

One important consequence remains:
- the composed pipeline does **not** currently outperform all single-method baselines
- FTS5-only is still the best retrieval configuration measured so far on hit@5 and hit@10

---

## Abstention Analysis

The current generation pipeline uses a two-gate abstention mechanism in [generate_flow.py](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/generate_flow.py):

1. `phrase gate`: detect explicit hedging language in the raw LLM response, such as `does not contain information`, `cannot answer`, or `not provided in`
2. `confidence gate`: if the response contains a citation marker anyway, only convert it into an abstention when the retrieval bundle is weak, using `max(hit.score) < 0.50`

Operationally:
- if the model hedges and gives no citation, the system abstains
- if the model hedges, cites anyway, and retrieval confidence is low, the system strips the guessed citation and abstains
- otherwise the response is treated as a normal cited answer

What this catches well:
- explicit unsupported-query responses with no evidence citation
- low-confidence guessed citations that look like “closest source” behavior rather than grounded answers

What it does not catch:
- negative questions where retrieval scores are still high enough to look plausible
- answerable questions where the model correctly notices that a specific metric or detail is absent, even though the broader paper is still the right source

Current abstention outcomes:
- Negative abstention recall: `1/5`
- Answerable abstentions: `8`
  - `pic2-009`
  - `jampacker-002`
  - `para-003`
  - `cross-005`
  - `densepack-002`
  - `sayplan-001`
  - `moma-001`
  - `vilain-003`

Interpretation of the answerable abstentions:
- some are likely genuine retrieval or generation misses
- some are detail-level abstentions where the model decides the requested fact is not sufficiently supported by the retrieved context, even though the paper-level answer is still expected by the golden set
- the harness treats all of them as answerable misses because the evaluation target is still a cited answer, not a hedge

Current limitation:
- after the FTS5 migration, four negative failures (`neg-002`, `neg-003`, `neg-004`, `neg-005`) still retrieve plausible enough context that the current abstention gate does not fire
- score alone is therefore not a reliable separator between valid evidence and unsupported-but-plausible retrieval bundles, especially once lexical retrieval gets stronger
- fixing this cleanly will require calibrated confidence scoring, a dedicated abstention classifier, or both

This is the practical ceiling of the current phrase-plus-score heuristic. It is good enough to expose unsupported answers honestly, but not yet strong enough to serve as a final confidence model.

---

## Failure Analysis

### Fixed retrieval/generation issues

1. Source-ID resolution bug
   - chunk citations were previously stored as `chk:` IDs instead of being resolved back to parent `src:` IDs
   - fixed in `_extract_citations()` in [generate_flow.py](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/generate_flow.py)

2. Golden-set source typo
   - SGVL expected source IDs used the wrong hyphen/underscore form
   - fixed in [golden.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/golden.json)

3. Partial-word lexical rescoring bug
   - lexical reranking was incorrectly crediting distractors on substring matches such as `mode` vs `models`
   - fixed in [retrieve_flow.py](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/retrieve_flow.py) by switching to exact word-token overlap

4. Expansion correctness bug
   - `_expand_to_sources()` used to follow any incoming edge
   - it now filters to `EdgeKind.contains` only

### Active failure classes

- rewrite-induced retrieval drift on paraphrase and cross-paper questions
- answerable questions where the LLM hedges about a missing detail instead of giving the broader supported answer
- unsupported questions whose retrieved context looks plausible enough that score-based abstention does not trigger

---

## Known Limitations

| Limitation | Status |
|---|---|
| Abstention is heuristic, not calibrated | Active |
| Three negative cases are indistinguishable from valid hits by score alone | Active |
| Query rewrite still hurts some paraphrase and cross-paper questions | Active |
| Keyword hit rate is lexical, not semantic | Acceptable but limited |
| Category counts are still small in several buckets | Active |
| Pure FTS5-only still outperforms the composed retrieval pipeline | Active |

---

## Next Evaluation Steps

- add calibrated confidence scoring or a dedicated abstention classifier
- run the generation harness again after any abstention-model change, not just retrieval ablations
- measure why BM25-only still beats the composed retrieval pipeline after the lexical migration
- report confidence intervals or small-sample caveats for tiny categories
- continue expanding the question set with externally written prompts
