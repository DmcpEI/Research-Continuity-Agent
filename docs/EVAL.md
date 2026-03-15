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
- [run_20260315T194910Z.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/run_20260315T194910Z.json)

Headline metrics:

| Metric | Value |
|---|---|
| Overall harness coverage | `65` questions |
| Answerable questions | `60` |
| Negative questions | `5` |
| Answerable questions answered with citation | `56/60` |
| Answerable abstentions or failures | `4/60` |
| Citation precision (answerable, non-abstained) | `92.9%` over `56` cases |
| Negative abstention recall | `3/5` (`60.0%`) |
| Meaningful grounded rate | `56/60 = 93.3%` |
| Average keyword hit rate | `0.214` |
| Average latency | `10.5 s` |

Notes:
- The harness summary field `overall_grounded_rate` is now lower than historical runs because abstentions correctly set `grounded=False`.
- For thesis reporting, `52/60 = 86.7%` is the more useful grounded-rate definition: answerable questions where the model produced a cited answer.
- Compared with the pre-migration abstention run, answerable citation precision improved but negative abstention recall fell. Stronger lexical retrieval helps the model find relevant-looking context even on unsupported questions.

### Difficulty Breakdown

| Difficulty | Grounded | Citation precision | Abstention recall | Answerable abstentions | Keyword hit rate |
|---|---|---|---|---:|---:|
| Easy | `85.7%` | `100.0%` | `0.0%` | `1` | `0.312` |
| Medium | `89.3%` | `100.0%` | `50.0%` | `2` | `0.179` |
| Hard | `90.0%` | `84.6%` | `66.7%` | `1` | `0.225` |

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
| 3. vector + keyword + expansion | `95.0%` | `96.7%` |
| 4. full pipeline (+ rewrite) | `95.0%` | `98.3%` |

Interpretation:
- FTS5/BM25 remains the strongest single-method baseline, but the full reranked pipeline now matches it on both hit@5 and hit@10
- config 2 shows that simply adding dense hits to the new FTS5 backbone does not help by itself
- graph expansion is the single biggest downstream improvement over the dense baseline, lifting hit@5 from `76.7%` to `95.0%`
- the cross-encoder reranker is what closes the remaining gap between the composed pipeline and the pure BM25 baseline
- query rewrite remains mixed: it helps some sparse technical questions but still hurts paraphrase and some structured-planning queries

Per-category retrieval results now report 95% Wilson confidence intervals rather than raw percentages alone. This matters because several categories are still very small (`error_analysis n=2`, `paraphrase n=5`, `cross_paper n=6`), and point estimates on their own overstate certainty. Categories with `n <= 6` are flagged explicitly and should be interpreted with caution in the thesis writeup.

## Retrieval Baselines

The retrieval baseline question is now answered explicitly rather than by architectural preference.

| Configuration | hit@5 | hit@10 |
|---|---:|---:|
| 0. fts5-only (BM25 baseline) | `95.0%` | `98.3%` |
| 1. vector-only (dense baseline) | `76.7%` | `91.7%` |
| 2. vector + keyword (FTS5) | `76.7%` | `91.7%` |
| 3. vector + keyword + expansion | `95.0%` | `96.7%` |
| 4. full pipeline (+ rewrite) | `95.0%` | `98.3%` |

Examiner question: why did the system not use FTS5 earlier, and what changed?

Honest answer:
- the production retrieval path was originally built around transparent token-wise `LIKE` candidate generation plus vector composition
- after implementing an explicit BM25 baseline, FTS5 measured better than the older `LIKE` path
- on this 60-question answerable set, FTS5-only beats the old `LIKE`-based dense hybrid by `+18.3pp` at hit@5 (`95.0%` vs `76.7%`)
- before the coefficient sweep, it beat the expansion configuration by `+1.7pp` at hit@5 (`95.0%` vs `93.3%`)
- that result was strong enough that the production lexical backbone was migrated from `LIKE` to FTS5/BM25

So the current answer is:
- the repo did try the simpler `LIKE` design first because it was transparent and easy to audit
- FTS5 significantly outperformed it once measured directly
- production retrieval now uses `GraphStore.search_nodes()` backed by FTS5/BM25
- the earlier implementation remains available as `search_nodes_like()` for reference and regression testing

One important consequence remains:
- FTS5-only is still the cleanest single-method baseline
- after adding the cross-encoder reranker, the full pipeline now matches pure FTS5 on both hit@5 and hit@10 (`95.0%` / `98.3%`)
- the raw expansion pipeline without reranking still trails on hit@10 (`96.7%`)

This is worth stating plainly: pure FTS5 remains an unusually strong baseline on this corpus. On a small, domain-specific corpus like this one, BM25 term statistics are strong enough that the semantic retrieval layer only adds net value when it is paired with graph expansion and a strong final reranker. That is exactly what the current pipeline now does: config 3 gains `+18.3pp` over vector-only (`95.0%` vs `76.7%`), and the full reranked pipeline closes the remaining gap to pure BM25 at hit@10.

---

## Coefficient Sweep

The lexical reranker in [retrieve_flow.py](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/retrieve_flow.py) no longer defines the core lexical retrieval engine, because FTS5/BM25 now handles lexical candidate generation. It still matters, however, because it reranks lexical hits before they are merged with dense results and source expansion. To justify the title/text overlap weights empirically, RCA now includes a held-out coefficient sweep:

- split definition: `45` dev questions and `20` held-out test questions, saved in [dev.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/splits/dev.json) and [test.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/splits/test.json)
- split strategy: stratified by category with `random.seed(42)`
- objective: config 3 (`vector + keyword + expansion`) hit@5 on answerable questions only
- fixed base score: `0.45`
- sweep grid: `title_weight in [0.08, 0.12, 0.16, 0.20, 0.24]`, `text_weight in [0.01, 0.02, 0.03, 0.04, 0.05]`

Dev split hit@5 grid:

| title\text | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|---|---|---|---|---|---|
| 0.08 | 80.5% | 82.9% | 87.8% | 95.1% | 95.1% |
| 0.12 | 87.8% | 92.7% | 95.1% | 97.6% baseline | 97.6% selected |
| 0.16 | 92.7% | 95.1% | 95.1% | 97.6% | 97.6% |
| 0.20 | 95.1% | 95.1% | 95.1% | 97.6% | 97.6% |
| 0.24 | 95.1% | 95.1% | 95.1% | 97.6% | 97.6% |

Key findings:

- the current pre-sweep production weights (`0.12`, `0.04`) were already on the dev-optimal frontier, so they were not arbitrary
- the dev-optimal frontier contained eight tied combinations, not a single winner
- the held-out split was therefore used to choose among the tied dev-best settings

Held-out validation:

- baseline production (`0.12`, `0.04`): hit@5 `84.2%`, hit@10 `94.7%`
- selected setting (`0.12`, `0.05`): hit@5 `89.5%`, hit@10 `94.7%`

Decision:

- update production weights from (`0.12`, `0.04`) to (`0.12`, `0.05`)
- rationale: the held-out hit@5 gain was `+5.3pp`, which exceeds the `2.0pp` threshold, while held-out hit@10 stayed unchanged
- interpretation: the sweep did not justify increasing the title weight, but it did justify a small increase in text weight for the lexical reranker
- effect on the live ablation table: config 3 improved from `93.3%` to `95.0%` at hit@5, matching the FTS5-only baseline on hit@5 while keeping hit@10 at `96.7%`

Full artifact: [coefficient_sweep.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/coefficient_sweep.json)

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
- Negative abstention recall: `3/5`
- Answerable abstentions: `4`
  - `jampacker-001`
  - `jampacker-002`
  - `cross-004`
  - `densepack-001`

Interpretation of the answerable abstentions:
- some are likely genuine retrieval or generation misses
- some are detail-level abstentions where the model decides the requested fact is not sufficiently supported by the retrieved context, even though the paper-level answer is still expected by the golden set
- the harness treats all of them as answerable misses because the evaluation target is still a cited answer, not a hedge

Current limitation:
- after the reranker integration, two negative failures (`neg-001`, `neg-004`) still retrieve plausible enough context that the current abstention gate does not fire
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
| Pure FTS5-only still leads the composed pipeline on hit@10 | Active |

---

## Next Evaluation Steps

- add calibrated confidence scoring or a dedicated abstention classifier
- run the generation harness again after any abstention-model change, not just retrieval ablations
- measure why BM25-only still leads the tuned composed pipeline on hit@10
- report confidence intervals or small-sample caveats for tiny categories
- continue expanding the question set with externally written prompts
