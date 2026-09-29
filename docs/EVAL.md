# Evaluation

RCA is evaluated at two layers:
- retrieval quality: does the correct paper surface in the top-k bundle?
- generation quality: does the final answer cite the correct source, use retrieved evidence, and abstain when the corpus does not support the question?

This document reflects the current **129-question** evaluation corpus and the current eval scripts in the repo.

---

## Golden Set

The active golden set is [eval/golden.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/golden.json).

- Total questions: `129`
- Answerable: `90`
- Negative / unanswerable: `39`
- Difficulties: `15 easy`, `62 medium`, `52 hard`

Coverage includes:
- single-paper factual lookup
- method and contribution questions
- paraphrase / lexical mismatch cases
- cross-paper comparison
- multi-chunk synthesis
- explicit negative / unsupported queries

Negative questions `neg-011`–`neg-039` carry a `negative_type` field so abstention can be reported per failure mode:

| `negative_type` | Count | Meaning |
|---|---|---|
| `unreported` | `10` | paper is in the corpus, the asked detail is not |
| `false_premise` | `7` | question assumes a method, setting, or result the paper does not have |
| `out_of_corpus` | `4` | real system or benchmark that is not in the corpus |
| `fabricated` | `4` | invented paper or system name |
| `off_domain` | `4` | topic unrelated to the corpus |

Each new negative was checked by pattern search over all chunk text; candidates the corpus answers even partially were dropped (for example AutoBag's 500 g YuMi payload). The original `neg-001`–`neg-010` are untyped.

Metric definitions:
- `citation_precision`: fraction of answerable, non-abstained cases where every expected source appears in the returned citation IDs
- `abstention_recall`: fraction of negative questions where the system abstains correctly
- `grounded_rate`: fraction of all cases where the model returned a cited answer; for reporting, the more meaningful view is answerable questions answered with citation
- `keyword_hit_rate`: fraction of `expected_keywords` matched by case-insensitive substring in the generated answer
- `hit@k`: retrieval-only metric indicating whether the expected source appears in the top-k resolved retrieval hits

---

## Generation Harness

```bash
uv run python eval/harness.py --model gemma3:12b
```

Optional model override for direct A/B runs:

```bash
uv run python eval/harness.py --model llama3.1:8b
```

The harness runs [GenerateFlow.generate_answer()](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/rca/flows/generate_flow.py) over all `129` questions and records:
- `grounded`
- `abstained`
- `citations`
- `source_correct`
- `keyword_hits`
- `max_retrieval_score`
- `unique_sources_top5`
- `failure_labels`
- `latency_ms`

It writes a run artifact to `eval/results/run_<timestamp>.json` and per-question traces to `eval/results/traces/<run_id>/`.

Important caveat:
- harness results depend on the live local generation backend
- model selection is explicit/configurable: `--model` wins, then `RCA_EVAL_MODEL`, then normal `RCA_GENERATION_MODEL` settings
- if Ollama or another configured endpoint is unavailable, the run is not comparable to a normal local run
- after changing the corpus, the right source of truth is a fresh local rerun, not an older checked-in artifact

Failure labels are compact diagnostics emitted by the generation flow and carried into eval output. Current labels include `missing_citation`, `invalid_citation`, `unsupported_claim`, `low_retrieval_confidence`, `llm_abstained`, and runtime labels such as `empty_retrieval` or `error`.

### External eval files

Human-authored external eval subsets use the same JSON schema as `eval/golden.json`: a top-level list or `{ "pairs": [...] }`, where each pair has `id`, `question`, `difficulty`, optional `category`, `expected_keywords`, and either `expected_source` or `expected_sources`; set `answerable: false` for negatives.

Run an external subset separately:

```bash
uv run python eval/harness.py --external-golden-path path/to/external_golden.json --model gemma3:12b
```

Artifacts are labeled with `dataset_kind: "external"` so they can be reported separately from the self-authored golden set.

Current production-baseline run:
- [run_20260421T111146Z.json](/Users/dmcp2003/Desktop/Universidade/Mestrado/Research-Continuity-Agent/eval/results/run_20260421T111146Z.json)

Headline metrics:

| Metric | Value |
|---|---|
| Overall harness coverage | `100` questions |
| Answerable questions | `90` |
| Negative questions | `10` |
| Citation precision (answerable, non-abstained) | `90.8%` over `87` cases |
| Negative abstention recall | `3/10` (`30.0%`) |
| Answerable abstentions | `3` |
| Average keyword hit rate | `0.205` |
| Average latency | `9.7 s` |
| Grounded rate | `94.0%` |

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
| 4. full pipeline (+ rewrite) | `93.3%` | `96.7%` |

Interpretation guidance:
- FTS5/BM25 is the production lexical backbone and should be treated as the main sparse baseline
- dense retrieval remains useful, but on this corpus the lexical signal is often very strong
- source expansion and the cross-encoder reranker are the main reasons the composed pipeline improves on simpler hybrids
- query rewrite is now implemented as a small append-only expansion; on the latest Gemma-linked run it trails the FTS5 baseline on this corpus and requires further tuning

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
- for the original 100-question corpus, the split was `69` dev / `31` test; the 29 negatives added later were appended with the same seed and ratio (`20` dev / `9` test) so existing assignments stay stable, giving `89` dev / `40` test. Rerunning the sweep regenerates the full stratified split.

As with the other eval scripts, rerun locally when the corpus or retrieval backend changes.

Note:
- the coefficient-sweep script itself was fixed to track the current retriever signature and current corpus size
- if the sweep is run in a sandboxed environment that forces JSON fallback, do not treat those numbers as comparable to the live Chroma/Ollama runs above

---

## Multi-Model Matrix (Ollama)

To compare model behavior beyond a single default model, run a matrix benchmark.

```bash
uv run python eval/run_model_matrix.py \
	--models gemma3:12b gemma4:e4b llama3.1:8b qwen2.5:14b \
	--pull-missing
```

Output artifacts:
- per-model harness runs in `eval/results/run_<timestamp>.json`
- aggregate matrix in `eval/results/model_matrix.json`

Model onboarding recommendation:
- keep at least one strong local baseline (`gemma3:12b`)
- test 1 fast/small model for latency-sensitive scenarios
- test 1-2 alternate instruction-tuned models for citation and abstention variance
- compare both quality (`citation_precision`, `abstention_recall`) and cost proxy (`average_latency_ms`)

Current model-selection verdict:
- Production default is `gemma3:12b`.
- Split-model rewriting is retained as a research option but is not recommended for production. On this corpus it increased latency by 10s+ while improving citation precision by under 2 points and keyword hit rate by about 1 point.
- `gemma4:e4b` was evaluated as a candidate default and rejected for daily use due to high latency and elevated answerable abstentions despite stronger citation precision.

---

## Known Limitations

- Keyword hit rate (`20.5%`) is currently the primary retrieval weakness.
- Root cause appears to be rewrite prompt quality and corpus vocabulary alignment, not base model selection alone.
- Split-model rewriting has been evaluated and rejected for production due to latency cost versus marginal gains.

---

## Response Structure Conformance (Gemma rollout)

To avoid losing current UI-model integration context, we now track response-shape regressions explicitly.

Observed examples from recent comparison:
- Some responses include bullet-heavy markdown prefixes (for example `* **Jampacker:**`) that look noisy in compact answer views.
- Some responses emit mixed citation grouping such as `[[chk:...], [src:...]]`, which does not align cleanly with the expected inline citation chip pattern.
- Less-structured responses from other models can omit explicit source markers entirely, reducing citation usability.

Why this matters:
- The UI currently renders source references as end-of-answer source buttons/chips.
- Citation shape drift from the model can reduce parsing quality and degrade consistency across models.

Current plan item:
- Add a response-structure validation pass to the v2 plan and treat citation-shape conformance as a release-check criterion for model onboarding.

Minimum acceptance checks for a model profile:
- Inline citations resolve to valid source IDs or chunk IDs that can be normalized to source IDs.
- Bullet/markdown formatting does not break answer readability in the chat surface.
- Extracted citations map correctly to right-panel/source-button behavior.

### Exact UI Acceptance Tests: Citation Rendering

These are release-gate tests for the chat UI citation experience.

Test data setup:
- use fixed QA prompts with known expected source IDs
- include one case with chunk IDs only, one with mixed chunk/source IDs, one with repeated IDs, one with malformed IDs

Pass criteria notation:
- `P0` = must pass for release
- `P1` = should pass; can ship with explicit known issue only if non-blocking

1. `CIT-001` (P0) Valid source IDs render chips
- Input: assistant answer with `citations=[{"source_id":"src:pdf/paper_a"}]`
- Expected: one visible chip labeled `src:pdf/paper_a`; clickable; no console/runtime error

2. `CIT-002` (P0) Duplicate citations deduplicated in UI
- Input: same `source_id` repeated in response citations
- Expected: each unique source appears once in chip row

3. `CIT-003` (P0) Chunk IDs normalize to source behavior
- Input: `citations=[{"source_id":"chk:pdf/paper_a:0009"}]`
- Expected: chip renders; click opens source panel and PDF successfully (normalization allowed in backend or UI contract)

4. `CIT-004` (P0) Mixed citation formats survive answer cleanup
- Input: answer text containing inline `[[...]]` plus API citation array
- Expected: inline markers removed from prose; chip row still complete and clickable

5. `CIT-005` (P0) Malformed citation IDs are ignored safely
- Input: invalid `source_id` values (empty string, non-matching prefix)
- Expected: invalid entries do not crash render; valid entries still render

6. `CIT-006` (P1) Long source IDs remain readable
- Input: very long IDs
- Expected: chip layout wraps/scrolls without overlapping input bar or truncating entire row

7. `CIT-007` (P0) Citation click drives right-panel source state
- Input: click a citation chip in assistant message
- Expected: right panel switches to Source tab and loads corresponding PDF endpoint

8. `CIT-008` (P0) Unknown source returns graceful empty/error state
- Input: valid-format but non-existent source ID
- Expected: right panel shows controlled "not found / unavailable" state; app remains interactive

9. `CIT-009` (P1) Keyboard accessibility for citation chips
- Input: tab to chip, press Enter/Space
- Expected: same behavior as mouse click

10. `CIT-010` (P0) Multi-turn persistence correctness
- Input: switch conversations and return
- Expected: chips and selected source linkage remain scoped to the correct conversation

Suggested automation split:
- unit/component tests: `CIT-001` to `CIT-006`
- end-to-end tests: `CIT-007` to `CIT-010`

---

## Current Status

What is true today, independent of any single artifact:
- the golden corpus is now `129` questions (`39` negative)
- the split files cover that full corpus exactly
- the eval schema tests pass against the updated corpus and split files
- abstention remains heuristic and is still one of the main open weaknesses
- a human-authored external subset would still be stronger for bias reduction than self-authored or model-authored additions

Known active failure themes:
- citation selection drift on otherwise relevant retrieval bundles
- unsupported questions whose retrieved context still looks plausible enough to evade abstention
- detail-level abstentions on answerable questions
- rewrite-induced drift on some paraphrase and named-entity queries

---

## Practical Reporting Guidance

For writeups and portfolio material:
- report corpus size, answerable/negative counts, and split sizes explicitly
- state which backend produced the metrics
- avoid comparing runs produced under different retrieval backends, especially JSON fallback versus Chroma
- treat human-external question authorship as a separate bias-reduction property from simple corpus size growth
