# RCA Plan

Written 2026-11-10 against commit `686f13b` (219 tests passing, ruff clean). This is the only plan:
the old ROADMAP.md was folded in and deleted in step 5 (see "Kept from ROADMAP" below), and where
an issue disagrees with this file, this file wins.

Every statement about the code here was checked by reading it or running it. Where something was
not checked, it says so.

## 1. The one-sentence goal

> RCA answers questions about my own papers on a 4 GB local model, cites the passage it used, and
> says "not in your papers" when that is the truth. Here is how often each of those happens on a
> held-out test split, and the one command that reproduces the numbers.

The numbers for that sentence are the generated scorecard in the README ("Results at a glance")
and EVAL.md, from `eval/scorecard.json` (step 4); they are not copied here, so they cannot drift.

Everything in this plan is ranked by how much it helps that sentence become true and provable.
A reviewer has about five minutes. A project that makes one claim and proves it beats a project
with twelve features and a mixed table of numbers.

## 2. Honest assessment

### What is already strong

- **Evaluation discipline.** 129 questions with 39 typed negatives, a stratified dev/test split, a
  rerank threshold calibrated on dev and reported on test, run provenance in every artifact, a
  failure-stage taxonomy, a retrieval regression gate in CI, and hand-labelled answer outcomes with
  a classifier comparison. Most student RAG projects have none of this.
- **Findings that came from measuring.** The "low keyword hit rate" was traced to the answer model
  seeing 240 characters per chunk and fixed (`2094d68`). The current abstention rule was measured
  at 1/8 on test and reported as such. Negative results are in the docs.
- **A real constraint.** 18 GB of RAM forced a small-model bake-off and a default (`qwen3.5:4b`)
  chosen from numbers, not taste.
- **Revisioned, idempotent ingest** with hash-based identity.

### What a strong reviewer will not be convinced by

1. **The headline numbers do not describe the system that ships.** Retrieval hit@5 98.9% and the
   rerank gate (5/12 at 1/28) were measured with `gemma3:12b` query rewrites; that model is gone.
   Answer quality for the default model exists only on the dev split (51/62 real cited answers,
   14/27 negatives refused). There is no single test-split scorecard for the default configuration.
   ROADMAP's "Current metrics" table is still the April run on 100 questions.
2. **The core promise is the weakest measured area.** On dev, 13 of 27 unanswerable questions get
   an answer, and 6 refusals are shown with a green "Grounded" badge because they cite a nearby
   block. A fix was measured (`verifier_facts_on_hedge`: 8/8 non-answers caught and 26/27 useful
   answers kept on test) but it is not in the product.
3. **"Grounded" overclaims.** The check behind the badge is: the answer has a citation that
   resolves to a retrieved source, and shares two content words with it. That is "cited", not
   "supported". An interviewer will find this in two minutes.
4. **Small n, one author.** Test has 8 `not_found` answers and 12 negatives. Questions, labels and
   verifier prompts were all written by the same person/assistant. Intervals are not reported next
   to most numbers.
5. **Surface area is larger than the evidence.** Two UIs (React and a 1,204-line Streamlit app
   that the default Docker `CMD` still starts), AWS ECS templates with no live deployment, an empty
   `rca/mcp_servers/arxiv/` package, and code nothing imports: `rca/orchestrator/` (Router and
   tool-policy enforcement), `rca/telemetry/metrics.py`, `rca/telemetry/tracing.py`, the `PyJWT`
   dependency. The ingest event log is written and never read.
6. **First run looks unfinished.** Three fake seeded conversations, suggested questions RCA cannot
   answer ("What changed in my retrieval quality over the last week?"), raw `API 502: {json}` in a
   chat bubble, "Unverified" on a correct refusal, a ten-second wait with only "Thinking…", an
   Agent tab that looks like a chat but forgets everything.
7. **State is implicit almost everywhere** (section 4). This is what makes both the UI and the
   code hard to reason about, for people and for coding agents.
8. **Docs drift.** Absolute local paths in ROADMAP links, wrong split sizes in places, a stale
   `pyproject.toml` description, README at 523 lines repeating the roadmap.
9. **CI is not running.** GitHub Actions is blocked by an account billing lock, so the badge at the
   top of the README does not back up anything. CI has no coverage and no type checker, although
   `CLAUDE.md` says to run "tests with coverage".

### Docs and code that disagree (flagged, not silently fixed)

| Doc says | Code does |
|---|---|
| `CLAUDE.md`: "Run the same command CI runs (e.g. tests with coverage)" | CI runs `uv run pytest -v --tb=short`, no coverage |
| `CLAUDE.md` is described as checked in | `/CLAUDE.md` is in `.gitignore` |
| ROADMAP "Current metrics": 146 tests, 20.5% keyword hit, 3/10 negatives | 219 tests; those metrics are from 2026-04-21 on a different set and model |
| ROADMAP: Streamlit is "the supported local demo surface" | README and the Docker demo lead with the React UI; the Dockerfile default `CMD` runs Streamlit |
| Frontend sends `model` and `history` to `/agent` | `AgentRequest` has no `model`; `agent.run(payload.query)` ignores `messages` |
| `/chat` clients are told an answer is `grounded` or not | The flow also knows `abstained`; the API did not return it (fixed in step 1) |

## 3. Decisions, ranked

1. **Prove one claim on the shipped configuration.** One scorecard, test split, default model,
   current code, one command. It replaces the mixed table in the README. *Why first in rank:* every
   other number in the repo is currently an answer to a question nobody asked.
2. **Ship the measured refusal fix.** The hedge-gated facts check goes into `GenerateFlow`, on by
   default, with a setting to turn it off. Cost: about 3.5 s extra on roughly 40% of answers.
   *Why:* it is the weakest area, the fix is already measured, and "we measured four classifiers
   and shipped the one that won" is the best story in the repo. It needs an end-to-end rerun
   because the comparison so far ran on saved answers.
3. **Make state explicit, starting with the answer outcome.** One enum from the flow to the API to
   the badge to the eval artifact. Decision 2 cannot be built cleanly without it, so it is step 1.
4. **Cut.** Delete the unused code and dependencies. Drop v2 tracks A to D (multi-select, pinned
   models, session timeline, in-app eval) and the Phase 2 platform path (pgvector, auth, SSR,
   object storage, background worker). Close #12 (multi-strategy literature agent); park #11.
   *Why:* these add breadth to a product with one user. "Production-shaped" without production is
   a claim a hiring manager discounts; one honest paragraph on what would change for multi-user
   is worth more than a `Postgres + pgvector` bullet.
5. **Name things for what they check.** The badge is renamed from "Grounded" to "Cited" in step 6
   and becomes "Supported" only when a faithfulness check exists and passes (step 12).
   #7 (LLM judge) is reframed as that check: "does the cited passage support this sentence",
   validated against about 50 hand labels, same method as #6.
6. **Then daily-use UX:** stage progress and cancel before token streaming (#16), honest error and
   empty states, a first run with no fake data.
7. **Robustness where it is visible:** vector-store recovery (#19), no silent agent-model switch.
8. **Agent: narrow and honest.** Typed stop reasons, a search tool that returns chunk IDs and full
   text so the agent can cite, rerun the 20-task bench. No new tools until the core claim is done.

### Where I think your assumptions are off

- **"Daily assistant" and "CV centrepiece" pull in different directions.** A daily assistant wants
  features; a CV project wants depth on one claim. I resolve it this way: daily-use work is done
  only when it also demos well (progress, cancel, honest states). Library bulk actions do not.
- **"Fix every state machine" is not one task.** There are twelve. Three are worth real work, four
  are small fixes, and the rest should be left alone or deleted. They are sequenced by what a user
  sees, not by completeness.
- **"Make no mistakes" cannot be promised by anyone.** What can be promised is that every claim is
  marked as tested live, tested by unit tests, or not tested. That is what this plan does.
- **The literature-search agent (#11, #12) is the wrong next feature.** It adds a second product
  with no eval before the first one has its headline number.
- **A background ingest worker is not needed.** One user, PDFs that index in seconds. A
  synchronous upload with an honest progress state and a consistency check is enough.

## 4. The implicit state machines

For each: what the states are today, what is wrong, the decision, and how "nothing broke" is
proven. UX is the person using RCA, DX the person changing it, AX both coding agents in this repo
and RCA's own agent.

### M1. Answer outcome — step 1 (done in the working tree)

- **Today:** two booleans (`grounded`, `abstained`), a list of string labels, canned answer strings
  and `trace.query_type`. `grounded=True, abstained=True` is representable. `/chat` drops
  `abstained`; the UI shows everything that is not grounded, including a correct refusal, a
  question about the chat itself and a failed request, as "⚠ Unverified".
- **Decision:** `AnswerOutcome` = `answered | unverified | not_found | conversational`. It is the
  only stored field; `grounded` and `abstained` are derived from it and stay in every payload.
  `partial` was planned for step 3 and left out: no classifier tells it apart reliably (see "Step 3:
  facts check, design"). `error` exists only in the client.
- **UX:** a refusal reads "No evidence found" in neutral colour; a question about the chat itself has
  no badge; a failed request says so. A greeting such as "hello" is not classified as
  conversational and still gets a refusal (section 7; not part of step 2, still open). **DX:** one field to switch on; a constructor must name the outcome.
  **AX:** an agent adding a new path cannot forget to decide what it is, and eval artifacts carry
  the same word the UI shows.
- **Proof:** unit tests for every return path (each fails on `686f13b`), an API contract test,
  vitest badge tests, a live `/chat` check. See section 7.

### M2. Ingest and re-ingest

- **Today:** `created | updated | unchanged`, decided by a five-step identity match. Writes go
  graph (one transaction) → vector delete → vector upsert → revision row → event log. There is no
  cross-store transaction, but because the revision row is written last, a failed ingest is seen
  as `updated` on retry and repairs itself. Nothing shows a half-indexed source if nobody retries.
  Uploading the same PDF twice leaves a second copy on disk (`name-1.pdf`).
- **Decision:** keep the order and say why in the code: *the revision row is the commit record.*
  Derive an "incomplete" flag for `/sources` from "no revision matches the node's content hash"
  instead of adding a status column. Hash the upload before writing it. No background worker.
- **Proof:** an integration test where the vector store fails mid-ingest, then a second ingest
  converges; a test that an incomplete source is flagged; upload-twice test.

### M3. Vector backend (#19)

- **Today:** `chroma → json`, one way, for the life of the process, on any exception, including an
  Ollama embedding failure that has nothing to do with Chroma. Chunks upserted after the flip live
  only in the JSON file; after a restart Chroma is back without them.
- **Decision:** no latch. Try Chroma on every call; on failure use the fallback for that call and
  record `degraded_since` and `last_error`; clear them on the next success. Remember IDs upserted
  while degraded and re-upsert them on recovery. Add a cooldown only if a failed attempt turns out
  to be slow. This is fewer states than the issue proposes.
- **Proof:** a fake collection that fails then recovers (no sleeps); live: stop Ollama, query,
  start Ollama, query, `/status` clears.

### M4. Reranker availability

- **Today:** lazy load on first query. If it fails, a warning goes into the trace and the
  pre-generation rerank gate silently stops applying, so refusal behaviour changes with no signal.
- **Decision:** load at startup so failure is visible at boot; report it in `/status` warnings
  (the mechanism exists since #17); add a `rerank_unavailable` failure label.
- **Proof:** unit test with a failing reranker stub; `/status` test.

### M5. Model lifecycle

- **Today:** a per-request `model` on `/chat` rewrites shared settings and the shared LLM client.
  The agent model has a tool-capability cache that never expires and a silent switch to "the first
  capable model", which can load a second large model into 18 GB. About 350 of 933 lines in
  `rca/api/main.py` are model ranking and probing.
- **Decision:** a per-request model builds a request-scoped client and mutates nothing. No silent
  switch: if the selected agent model cannot call tools, `/agent` says so and the UI opens the
  picker. Probe only the selected model, keyed by model digest. Move ranking and probing to
  `rca/llm/models.py`. This changes one tested behaviour on purpose
  (`test_agent_models_auto_reconciles_invalid_current_model`); that test is rewritten to assert
  the new behaviour, not deleted.
- **Proof:** existing model-selection API tests; a test that two interleaved `/chat` requests with
  different models do not affect each other.

### M6. Agent loop

- **Today:** seven stop reasons as free strings; a clarification heuristic with a hard-coded corpus
  word (`pic2`); the search tool returns 240-character excerpts without chunk IDs, so the agent
  cannot cite or read the passage; the UI is a chat but every run is single-shot.
- **Decision:** `StopReason` enum, same pattern as M1; the UI maps each to a sentence. The search
  tool returns chunk IDs and bounded full text. Remove `pic2`. Make single-shot honest: the client
  stops sending `model` and `history`, and the view says each task is independent.
- **Proof:** agent unit tests; rerun the 20-task bench on `qwen3.5:4b` (today 17/20) and report the
  change either way.

### M7. Conversations

- **Today:** `localStorage` only; three fake seeds; a global "another conversation is waiting"
  lock; error bubbles and refusals are stored as assistant turns and sent back as history, with no
  length bound.
- **Decision:** keep `localStorage` (one user; no backend sessions). Remove the seeds. Send only
  the last N turns and never `error` turns (the second half is done in step 2; the bound is not).
  The global lock goes once M5 makes requests independent.
- **Proof:** vitest for the history filter; a browser check of a fresh profile.

### M8. Request and UI states (#16)

- **Today:** idle → "Thinking…" → answer, or an error dressed as an answer. No progress, no cancel.
  Agent view state is lost on navigation.
- **Decision:** `/chat/stream` (SSE) sends `stage` events (`retrieving`, `generating`,
  `checking`), then one `final` event with today's payload. Tokens shown during `generating` are
  provisional until `final`, because the outcome can still become `not_found`. Cancel aborts the
  request and stops generation. Stage events ship before token streaming: they are cheap and
  already honest. `/chat` stays as is for the eval harness and tests.
- **Proof:** an API test that reads the event sequence from a stub flow; vitest for the reducer;
  a live run.

### M9. Eval runs

- **Today:** good. `complete`, provenance, `index_degraded_during_run`.
- **Decision:** add the outcome distribution (done in step 1) and the scorecard (step 4). Put the
  rules that are currently folklore into `CLAUDE.md`: test split only for final numbers; no number
  without split, n, model and commit; nothing published from a dirty or degraded run.

### M10. Experiment run status, M11. MCP client lifecycle, M12. Event log

- **M10** (`pending | running | complete | failed`, any transition allowed): leave. The agent only
  reads it.
- **M11** (lazy start, closed on error, restarted on next call): reads as sound. Not audited in
  depth; no change planned.
- **M12:** the event log is write-only. Delete it in step 5 unless you use the file by hand.

### Smaller state found on the way (each has a step)

- An empty model reply with low retrieval confidence returns `not_found` with an **empty answer
  string**. Reproduced on `686f13b` with a stub LLM (`'' False True ['missing_citation',
  'low_retrieval_confidence']`). Fixed in step 2; a citation-only reply had the same hole.
- A hedged reply with one valid citation and good retrieval is `answered` with no failure label:
  the `llm_abstained` label is computed and then dropped. This is the #6 gap; step 3 sends such
  replies to the facts check.
- Two definitions of "hedged" existed (`GenerateFlow._ABSTENTION_SIGNALS` and
  `eval/sample_answer_labels.is_hedged`). Step 3 kept the second, now `rca.flows.facts_check.is_hedged`.

## 5. Steps

Each step is one commit, leaves `main` shippable, and is independent of the ones after it.
"Live" means run against the real API and Ollama, not only unit tests.

| # | Step | Verified by | What it adds to the CV story |
|---|---|---|---|
| 1 | **Typed answer outcome** end to end (flow, `/chat`, badge, harness). Done (`1476225`). | New tests fail on old code; full suite; frontend checks; live `/chat` | "Illegal states are unrepresentable"; a correct refusal is no longer shown as a warning |
| 2 | **A refusal always has text; an error is not an answer.** Empty-reply fix in the flow; the client marks failed requests as `error`, shows a retry, and keeps them out of history. Done. | Failing test first for the empty reply; vitest for history | Small, shows root-cause habit |
| 3 | **Ship the facts check** (#6): hedge-gated second call, one hedge definition, setting to disable. The planned `partial` outcome was dropped (see "Step 3: facts check, design"). Done: test split, unanswerable declined 6/12 → 12/12, answerable 1/28 → 2/28, +0.6 s mean (EVAL.md). | Unit tests with stub LLM; harness on test split before/after: negatives refused, false refusals, latency | "Refusals on unanswerable questions: from X to Y at +Z s" with the comparison that chose the method |
| 4 | **One scorecard** (#5): harness on the test split with shipped defaults; retrieval hit@k with `qwen3.5:4b` rewrites; Wilson intervals; README table replaced by it. Recalibrate the rerank gate on dev if the rewrite model moved the scores. Done: `eval/scorecard.json`, generated into README and EVAL.md; rerank gate unchanged (see "Step 4: one scorecard, design"). | The run itself, with provenance; numbers copied by script, not by hand | The headline sentence in section 1, with numbers |
| 5 | **Cut and tidy.** Delete `rca/orchestrator/`, `telemetry/metrics.py`, `telemetry/tracing.py`, `mcp_servers/arxiv/`, `PyJWT`, the event log; decide Streamlit and `infra/` (open questions); merge ROADMAP into this file; README under 200 lines; fix absolute links and the `pyproject.toml` description. Consider no longer tracking `frontend/dist` (every UI change rewrites hashed files and the push hook blocks until the rebuild is committed); build it in CI and the Docker image instead. Done: dead code, the event log, Streamlit and `infra/` deleted; ROADMAP folded in; README 187 lines (see "Step 5: cut and tidy, design"). | Full suite; `uv sync`; `docker compose build demo`; link check | A reviewer finds the claim in the first screen |
| 6 | **First run.** Badge "Grounded" renamed "Cited"; no seeded conversations; suggested questions that work on the demo corpus; clickable agent examples. Done: see "Step 6: first run, design". | vitest; browser check on a fresh profile | The 60-second demo does not embarrass |
| 7 | **Vector store recovers** (#19), as in M3. | Fake collection test; live Ollama stop/start | "Degrades and recovers without a restart" |
| 8 | **Reranker failure is visible** (M4). | Stub test; `/status` test | Same theme: no silent mode changes |
| 9 | **Request-scoped models, no silent agent switch** (M5); move model code out of `main.py`. | API tests incl. interleaved requests | `main.py` shrinks by a third; removes a RAM foot-gun |
| 10 | **Agent contract** (M6): `StopReason`, citeable search results, no `pic2`, honest single-shot. | Unit tests; 20-task bench rerun | Agent answers can cite; bench number is for the shipped tool |
| 11 | **Progress and cancel** (#16, M8): stage events first, tokens second. | Event-sequence API test; live | The daily-use fix people notice |
| 12 | **"Cited" becomes "Supported"** (#7 reframed): per-sentence support check validated on ~50 hand labels; the badge changes from "Cited" to "Supported" only when it passes. | Label file; classifier comparison as in #6 | The second measured claim |
| 13 | **Ingest consistency** (M2): commit-record comment, incomplete flag, upload hash check. | Mid-ingest failure test | Shows the two-store trade-off was understood |
| 14 | **External questions** (needs you): 20 questions written by someone else, reported separately. | Harness | Answers "you graded your own homework" |
| 15 | **Own model: distil the facts check** (needs 3 and 4). Step 3's `qwen3.5:4b` verifier labels a few thousand answers to questions generated from the library (teacher); fine-tune the reranker's MiniLM cross-encoder (22M parameters; `sentence-transformers` is already a dependency) to predict `answered` / `partial` / `not_found` from the question, the answer and the cited passage (student). Ship it in place of the second LLM call only if it matches the teacher on the hand labels. | Hand-labelled test split never used for training or tuning; agreement with labels and with the teacher; latency per answer; training time on the M3 Pro | "Trained my own model: the same refusal accuracy as the 4B verifier in milliseconds instead of 3.5 s", with the numbers |

**Frontend.** The UI review, the design system (colour, type, spacing, motion tokens) and its own
steps F1–F10 are in [DESIGN.md](DESIGN.md). F3 and F4 (a citation shows the passage; the side
panel shows evidence, not a graph) rank with steps 2–3 here: they are the claim made visible.
Step 6 above is covered by F8. Backend work they need: store uploaded PDFs under the data
directory (today an upload is ingested from a temp path that is later deleted, so 1 of 25
sources has no PDF to open and the viewer shows raw JSON), and a page number per chunk.

**Own model (step 15).** A model of your own is worth it when it does a job RCA already measures,
so the result is a number, not a demo. The facts check is that job: step 3 adds a 3.5 s LLM call to
about 40% of answers, hand labels for it already exist (`eval/answer_labels_fulltext_*.json`), and a
small classifier is fast to train and run on 18 GB. The 4B verifier writes the training labels, so
the hand labels stay a clean judge; their test split is never trained or tuned on. Not chosen:
training a language model from scratch (no data or compute to beat `qwen3.5:4b`, and nothing to
measure it against); a LoRA fine-tune of the chat model to cite better (possible later with MLX, but
only once step 12 gives a support metric to train and judge against); fine-tuning the reranker on the
library (129 eval questions are too few, and the rerank gate is calibrated on them).

Steps 1 to 4 are the project. Stop after 4 and the CV story is already better than after all of
the old v2 tracks. Steps 5 and 6 make it presentable. Steps 7 to 13 are depth, in the order a user
would notice them.

Not planned: pgvector, auth, SSR, live AWS, background worker, library bulk actions, pinned
models, session timeline, in-app eval mode, literature-search strategies (#12).

## 6. Rules for coding agents (`CLAUDE.md`)

The proposed file is a diff at [PLAN-claude-md.diff](PLAN-claude-md.diff) (apply with
`patch -p0 < docs/PLAN-claude-md.diff` from the repo root; delete the diff afterwards). It is not
applied: `CLAUDE.md` is your file and is git-ignored.

**Kept, because each has caught a real problem in this repo:** reproduce first; root cause and
grep every caller; prove a new test fails on the old code; never weaken a test; live verification
with an honest tested/not-tested report; ask before pushing or deleting; no secrets or real names;
stop only your own processes; back up config.

**Changed, because the cost was higher than the gain:**

| Rule today | Problem | Proposed |
|---|---|---|
| Generic rules plus an "overrides" section | Two sources of truth; half the rules say "PR" in a repo that commits to `main` | One set of rules written for this repo |
| Design note before every change | Ceremony for a one-line fix | Only when a change crosses a module boundary or changes a contract; it goes in the commit body or this plan |
| "How could this break" checklist on every PR | Eight items, most not applicable to most changes, so it gets skimmed | Shorter list tied to this repo's real failure modes; say which ones you checked |
| Independent break-it reviewer before every PR | Spawns an agent for a typo fix | Required for behaviour changes in flows, stores and the API; a self-review pass otherwise |
| "Every supported language" | Only English exists | Removed |
| "Run the same command CI runs (e.g. tests with coverage)" | CI has no coverage | The exact commands, listed |

**Added, because their absence has already cost time:**

- **Eval rules.** Dev for tuning, test for final numbers only. No number without split, n, model
  and commit. Thresholds (`-3.618`, `0.55`, `0.50`) and the reranker's 240-character input are
  calibrated values: changing one means recalibrating and rerunning.
- **Hardware.** One chat model in RAM; `gemma3:12b` is not to be used; long runs go in the
  background.
- **The user's library is real data.** `.rca/` is not a test fixture. Tests and probes use a temp
  directory or a copy.
- **State.** A new state is an enum value, not another boolean or a match on message text.
- **Commit format and no AI attribution lines** (today this lives only in an agent's memory).
- **Definition of done** in one place, and **when to stop and ask**.
- **Where things are:** this plan, the handoff, `docs/EVAL.md` as the only home for numbers.

## 7. Step 1: what was done and how it was checked

See the "Unreleased" entry in [CHANGELOG.md](../CHANGELOG.md). Verification results are recorded
in the handoff after each run; the summary for this step:

- **CI commands, by exit code (2026-11-10):** `uv run ruff check .` 0, `uv run ruff format --check .` 0, `uv run pytest -v --tb=short` 0 (228 passed, 219 before), and in `frontend/`: `npm run typecheck` 0, `npm run build` 0, `npm test` 0 (26 passed). The rebuilt `frontend/dist` belongs to this change.
- **New tests fail on the old code:** all backend outcome tests fail on `686f13b`; on the frontend the refusal, chat-about-chat, error and wiring tests fail on the old components, and the five that pass there are the backward-compatibility guards (old messages keep their badge).
- **Live, API:** a server on port 8765 with `qwen3.5:4b`, no `model` field: answerable question → `answered` / `grounded: true` with 3 citations; out-of-corpus → `not_found`; "Why didn't you respond?" → `conversational`; "hello" → `not_found`.
- **Live, UI (built `frontend/dist` in the browser):** "No evidence found" on the refusal, no badge on the chat-about-chat reply, "✓ Grounded" on the cited answer, badges unchanged after a reload, and "Request failed" after the server was stopped.
- **Not tested live:** the `unverified` badge (no live question produced an uncited answer), the eval harness `outcome_counts` (unit tests only; no eval run), and the Streamlit app.
- **Review:** self-review of the full diff only; no independent reviewer was run.
- **Not changed, on purpose:** the Streamlit app (`app.py:843`) still shows "⚠ unverified" on a refusal. It has no tests and its deletion is open question 5; if it stays, its badge must read `outcome`.

Found while checking (not fixed in step 1; each needs a step):

- **`RCA_DATA_DIR` does not move the stores.** `graph_db_path`, `vector_dir`, `event_log_path` and `experiment_db_path` each default to `.rca/...` on their own, so the live check, started with `RCA_DATA_DIR` pointing at a copy, opened the real library. Compared with the copy taken beforehand: `nodes`, `edges` and `source_revisions` are identical row for row, vectors, uploads, event log content and experiments are byte-identical, and `integrity_check` is ok. What changed is what every app start changes: the event log's modification time and the FTS index, which `GraphStore.create_schema` rebuilds on each start. Fix: derive the four paths from `data_dir`, and stop rebuilding FTS on every start.
- A greeting such as "hello" is not classified as conversational and gets a refusal.
- The chat-about-chat reply leaks prompt wording ("the provided context does not contain...").
- A refusal still fills the "Active sources" panel, and the status bar keeps saying "Ollama connected" when the API is down.
- The UI took 7.4 s for the refusal that took 1.0 s through the API alone; cause not investigated.
- **The test suite opens the real library.** `rca/api/main.py` builds `app = create_app()` at import
  time with default settings, so every test that imports it (three test files) opens `.rca/`:
  the FTS index is rebuilt and the event log reopened. Checked on 2026-11-10: row counts and event
  log content unchanged, timestamps bumped on every `pytest` run. Present on `686f13b`. Fix with
  the `RCA_DATA_DIR` fix above: tests get a temp data directory, and the module-level app is
  created lazily.
- **`index.html` is served cacheable** (`ETag`/`Last-Modified`, no `Cache-Control: no-cache`).
  After a rebuild the browser kept the old `index.html`, which names asset files that no longer
  exist; a user upgrading the demo can get a stale or blank page. Fix: `no-cache` on `index.html`
  only; hashed assets can stay cacheable.

## Step 3: facts check, design (2026-11-11)

- **What ships** is the measured `verifier_facts_on_hedge` (EVAL.md, "Answer classifiers") plus the
  citation rules that were already there: when the reply hedges, a second call with the same chat
  model lists the statements that answer the question, and none means RCA declines. Because the
  citation rules still decline on their own, the offline numbers are an upper bound: recomputed
  from the saved predictions and pipeline labels, dev is unchanged (19/21 caught, 49/55 kept) and
  test keeps 25/27 instead of 26/27 (`jampacker-001`, unhedged, declined for `invalid_citation`). Prompt, schema and parser move from `eval/` into
  `rca/flows/facts_check.py`, and the comparison script imports them, so the eval measures the
  shipped prompt.
- **It replaces only the hedge rule.** Before, a hedged reply was a refusal only when its citation
  failed or retrieval was weak, so a refusal citing a nearby block showed as "Grounded". Now the
  check decides whether a hedged reply is a refusal. The citation rules (invalid or unsupported
  citation, missing citation with weak retrieval) are unchanged, as are unhedged replies.
- **No `partial` outcome.** Section 5 planned one; EVAL.md found partial versus answered unreliable
  with every classifier (test: 2 of 6 partials predicted), so a "Partial answer" badge would mostly
  be wrong. The check is used for useful or not, as EVAL.md recommends. Add `partial` only when a
  classifier earns it.
- **One runtime hedge definition,** `rca.flows.facts_check.is_hedged`: the eval sampler's regex,
  widened to cover every phrase of the old list (the review found "not provided in" and "does not
  contain information" refusals missed by the regex alone) and a few close variants (passages,
  excerpts, "couldn't find"). On the 211 labelled answers it matches what the sampler's regex
  matches, plus 41 replies the old phrase list missed, mostly partial answers and refusals citing a
  nearby block. The sampler keeps its own frozen copy so the label files stay reproducible; a test
  checks the runtime rule is a superset. With the check off or failed the old hedge rule runs with
  the wider definition, so "off" declines slightly more often than before step 3.
- **Failure is not a refusal:** if the call fails or the reply is malformed, the old rule decides
  and the answer carries `facts_check_failed`. A refusal from the check carries
  `facts_check_no_answer`. The trace gets a `facts_check` stage.
- **On by default** (open question 4), `RCA_FACTS_CHECK_ENABLED=false` turns it off. Cost: about
  3.5 s on the roughly 40% of replies that hedge.
- **Proof:** stub-LLM tests for each branch, failing first; harness on the test split with the check
  on and off; a live `/chat` check.

## Step 4: one scorecard, design (2026-11-11)

- **One run, one artifact.** `eval/harness.py --split test` runs the 40 test questions with the
  shipped defaults; `eval/scorecard.py <run>` turns that run into `eval/scorecard.json` (committed)
  and rewrites one generated block in README.md and docs/EVAL.md. A test fails if either block
  differs from what the committed scorecard renders, so the numbers are never typed by hand.
- **The script refuses** a run that is not the golden test split, was made from a dirty tree, had
  the vector index degrade or start off Chroma, has an errored question, overrides the rewrite
  model, or has a setting grounded chat uses off its code default (`facts_check_enabled` is now
  recorded too; the agent and OpenAI model names are ignored). It also refuses a run where a
  component fell back, which the flow only records as a trace warning or label (reranker
  unavailable, query rewrite failed, facts check failed): the harness now copies each question's
  trace warnings into the run, and any warning outside the normal ones (proper-noun rewrite skip,
  rerank gate, failure labels, empty retrieval) blocks publishing. And it pins the questions: the
  run records the golden file's SHA-256 and the question IDs, which must equal the committed
  `eval/golden.json` and `eval/splits/test.json`, so a hand-picked file named "test" cannot pass.
  This is the M9 rule "nothing published from a dirty or degraded run" as code. The break-it
  review found the fallback, question-set and missing-field gaps; each has a test.
- **What it reports:** answerable questions answered citing the expected paper (end to end, over
  all answerable, not only the answered ones), answerable refused, unanswerable refused, citation
  precision when it answers, retrieval hit@5, answer keyword hit rate and time per question, each
  proportion with a Wilson 95% interval, plus the IDs of every miss.
- **Retrieval hit@5 comes from the same run:** the harness already records the top 5 sources of
  the final bundle (`retrieved_sources`, from the trace), which is ablation config 5 with this run's
  `qwen3.5:4b` rewrites. No hit@10 and no separate ablation run: production uses 5. The 90-question
  ablation table stays in EVAL.md as history, labelled with its `gemma3:12b` rewrites.
- **Rerank gate: not recalibrated, because the rewrite model did not move it.** Retrieval features
  on all 129 questions with `qwen3.5:4b` rewrites (commit `fb78227`, clean tree, Chroma, corpus
  `51b6785c3405cb16`) select the same rule on dev, `rerank_max <= -3.618` (12/27 negatives at 2/62
  false abstentions, as with `gemma3:12b`); on test it catches 5/12 at 0/28 (was 1/28).
- **Not in the scorecard:** dev numbers (tuning only), the agent bench (a separate claim with its
  own README line), hit@10.
- **Open, found by the second review:** an empty model reply (the #21 failure mode) and a reply
  cut off at `num_predict` leave no trace warning on `/chat`, so the scorecard cannot refuse a
  run that had them. Fixing it means a flow change (a warning on both paths), so it is not part
  of this step. The published run had no empty reply.
- **Run-to-run variation is larger than the intervals suggest.** The two step-4 runs gave
  identical replies; the step-3 run of the same flow code that morning differed on 18 of 40
  replies and 4 outcomes. Cause not found (same model digest, settings and corpus). Noted next to
  the scorecard in EVAL.md.

## Step 5: cut and tidy, design (2026-11-11)

Your answers (2026-11-11): delete the Streamlit app, delete `infra/`, delete the event log (not
read by hand), keep tracking `frontend/dist` for now. One commit per cut, so each can be reverted
alone:

1. **Code nothing imports.** `rca/orchestrator/` (Router, tool policies) with
   `rca/config/tool_policies.yaml` and the `tool_policy_path` setting it alone read,
   `rca/telemetry/metrics.py`, `rca/telemetry/tracing.py`, the empty `rca/mcp_servers/arxiv/`, and
   the `PyJWT` dependency (no importer). `PyYAML` stays: the experiment extractor uses it. Checked by
   grep for every module and import name. An old `.env` that still sets `RCA_TOOL_POLICY_PATH` keeps
   working: settings ignore unknown keys.
2. **The event log.** `rca/store/event_log.py` is written on every ingest and read by nothing. The
   writer, the `event_log_path` setting and `RCA_EVENT_LOG_PATH` go; existing `events.jsonl` files
   are left on disk.
3. **Streamlit.** `app.py`, `.streamlit/`, and the `streamlit`, `plotly` and `networkx`
   dependencies (only `app.py` imports them). The Docker image now serves the React app and API
   (`create_site`) on port 8000 instead of Streamlit on 8501; `docker compose up rca` and `make run`
   serve the user's `./.rca` there, bound to localhost like the demo (the API has no login and can
   delete sources). `rca` and `demo` both use port 8000, so run one at a time.
4. **`infra/`.** The AWS ECS scripts and templates (never deployed) and the `aws-*` Make targets.
   The image's default backend was `openai_compatible` only so it could run on ECS; it becomes
   `ollama`, which is what both compose services already set.
5. **Docs.** ROADMAP.md is folded into this file and deleted (its still-useful parts: shipped
   history goes to the CHANGELOG pointer, open items already map to steps here); README under 200
   lines; absolute local links made relative; `pyproject.toml` description says what RCA is.

Proof: full CI after each commit; `docker compose build` and a live `rca` container answering one
`/api/chat` question for cut 3; `uv sync` from the new lock.

## Step 6: first run, design (2026-11-11)

Scope is the step-6 row; the rest of DESIGN F8 (Settings link, textarea, title and favicon, "API
unreachable") stays in F8.

- **Badge "✓ Grounded" becomes "✓ Cited"** (decision 5): the check behind it is "cites a retrieved
  source", not "supported". Only the label changes; `outcome` and the API are untouched. The agent
  header's "return grounded results" goes too: the agent has no grounding check.
- **No seeded conversations.** Correction found while building: the three fake conversations were
  never on screen (the list hides conversations with no messages), so "three fake seeded
  conversations" in section 2 overstated it; they were dead state in `localStorage` that also
  hid a bug. A fresh profile now starts with one empty "New conversation". The three
  fake ones a browser already saved (ids `conv-1` to `conv-3`, their exact titles, no messages)
  are dropped on load; anything with messages is kept. Deleting the last conversation opens a new
  empty one: without the hidden seeds, the deleted one stayed active and every later question went
  to a conversation no list showed (reproduced in a test with the seeds removed). Loading never
  deletes stored messages, even when every listed conversation was a seed (break-it review).
- **Suggestions come from the library, not a fixed list.** The old ones ("What changed in my
  retrieval quality over the last week?") could not be answered from papers, and fixed demo-corpus
  questions would be refused on any other library. The empty chat shows "What problem does
  “<title>” address, and how?" for up to three sources, titles with underscores turned into spaces,
  preferring titles short enough to read (filenames are often cut). Checked live on the library
  copy before building: 5 of 5 such questions answered with citations, 13–23 s each. An empty
  library shows a pointer to Library instead.
- **Agent examples are buttons that fill the box**, like the chat chips: one knowledge-base task
  from a library title, "List the files in the workspace folder." and "List the recorded experiment
  runs and their statuses.", which work on any setup (an empty list is a correct answer).
- **Proof:** vitest for the title helpers, the seed cleanup and the chips (each failing on the old
  code); a browser check on a fresh profile against the library copy.

## Kept from ROADMAP (folded in 2026-11-11)

ROADMAP.md was deleted in step 5. Its shipped-milestone lists are history (git log, CHANGELOG.md);
its "Current metrics" are replaced by the scorecard; its Now/Next/Later, Phase 2 and v2 tracks are
superseded by the steps above or listed under "Not planned". What was still worth keeping:

- **Audience:** AI/ML, orchestration and agent-systems roles in robotics-adjacent teams. The
  project is the reasoning and memory layer, not robot control (question 1 asks whether this holds).
- **Guardrails.** RCA is a research memory and grounded QA system, not a robot controller.
  Orchestration claims stay honest: grounded chat is direct flow composition, the agent loop is a
  separate tool-use path. Local-first is the default; cloud and API backends are optional
  extensions. Split-model rewriting was measured (10 s+ more latency for marginal gain) and is not
  a recommended configuration.
- **Hardware:** on the 18 GB M3 Pro, `phi4:14b` is at the practical memory limit; `gemma3:12b`
  left 2–4% RAM free and was dropped; `qwen3.5:4b` serves chat and agent so one chat model is
  loaded.
- **Not planned, kept as ideas:** response-structure checks when onboarding a new model (citation
  tag shape, markdown noise), useful only if the default model changes; a human-authored external
  question subset is step 14; per-conversation request isolation is M7.

## 8. Questions only you can answer

1. **Is the audience still right?** ROADMAP said "AI/ML, orchestration, and agent-systems roles
   in robotics-adjacent teams". If one of those matters most, say which: it changes the README's
   first paragraph and whether step 10 (agent) or step 12 (faithfulness) comes first.
2. **The GitHub billing lock.** CI cannot run until it is cleared. Is the repo public, and can the
   lock be removed?
3. **Commit and push.** Settled on 2026-11-10: step 1, this plan and F1 are separate commits, each
   reviewed before the push.
4. **Facts check on by default (step 3)?** My decision is yes, with a setting to turn it off. It
   costs about 3.5 s on about 40% of answers. Say no if latency matters more to you day to day.
5. **Delete the Streamlit app?** Answered 2026-11-11: yes, deleted in step 5.
6. **`infra/` (AWS).** Answered 2026-11-11: delete; deleted in step 5.
7. **Who can write 20 questions about your papers** (step 14), and can someone second-label the
   `para-006` answer?
8. **Track `CLAUDE.md` in git?** A reviewer cannot see how the project is run with coding agents
   today. I would track it; some reviewers read AI-assisted development as a negative, so it is
   your call.
9. **`gemma4:e4b`** (9.5 GB on disk): remove?
10. **The screencast** (#15): only you can record it; do it after step 6.
11. **Own model (step 15).** I placed the one that can be measured: a small classifier distilled
    from the facts check. If you meant a chat model of your own, say so; it fits after step 12.
