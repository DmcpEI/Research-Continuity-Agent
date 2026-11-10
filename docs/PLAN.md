# RCA Plan

Written 2026-11-10 against commit `686f13b` (219 tests passing, ruff clean). This file supersedes the
"Now / Next / Later", "Phase 2 platform path" and "v2 execution tracks" sections of
[ROADMAP.md](ROADMAP.md); step 5 below folds the two files together.

Every statement about the code here was checked by reading it or running it. Where something was
not checked, it says so.

## 1. The one-sentence goal

> RCA answers questions about my own papers on a 4 GB local model, cites the passage it used, and
> says "not in your papers" when that is the truth. Here is how often each of those happens on a
> held-out test split, and the one command that reproduces the numbers.

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
  `partial` is added when the facts check ships (step 3). `error` exists only in the client.
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
  the `llm_abstained` label is computed and then dropped. This is the #6 gap; step 3 closes it.
- Two definitions of "hedged" exist (`GenerateFlow._ABSTENTION_SIGNALS` and
  `eval/sample_answer_labels.is_hedged`). Step 3 keeps one.

## 5. Steps

Each step is one commit, leaves `main` shippable, and is independent of the ones after it.
"Live" means run against the real API and Ollama, not only unit tests.

| # | Step | Verified by | What it adds to the CV story |
|---|---|---|---|
| 1 | **Typed answer outcome** end to end (flow, `/chat`, badge, harness). Done (`1476225`). | New tests fail on old code; full suite; frontend checks; live `/chat` | "Illegal states are unrepresentable"; a correct refusal is no longer shown as a warning |
| 2 | **A refusal always has text; an error is not an answer.** Empty-reply fix in the flow; the client marks failed requests as `error`, shows a retry, and keeps them out of history. Done. | Failing test first for the empty reply; vitest for history | Small, shows root-cause habit |
| 3 | **Ship the facts check** (#6): hedge-gated second call, `partial` outcome, one hedge definition, setting to disable. | Unit tests with stub LLM; harness on test split before/after: negatives refused, false refusals, latency | "Refusals on unanswerable questions: from X to Y at +Z s" with the comparison that chose the method |
| 4 | **One scorecard** (#5): harness on the test split with shipped defaults; retrieval hit@k with `qwen3.5:4b` rewrites; Wilson intervals; README table replaced by it. Recalibrate the rerank gate on dev if the rewrite model moved the scores. | The run itself, with provenance; numbers copied by script, not by hand | The headline sentence in section 1, with numbers |
| 5 | **Cut and tidy.** Delete `rca/orchestrator/`, `telemetry/metrics.py`, `telemetry/tracing.py`, `mcp_servers/arxiv/`, `PyJWT`, the event log; decide Streamlit and `infra/` (open questions); merge ROADMAP into this file; README under 200 lines; fix absolute links and the `pyproject.toml` description. Consider no longer tracking `frontend/dist` (every UI change rewrites hashed files and the push hook blocks until the rebuild is committed); build it in CI and the Docker image instead. | Full suite; `uv sync`; `docker compose build demo`; link check | A reviewer finds the claim in the first screen |
| 6 | **First run.** Badge "Grounded" renamed "Cited"; no seeded conversations; suggested questions that work on the demo corpus; clickable agent examples. | vitest; browser check on a fresh profile | The 60-second demo does not embarrass |
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

## 8. Questions only you can answer

1. **Is the ROADMAP audience still right?** It says "AI/ML, orchestration, and agent-systems roles
   in robotics-adjacent teams". If one of those matters most, say which: it changes the README's
   first paragraph and whether step 10 (agent) or step 12 (faithfulness) comes first.
2. **The GitHub billing lock.** CI cannot run until it is cleared. Is the repo public, and can the
   lock be removed?
3. **Commit and push.** Settled on 2026-11-10: step 1, this plan and F1 are separate commits, each
   reviewed before the push.
4. **Facts check on by default (step 3)?** My decision is yes, with a setting to turn it off. It
   costs about 3.5 s on about 40% of answers. Say no if latency matters more to you day to day.
5. **Delete the Streamlit app?** 1,204 lines, the only user of `plotly` and `networkx`, and the
   Dockerfile's default `CMD`. I would delete it once the React UI shows the eval views you still
   use there, if any. Which ones do you use?
6. **`infra/` (AWS).** Keep as an unmaintained example, or delete? I would delete: it has never
   been live and invites questions it cannot answer.
7. **Who can write 20 questions about your papers** (step 14), and can someone second-label the
   `para-006` answer?
8. **Track `CLAUDE.md` in git?** A reviewer cannot see how the project is run with coding agents
   today. I would track it; some reviewers read AI-assisted development as a negative, so it is
   your call.
9. **`gemma4:e4b`** (9.5 GB on disk): remove?
10. **The screencast** (#15): only you can record it; do it after step 6.
11. **Own model (step 15).** I placed the one that can be measured: a small classifier distilled
    from the facts check. If you meant a chat model of your own, say so; it fits after step 12.
