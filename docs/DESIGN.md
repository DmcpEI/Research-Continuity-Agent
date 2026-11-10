# RCA frontend: review and design system

2026-11-10. Companion to [PLAN.md](PLAN.md). Based on reading `frontend/` and using every screen
of the built app in a browser at 1280×800 and 1440×900. Step F1 is done (section 6); the rest is
not implemented yet. The values themselves live in `frontend/src/styles/tokens.css`; this file
explains them.

## 1. What the UI is for

RCA's claim is: a small local model answers from your papers, shows the passage it used, and says
so when the answer is not there. The UI has one job: **make the evidence and the outcome visible
faster than the user can doubt them.** Everything below is judged against that. A screen, colour or
animation that does not help the user ask, wait, check or manage sources is decoration.

## 2. What is there today (measured)

Stack: React 18, Vite 5, TypeScript, react-query, react-router, d3, react-markdown, vitest. One
global stylesheet, `src/styles/globals.css`, 1,554 lines.

| Thing | Today | Problem |
|---|---|---|
| Colours | 9 tokens declared, but 24 distinct hex values and 56 distinct `rgba()` values in use | No system: two accents (purple `#7c6af7` declared, blue `#5b8af0` actually used), two greens |
| Secondary text | `#6b6b7e` on `#0a0a0f` = 3.8:1, on surface 3.6:1 | Fails WCAG AA (4.5:1), and it is used at 10–11 px |
| Font sizes | 19 distinct, from 0.62rem (9.9 px) to 1.15rem | No scale; most UI text is under 12 px |
| Radii | 21 distinct | No scale |
| Padding / gap | 47 / 20 distinct values | No spacing scale; nothing aligns to a grid |
| Motion | 1 transition, 1 animation (three dots) | A 19 s wait shows only "Thinking..." |
| Focus | 3 `:focus` rules in the whole app | Keyboard users cannot see where they are |
| Reduced motion | not handled | |
| Tailwind | installed and configured, 0 utility classes, no `@tailwind` directive | Dead dependency; its config repeats the tokens and already disagrees with the CSS |
| Fonts | IBM Plex from Google Fonts by `@import` | A local-first app that calls Google on every load and loses its fonts offline |
| d3 | `import * as d3` for one graph | Whole library in a 443 kB bundle |
| Lint | none for the frontend | Only `tsc` guards it |
| Shared state | `window` CustomEvents and `localStorage` between `ChatView` and `StatusBar` | Hidden coupling; `StatusBar` has four effects that sync the model |

What already works and should stay: the three-zone shell (rail, work area, side panel), the status
bar as an instrument strip, IBM Plex as the typeface, the dark neutral base, the query trace idea,
the outcome badge.

### Product problems seen while using it

1. **A citation does not show the evidence.** The chip is the raw ID
   (`src:pdf/robotic_grocery_bagging-structured_perception_for_...`). Clicking it opens the whole
   PDF at page 1. The API already returns `title` and `excerpt` for every citation; the UI drops
   both. Inline `[[...]]` markers are stripped, so no sentence is tied to a source, and stripping
   leaves " ." in the text.
2. **A missing PDF shows raw JSON** (`{"detail":"PDF file not found on disk"}`) in a white iframe.
   Cause found: that source was uploaded, and ingest stored a temp-directory path
   (`/var/folders/.../T/tmp.../x.pdf`) that no longer exists. 1 of 25 sources is affected.
3. **The wait is blind.** 19 s with three dots, no stage, no elapsed time, no way to stop.
4. **The side panel leads with a graph.** 25 overlapping labels; it answers no question the user
   has. "Concepts" are words from the title (`robotic`, `grocery`, `bagging`). A refusal still
   fills "Active sources".
5. **Titles are filenames**: `LLM-Pack_Intuitive_Grocery_Handling_for_Logistics_Applications`,
   repeated underneath as the `src:` ID. "Updated" is "Unknown" for 24 of 25 rows, each with a
   green dot that means nothing.
6. **The primary Library action is below the fold.** "+ Add sources" is the last table row.
7. **Dead ends in the navigation.** Settings is "coming soon". The empty-chat suggestions ("What
   changed in my retrieval quality over the last week?") cannot be answered from a paper library.
8. **Lines are too long to read.** The answer runs about 110 characters per line at 14.7 px.
9. **The trace repeats the query on every row** instead of showing where the time went.
10. **Single-line `<input>`** for questions; the status bar says "Ollama connected" when the API is
    down; the page title is "RCA Frontend"; no favicon.

## 3. Design system

One file, `src/styles/tokens.css`, holds every value. Components use tokens only: a raw hex, a raw
pixel size or a new radius in a component rule is a review failure. Plain CSS custom properties;
no CSS framework.

### Colour

Neutral surfaces carry the app. **One accent**, used only for things you can act on or that are
selected. **Four outcome colours**, used only for outcomes and health, always with a label or icon
so colour is never the only signal. Every text colour is at least 4.5:1 on every surface it sits on
(ratios below are on background / surface / raised).

```css
:root {
  /* surfaces */
  --bg: #0b0c10;            /* app background */
  --surface: #13151b;       /* panels, cards, inputs */
  --raised: #1b1e27;        /* hover, selected row, menus */
  --border: #272b36;        /* dividers */
  --border-strong: #3a4050; /* inputs, focusable outlines at rest */

  /* text */
  --text: #e6e8ee;          /* 16.0 / 14.9 / 13.6 */
  --text-2: #a7adbb;        /* 8.7 / 8.1 / 7.4   secondary */
  --text-3: #868d9e;        /* 5.9 / 5.5 / 5.0   captions, never below 12 px */

  /* accent: interactive and selected only */
  --accent: #9d91ff;        /* 7.4 / 6.9 / 6.3   links, citation markers, focus ring */
  --accent-solid: #6657e8;  /* button fill; white text on it is 5.1:1 */
  --accent-soft: rgb(157 145 255 / 0.12);

  /* outcomes and health: nothing else may use these */
  --ok: #4cc38a;            /* answered, connected */
  --warn: #e5a83b;          /* unverified, degraded */
  --danger: #f47c74;        /* request failed, destructive */
  --neutral: var(--text-2); /* not found: a correct refusal is not a warning */
  --ok-soft: rgb(76 195 138 / 0.12);
  --warn-soft: rgb(229 168 59 / 0.12);
  --danger-soft: rgb(244 124 116 / 0.12);
}
```

Light theme is the same token names under `@media (prefers-color-scheme: light)`:
`--bg #fbfbfc`, `--surface #ffffff`, `--raised #f2f3f6`, `--border #dfe2e8`, `--text #16181d`,
`--text-2 #4d5361`, `--text-3 #636a79`, `--accent #5546d6`, `--ok #1a7f4e`, `--warn #8f5a00`,
`--danger #c0352d` (all at least 4.5:1). It costs about 15 lines once components use tokens, and a
paper reader next to a white PDF is the case for it. Do it last.

### Type

Keep IBM Plex Sans and IBM Plex Mono, served from `frontend/public/fonts` as `woff2` (weights 400
and 600 for Sans, 400 and 500 for Mono) with a system-font fallback. No network request.

| Token | Size / line | Use |
|---|---|---|
| `--fs-xs` | 12 / 16 | captions, table headers, badges. The minimum. |
| `--fs-sm` | 13 / 20 | controls, rows, chips, status bar |
| `--fs-md` | 14 / 20 | default UI text |
| `--fs-read` | 16 / 26 | answers and questions; column capped at `68ch` |
| `--fs-lg` | 20 / 28 | page titles |

Weights: 400 and 600 only. **Mono is for data**: IDs, numbers, timings, model names, code. Labels,
buttons and suggestions are Sans. Uppercase with letter-spacing is for section headers only
(12 px, `0.06em`); today there are eight different letter-spacings.

### Space, shape, depth

- Spacing scale, 4 px base: `--s-half` 2 (hairline gaps inside pills only), `--s-1` 4, `--s-2` 8, `--s-3` 12, `--s-4` 16, `--s-6` 24, `--s-8` 32,
  `--s-12` 48. Nothing in between, and no `%` or viewport units in padding, margin or gap (use
  `aspect-ratio` for proportional boxes). `tokens.test.ts` enforces this, with custom properties
  outside `tokens.css` counted as raw values.
- Radius: `--r-sm` 4 (chips, badges inside text), `--r-md` 8 (inputs, buttons, cards, menus),
  `--r-full` (pills, dots).
- Depth: borders, not shadows. One shadow token, for things that float (model picker, row menu).
- Control heights: 28 (dense: status bar, chips), 36 (default), 44 (the question box).

### Alignment (the "by eye" rules)

- **One reading column.** Question, answer, pending line and question box share the same left and
  right edges: `min(100%, 68ch + 2 × var(--s-6))`, centred in the work area. Today the answer card,
  the "Thinking" line and the input each start at a different x.
- **Text aligns to text, not box to box.** A bordered element is pulled out by its own padding so
  its text sits on the column edge. The answer loses its outer card: badge line, text, sources,
  trace, separated by space, not by nested boxes (today: card in card in card).
- **Rows share a baseline grid.** Page header, panel tabs and the conversation list header are all
  48 px tall so their text baselines line up across the three zones.
- **Icons are optically centred**, 16 px in a 28 px target, stroke 1.5, all drawn the same way.
  Text glyphs used as icons (`✓`, `⚠`, `↗`, `›`, `+`) become SVGs, because glyphs sit on the text
  baseline and render differently per font.
- Numbers in tables are right-aligned with `font-variant-numeric: tabular-nums`.

### Interaction states

Hover is a neutral tint (`--raised`). Selected or current is the accent (`--accent-soft`, plus an
accent bar or border where the row needs it). The two never share a colour, so the user can tell
"under the pointer" from "chosen". Destructive actions use `--danger`, not `--warn`.

### Motion

Motion is feedback. If it does not tell the user that something was received, is progressing, or
where something went, it is removed.

| Token | Value | Use |
|---|---|---|
| `--t-fast` | 120 ms ease-out | hover, press, colour, focus ring |
| `--t-base` | 200 ms ease-out | a panel, menu or disclosure opening; a new message arriving |
| `--t-slow` | 600 ms ease-out | one-shot highlight on the passage a citation points to |
| `--t-loop` | 1 s | one cycle of a looping indicator (the thinking dots; delays are fractions of it) |

`@media (prefers-reduced-motion: reduce)` sets every duration to 0 and stops looping animations.

### States every component must have

A component is not done until each of these is designed, not defaulted:
**rest, hover, focus-visible, active, disabled, busy, empty, error.** Focus is one rule for the
whole app: `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }`.

| Where | Empty | Busy | Error |
|---|---|---|---|
| Chat thread | Three questions that work on this library, built from real titles | Pending block: current stage, elapsed seconds, Stop | "Request failed" badge, the reason in plain words, Retry |
| Answer | n/a | n/a | Outcome badge is the state (section 4) |
| Evidence panel | "Select an answer to see its passages" | Skeleton lines | "This PDF is no longer on disk. Re-add it in Library." |
| Library | "Add your first PDFs", with the drop target | Skeleton rows on load; per-file progress on upload | Row-level message; the list stays usable |
| Agent | Clickable example tasks that work | Live step list (tool, argument, result size) | Stop reason in words |
| Status bar | n/a | "Loading model…" while Ollama loads it | "API unreachable" (today it keeps saying connected) |

Skeletons are for content whose shape is known (table rows, passage list). An answer has no known
shape, so the pending block shows stages instead.

## 4. Recommendations, ranked

1. **Make the citation show the passage.** Render each `[[...]]` marker as a numbered link at its
   place in the sentence. Under the answer, a numbered source list: paper title, then the excerpt
   the API already sends. Clicking a number scrolls to and flashes that passage in the side panel
   (`--t-slow`). This is the product claim made visible, and it is the screenshot for the README.
2. **Side panel becomes Evidence, not Graph.** Default tab: the passages behind the selected
   answer, then "Open PDF". Remove "Concepts". Decide the graph separately (question 1 below).
3. **An honest pending state.** Now: elapsed seconds and the typical time for this model, plus
   Stop. With plan step 11 (stage events from the server): the real stage list, each with its
   time as it finishes. Never a timer-driven fake.
4. **Adopt the tokens.** Add `tokens.css`, replace every colour, size, radius and space in
   `globals.css` with a token, delete Tailwind and its config. No visual redesign in this commit;
   it only snaps values to the scale. Largest diff, lowest risk, and every later step is smaller
   because of it.
5. **Reading column and alignment rules** from section 3. 16 px answers, 68ch, one left edge.
6. **Outcome language everywhere.** The badge words and colours from `AnswerOutcome` are the only
   status vocabulary: the same four states in chat, in the trace, in the eval views if they come
   to React. "Grounded" becomes "Cited" (plan decision 5). A refusal is neutral, never amber.
7. **Trace as a timing bar.** One row per stage: name, a bar proportional to its share, ms. Query
   variants and hit lists behind a disclosure. It shows the project's "measured" character in the
   product itself.
8. **Library that reads like a library.** Human titles (at least underscores to spaces; better,
   the PDF's own title at ingest), ID in a detail line only on hover or selection, no dot and a
   dash when the date is unknown, "Add sources" as the header button plus drop-anywhere, and a
   visible flag on a source whose file is gone.
9. **Remove dead ends.** Drop Settings from the rail until it has content. Replace the
   suggestions. Title "RCA", a favicon.
10. **Question box**: auto-growing `<textarea>`, Enter sends, Shift+Enter breaks the line.
11. **Agent screen uses the same parts**: same column, same question box, same pending block, its
    trace in the same side-panel slot as Evidence. One layout learned once.
12. **Replace window events with one small React context** for the active model and the pending
    flag. Removes four syncing effects in `StatusBar`.
13. **Fonts served locally.** See Type.

## 5. Anti-recommendations

- **Do not adopt a component library or Tailwind now** (shadcn, MUI, Radix themes). The app has
  about 15 components. A rewrite changes 1,500 lines of CSS for no visible gain and adds
  dependencies to defend in an interview. Tokens plus six plain primitives (Button, Badge, Chip,
  Panel, Field, Skeleton) are enough.
- **Do not polish the knowledge graph.** It is the most "demo" part and the least useful. Effort
  there is taken from the citation path.
- **No typewriter effect on an answer that already arrived**, and no staged progress driven by a
  timer. Both are the UI claiming something that is not true, in a project whose point is that it
  does not do that. Real token streaming later is fine.
- **No gradients, glows, glass or animated backgrounds.** The outcome colours must be the only
  saturated things on the screen or they stop meaning anything.
- **Do not use green or amber for decoration.** Not for "Updated", not for the active nav item.
- **Do not add a second accent.** Selected, link and focus are the same colour on purpose.
- **No placeholder screens or disabled features in the navigation.**
- **No mobile layout.** It is a desktop tool beside a PDF. Support down to 1024 px wide by
  collapsing the side panel; below that, nothing.
- **No theme picker.** Follow the system setting.
- **No Storybook, no visual-regression service.** Vitest plus the browser check is proportionate.
- **Do not upgrade React or Vite as part of this.** Separate change, no user gain.
- **Do not put more data in mono or uppercase** to look technical. It reads slower.

## 6. Steps

Each is one commit, checked with `npm run typecheck && npm run build && npm test` and in the
browser at 1280×800, in both colour schemes once the light theme exists.

| # | Step | Proof |
|---|---|---|
| F1 | **Done.** `tokens.css`; snap `globals.css` to it; delete Tailwind; one `:focus-visible` rule; reduced motion. Local fonts moved to after question 3; graph labels (SVG, 9 px) and graph colours wait for question 1. | `tokens.test.ts` (colours, scaled properties incl. `font`, `animation*` and `transition*`, colour literals in components; fails on the old files). Browser audit of Chat, Library and Agent: every HTML text element at least 4.5:1, smallest 12 px, focus ring present. Before: 34 text elements under 4.5:1 on Agent, smallest 9.9 px |
| F2 | Reading column, 16 px answers, alignment rules, answer without nested cards | Screenshot; measured line length ≤ 75 characters |
| F3 | Numbered inline citations, source list with title and excerpt | vitest: marker `[[chk:..]]` renders link "1" and the excerpt; no " ." left behind |
| F4 | Evidence tab in the side panel; missing-PDF message; remove Concepts | vitest for empty, loaded and missing-file states; live check with the affected source |
| F5 | Pending block with elapsed time and Stop (client side; aborts the fetch) | vitest with fake timers; live |
| F6 | Trace as timing bars | vitest: bar widths sum to 100% |
| F7 | Library: titles, header "Add sources", drop target, skeleton rows, no meaningless dot | vitest; live upload of one PDF into a temp data directory |
| F8 | Remove Settings link, new suggestions, textarea, title and favicon, "API unreachable" | vitest; live with the server stopped |
| F9 | Agent screen on the shared parts; model/pending context replaces window events | Existing tests unchanged; live |
| F10 | Light theme | Contrast script over both token sets |

F3 and F4 come before plan step 6 in value: they are the difference between "cites" and "shows".
Backend work these need, tracked in PLAN.md: store uploaded PDFs under the data directory, not a
temp path (F4); page number in chunk metadata so "Open PDF" can jump to the page (after F4); stage
events (plan step 11, upgrades F5).

## 7. Questions for you

1. **The graph: keep as a Library view, or delete with d3?** I would delete it. It is 283 lines and
   the whole of d3 for a picture that answers nothing. Keep it only if you use it.
2. **Dark and light, or dark only?** I recommend both, following the system, as the last step.
3. **Plex served from the repo** (four `woff2` files, about 100 kB, OFL licence), or drop Plex for
   the system font and add no files?
4. **May I add a frontend linter** (one dev dependency)? Not required for any step above.
