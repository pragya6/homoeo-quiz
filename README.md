---
title: HomeoQuiz
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
---

# 🌿 HomeoQuiz

Grounded MCQ practice for Indian homoeopathy entrance exams (UPSC Homoeopathy,
SR-ship). Every question is generated from the classical public-domain canon and
**verified against its source passage before the student sees it**.

The bot does not answer from memory. If the corpus doesn't cover a topic, it
refuses. That refusal is the product.

---

## Quickstart

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # or put it in .env

# 0. Corpus is already in out/*.json (7 books, see CORPUS_HANDOFF.md /
#    data/raw/SOURCES.md) -- nothing to download or clean first.

# 1. Build the remedy alias table -- do this first, everything else joins on it
python -m core.remedies

# 2. Build the vector index -- tiers are a flag, not a code edit
python -m ingest.build_index --books organon,boericke,allen   # tier 1
# python -m ingest.build_index --books organon,boericke,allen,hering,clarke,nash  # tier 2, once tier 1 is calibrated

# 3. Build the Kent repertory store -- SQLite + FTS5, never embedded
python -m ingest.build_repertory

# 4. Prove it's grounded before you ship it
python -m evaluation.run_eval --repeats 3

# 5. Serve
python app.py
```

On Hugging Face Spaces: add `GEMINI_API_KEY` as a **Secret**, commit
`data/chroma/`, `data/repertory.db`, and `data/remedy_aliases.json` alongside
the code.

---

## System design

### 1. Scale and scope

Target: **2–10 concurrent students.** Sized to sit inside the Gemini free
tier — though "free tier" turned out to be per-project, not a fixed published
number: this project measured **10 RPM and a 20 req/day cap** on
`gemini-2.5-flash-lite` directly from live 429 responses, well under the
documented 15 RPM / 1,000 req/day (see "Known limits"). `core/ratelimit.py`
throttles at 8 RPM with a token bucket and retries 429s with jittered
backoff, so a burst degrades into a short wait rather than an error — but no
local throttle fixes a daily cap; that needs a linked billing account
(Gemini Tier 1) or patience.

A served question costs ~1 LLM call (`rag/deterministic.py`, Kent/Hering
facts) or ~3 (`rag/generator.py`, generate → distractors → verify), mixed
`DETERMINISTIC_ROUTE_RATE` : the rest — roughly 2.2 calls/question blended at
the default 0.4. "Repertory" always routes deterministic (Kent was never
embedded — see §2); everything else rolls the split.

Scoped to the homoeopathy core (Organon, Materia Medica, Repertory). General
medical subjects are excluded on purpose — they age, and they aren't public
domain. See `data/raw/SOURCES.md`. Homoeopathic Philosophy is in `config.
SUBJECTS` but has no indexed content yet — see "Known limits" for what's
blocking it and the concrete steps to close it.

**Scaling path:** session state is in-process (`gr.State`). Beyond ~15 concurrent
students, move `Session` into Redis keyed by user id, put the Chroma index behind
a read-only service or swap for a managed vector DB, and flip `FREE_TIER_MODE=0`
to lift throttles and route the hard tasks to the smarter model.

### 2. Data and retrieval

- **Corpus:** seven JSON files (`out/*.json`, ~72,800 records), public domain,
  clean HTML-derived — not OCR — from homeoint.org. Legally clean *and*
  canonical: the UPSC Organon syllabus is §1–§291 of the 6th edition, which
  *is* the public-domain text (5th-edition-only §292–294 excluded by design,
  not an oversight). See `CORPUS_HANDOFF.md` for the full schema-by-schema
  breakdown and `data/raw/SOURCES.md` for provenance.
- **Remedy names are canonicalised once, up front** (`core/remedies.py` →
  `data/remedy_aliases.json`), seeded from Hering's `{remedy, abbrev}` pairs
  and resolved against the other five remedy-bearing books (Kent, Clarke,
  Boericke, Nash, Allen). Every cross-book feature — distractor mining,
  Kent's reverse index, Allen/Hering dedup — joins on this rather than
  re-guessing spelling per call. Coverage is real, not 100%: unresolved names
  are printed, never silently dropped (`python -m core.remedies` reports the
  count per book).
- **Chunking is per-schema, not fixed-size** (`ingest/build_chunks.py`, one
  function per book). Organon splits on `§n` and never merges two aphorisms;
  footnotes become sibling chunks carrying `parent_locator`, not appended
  text. Boericke/Clarke split one chunk per `fields` entry. Hering splits one
  chunk per section per remedy (its fixed 1–48 schema), with the running
  "remedy name repeated as a symptom" artifact stripped and Allen-flagged
  near-duplicate symptoms suppressed (below). Allen has no `fields` (verified
  against the actual JSON, contrary to its schema doc) and Nash is
  prose-only, so both get paragraph-packed from `raw_text` instead. A
  fixed-width splitter would bisect aphorism §17 and quietly destroy
  grounding; this doesn't.
- **Kent is a database, not a retrieval corpus** (`ingest/build_repertory.py`
  → `data/repertory.db`). 69,991 rubrics / 394,352 graded remedy entries of
  terse, abbreviation-heavy, no-sentence text ("WAX, black > tympanum") is
  close to the worst case for dense embedding, and at 77% of the corpus it
  would crowd out every other book's hits. It never touches Chroma; SQLite +
  FTS5 gives keyword/BM25 rubric lookup, and an index on
  `(remedy_canonical, grade)` gives the reverse direction (remedy → its
  grade-3 rubrics) — the direction the handoff says most repertory questions
  actually run.
- **Allen ↔ Hering overlap is suppressed, not filtered out.** Both books are
  indexed; where `allen_keynotes.dedup.json` flags a specific keynote as a
  near-duplicate (containment ≥ 0.7 against a specific Hering sentence —
  precomputed, not retuned here), that exact Hering symptom is dropped from
  its section chunk at build time (`ingest/build_chunks.py`), so the same
  fact isn't askable from both authors. The threshold isn't lowered to catch
  more: a false "duplicate" pairing unrelated symptoms is worse than a missed
  one.
- **Embeddings run locally** (`bge-small-en-v1.5`, 384-dim, CPU). Not primarily a
  cost decision — it removes a rate-limited API from the hot path.
- **Metadata filtering** on `subject` so "quiz me on Repertory" cannot return
  Materia Medica — though "Repertory" is served entirely by the deterministic
  path below, since Kent carries no vector metadata to filter on in the first
  place.
- **Confidence gate, two accept paths:** raise `NotInCorpus` and refuse unless
  EITHER top-1 cosine similarity clears `MIN_RETRIEVAL_SIMILARITY` outright,
  OR it clears a lower `SOFT_SIM_FLOOR` *and* leads the next-best non-sparse
  candidate by `MIN_MARGIN` (a real topic usually has one clear best match; a
  nonsense topic usually has several tied, so-so ones). Both are calibrated
  on the goldset, not guessed — see "Known limits" for the sweep that picked
  the values and the honest cases neither path can rescue.
- **Sparse-source exclusion:** single-line fields (Boericke's `Dose`,
  `Relationship`/`Complementary`, Clarke's `Relations`/`Causation`) and fields
  explicitly labelled non-homoeopathic (`Non-Homeopathic Uses`) never win
  rank-0, nor does anything under a 15-word floor. None of these carry enough
  propositional content — or the right *kind* of content — to build a
  defensible exam question on; letting one win drove the generator to invent
  an explanation the passage didn't actually support. They're still eligible
  as neighbour/distractor material — exactly the kind of "related but
  different" content that makes a wrong option tempting. See
  `rag/retriever.py::retrieve_for_question`. Measured effect across both
  calibration rounds: `false_refusal_rate` 0.606 → 0.30 → 0.133,
  `refusal_accuracy` 0.6 → 0.83 → 1.0 — see "Known limits" for the full
  before/after tables and what each round actually fixed.
- **Retrieval itself has to be reproducible for a threshold to mean
  anything:** Chroma's HNSW index defaults to an approximate search tuned for
  speed at web scale, which measurably returned different top-k sets for the
  *same* query on different calls at this corpus's edge cases. Fixed by
  raising `hnsw:search_ef` to 400 at index-build time (`ingest/
  build_index.py`) — negligible cost at 6,448 vectors, and it's what made the
  margin calibration above trustworthy in the first place.
- **Deterministic path** (`rag/deterministic.py`) skips retrieval and
  generation entirely for a target ~40% of served questions
  (`DETERMINISTIC_ROUTE_RATE`): Kent's grade-3 marking and Hering's
  `important`/`clinical` flags already *are* the answer, so the question,
  options, and correct index come straight from a database row —
  structurally zero hallucination risk, nothing to verify. The LLM is used
  for exactly one thing, the mnemonic and exam tip (`task="coaching"`),
  logged with `path: "deterministic"` in `logs/metrics.jsonl` so the actual
  share is measurable rather than assumed. "Repertory" always routes here —
  it's the only way that subject is served at all, now that Kent isn't
  embedded. Falls back to the generative pipeline if a random draw can't find
  eligible material, so this never fails just because the dice landed badly.

### 3. Model and inference

Every LLM call goes through one function (`rag/llm.py::call`) keyed by task name.
Adding a call forces an explicit cost decision.

| Task | Model (free tier) | Model (paid) | Why |
|---|---|---|---|
| MCQ generation | Flash-Lite | Flash-Lite | bounded, extraction-shaped |
| Distractor generation | Flash-Lite | **Flash** | hardest reasoning step |
| Grounding verification | Flash-Lite | **Flash** | judging must not slip |
| Coaching text | Flash-Lite | Flash-Lite | cheap, low stakes |

- **Structured output**: Pydantic schemas passed as `response_schema`, so the
  model is constrained at decode time instead of us regex-scraping prose.
- **Thinking disabled** (`thinking_budget=0`) — thinking tokens bill as output
  and buy nothing on extraction-shaped work.
- **Temperature split**: 0.35 for generation (variety), 0.0 for verification
  (a judge must be deterministic).
- Uses the current `google-genai` SDK. The older `google-generativeai` package is
  deprecated; most tutorials still show it.

### 4. Serving, scale and cost

Gradio `Blocks` on HF Spaces. Ingestion is offline and never runs at request
time.

`python -m evaluation.cost_model --questions 50000` prints the monthly bill
across models using **token counts observed from `logs/metrics.jsonl`**, not
guesses — blended across the deterministic (~1 call) and generative (~3
call) paths at the real `DETERMINISTIC_ROUTE_RATE` mix, not a flat
calls-per-question guess. At 50K questions/month the routed design measured
**~$13/mo** versus **~$819/mo** for a flagship-default build (`gpt-5.5`) — and
$0 during the free tier. Re-run it yourself; it's cheap to reproduce and the
figure will drift as real usage accumulates in the log.

The comparison is honest about its floor: GPT-4.1 nano matches Flash-Lite on
price. The real edge is the ongoing free tier, a single-SDK cascade, and a 1M
context window that swallows retrieved chunks without truncation games.

### 5. Eval, monitoring and guardrails

**Guardrails (runtime):**
1. Strict grounding instruction in the system prompt.
2. Schema forces a verbatim `supporting_quote` — hard to fabricate a quote for a
   fact that isn't there.
3. Deterministic fuzzy check that the quote really appears in the source. We do
   not take the judge's word for something a string match can settle.
4. A separate verifier model adversarially checks every claim. Fail → regenerate.
   Fail again → refuse.
5. Confidence gate on retrieval (above), with sparse single-line fields
   excluded from ever being the answer source.
6. The student's message is only ever parsed as an answer or a topic, never
   passed to the model as instructions — a small but real injection surface
   reduction.
7. ~40% of served questions (`rag/deterministic.py`) skip 1–6 entirely by not
   needing them — the answer is a database row, so there's structurally
   nothing for a model to hallucinate.

**Eval (offline, pre-deploy):** `python -m evaluation.run_eval`

| Metric | What it catches |
|---|---|
| `hallucination_rate` | verifier leaking |
| `quote_validity` | generation drift (deterministic, no LLM opinion) |
| `refusal_accuracy` | broken confidence gate (out-of-corpus probes) — reported with raw counts and a Wilson 95% CI, not just a point estimate |
| `false_refusal_rate` | gate set too tight — same CI treatment |
| `retrieval_top1_sim_*` | calibration data for the threshold |
| `retrieval_hit_rate_at_k` | separates retrieval quality from generation quality — a topic can fail either one for different reasons, and only this tells you which (needs `expect_locator` in the goldset row) |
| `latency` (`p50`/`p95`, overall + by task + by generative/deterministic path) | from `logs/metrics.jsonl`; the deterministic path's one cheap call should read very differently from the generative pipeline's three |

The goldset (62 cases: 35 answerable, 27 refuse — see "Known limits" for why
it grew from 16) deliberately includes out-of-corpus probes across several
domains (modern pharma/biotech, other CAM systems, unrelated technical/
humanities topics, and adversarial near-miss anatomy/pathology terms). A
homoeopathy bot that answers those has a broken gate, and only a negative
test will tell you.

**Deterministic path eval (separate, offline):** `python -m evaluation.run_eval
--repertory` checks `rag/deterministic.py`'s Kent-backed path specifically —
100+ rubric-to-remedy and remedy-to-rubric cases sampled straight from
`data/repertory.db` (`evaluation.build_repertory_goldset`, never hand-written),
targeted at an *exact* rubric/remedy rather than a random draw. Reports
`answer_correctness` (should be 1.0 — it's a database lookup; anything less
means the generator has an actual bug) and `distractor_validity` (no wrong
option secretly also grades 3). This is a different question from the main
eval above: it never touches retrieval, the confidence gate, or the verifier,
so it can't tell you anything about hallucination or refusals — only whether
the "look it up" path copies the database correctly.

**Monitoring:** every call appends model, task, latency, and token counts to
`logs/metrics.jsonl`. That file feeds the cost model, so cost estimates and
production reality never drift apart.

---

## Layout

```
app.py                        Gradio entry
config.py                     routing table, thresholds, limits
core/ratelimit.py             token bucket + jittered retry
core/session.py               state machine, grading, weak-topic tracking
core/remedies.py               cross-book remedy name canonicalisation
core/hering_text.py            shared Hering running-header-marker stripping
core/grounding.py               deterministic quote-presence check
out/*.json                     the corpus -- 7 books, extracted by extract_homeoint.py
ingest/build_chunks.py          one chunk-builder function per book schema
ingest/build_index.py           offline vector indexer (Chroma), tiers via --books
ingest/build_repertory.py       offline Kent SQLite builder (data/repertory.db)
rag/schemas.py                  typed LLM contracts
rag/prompts.py                  prompts (config, not code)
rag/llm.py                      the one place an LLM is called
rag/retriever.py                filtered search, confidence gate, sparse-source exclusion, distractor mining
rag/repertory.py                Kent keyword/BM25 lookup + remedy -> grade-3 reverse index
rag/generator.py                generate -> distractors -> verify -> serve (routes to deterministic first)
rag/deterministic.py            zero-LLM-fact MCQs from Kent/Hering; LLM only for mnemonic + exam tip
evaluation/run_eval.py          the numbers behind "it doesn't hallucinate" (--repertory: the deterministic path specifically)
evaluation/build_repertory_goldset.py  samples Kent rubric-lookup gold cases from the real database
evaluation/repertory_goldset.jsonl     generated, not hand-written -- see the script above
evaluation/cost_model.py        the numbers behind the routing decision
data/remedy_aliases.json        persisted, diffable output of core/remedies.py
data/chroma/                    vector index (Organon/Boericke/Allen/Hering/Clarke/Nash)
data/repertory.db               Kent rubrics + FTS5 index, never embedded
```

## Known limits

- In-process session state; single replica only.
- **The goldset grew from 16 to 62 cases (2026-07-27) because the old numbers
  weren't statistically trustworthy, and the bigger sample changed the
  headline result.** At n=6 refuse-cases, a binomial proportion's standard
  error is `sqrt(p(1-p)/n)` ≈ ±15 points — enough that two back-to-back runs
  of the *same* code and config had actually printed `refusal_accuracy` 0.833
  and 1.0. `--repeats 3` didn't fix this either: it re-samples the same 10
  answerable topics three times each (pseudo-replication), so
  `false_refusal_rate`'s real effective n was closer to 10 than 30.

  The expansion (`evaluation/goldset.jsonl`, still built by hand-verifying
  each case against real source text, never fabricated) is now 35 answerable
  cases — 15 Organon aphorisms (up from 5, spread §8–§288, each locator
  confirmed by reading the actual aphorism text), 15 Materia Medica remedies
  (up from 5; well-known polychrests plus two nosodes, Psorinum and
  Tuberculinum, checked against `out/boericke_materia_medica.json`), and 5
  `subject: "Repertory"` cases (checked against `rag.repertory
  .remedies_with_grade3()`) added specifically to confirm the *routing*
  itself — that `subject == "Repertory"` reaches the deterministic path in
  the main eval, a different question from `--repertory`'s own correctness
  check of that path — plus 27 refuse cases (up from 6) spanning modern
  pharma/biotech, other complementary-medicine systems that are genuinely not
  homoeopathy (Ayurveda, TCM, acupuncture, naturopathy, Unani), unrelated
  tech/finance/current-events, unrelated humanities/general-knowledge, and
  adversarial near-miss probes — real anatomy/pathology/pharmacology terms
  chosen because they sound homoeopathy-adjacent without being covered by
  this corpus. `run_eval.py` now reports `refusal_accuracy` and
  `false_refusal_rate` as raw counts plus a Wilson 95% CI (`_wilson_ci` /
  `_rate_report`), not a bare point estimate — Wilson rather than the normal
  approximation because it stays inside [0, 1] and behaves better near p=0/1,
  exactly where these two metrics tend to sit.

  **Full 62-case result, `--repeats 3` (105 answerable generations + 27
  refuse probes ≈ 170 calls, within the 100–150-generation budget this
  expansion was scoped for):**

  | metric | old (n=16, 30 generations) | new (n=62, 105 generations) |
  |---|---|---|
  | served | 24 | 95 |
  | hallucination_rate | 0.0 | 0.0 |
  | quote_validity | 1.0 | 1.0 |
  | refusal_accuracy | 0.833 (5/6), 95% CI not previously computed | **0.7037 (19/27), 95% CI 0.515–0.842** |
  | false_refusal_rate | 0.2333 (7/30) | **0.1714 (18/105), 95% CI 0.111–0.255** |
  | retrieval_hit_rate_at_k | 0.8 (8/10) | 0.70 (21/30) |

  The confidence interval is honestly tighter — refusal_accuracy's 95% CI
  width shrank from an unusable ±15-ish points at n=6 to ±16 points on a
  properly asymmetric Wilson interval at n=27, and the *center* moved too:
  0.70 is a real, worse-looking number that a 6-case goldset could never have
  shown, not noise. The bigger, more varied refuse taxonomy caught genuine
  gate leaks the old 6-case set was too narrow to find — 8 of 27 refuse
  probes were wrongly answered:

  | leaked topic | category | top-1 sim | margin |
  |---|---|---|---|
  | monoclonal antibody dosing schedule | modern pharma (previously documented) | 0.657 | 0.006 |
  | acupuncture point selection for lower back pain | other CAM system | 0.713 | 0.004 |
  | naturopathic hydrotherapy protocols | other CAM system | 0.693 | n/a |
  | Unani medicine humoral theory (Tibb) | other CAM system | 0.697 | 0.001 |
  | GDPR data privacy compliance requirements | unrelated tech/finance | 0.564 | 0.029 (soft-path) |
  | histopathology of hepatic cirrhosis | near-miss pathology | 0.694 | 0.002 |
  | anatomy of the brachial plexus | near-miss anatomy | 0.697 | 0.013 |
  | physiology of the cardiac conduction system | near-miss physiology | 0.689 | 0.007 |

  Unlike the original "monoclonal" leak (a single borderline case at the
  decision boundary), most of these clear `MIN_RETRIEVAL_SIMILARITY = 0.65`
  outright rather than sneaking in through the `SOFT_SIM_FLOOR`/`MIN_MARGIN`
  path — general clinical vocabulary (anatomy, pathology, other medical
  systems' terminology) cosine-matches Boericke symptom fields (fever, back,
  abdomen, heart) and generic Organon aphorisms closely enough to clear the
  hard floor on this embedding model. This is a real, larger-sample finding,
  not a bug introduced by the expansion — closing it is a threshold/retrieval
  redesign question out of scope for this goldset-expansion task, and is left
  as the next concrete thing to fix, now that it's actually measured instead
  of assumed away by a too-small sample.
- **`MIN_RETRIEVAL_SIMILARITY = 0.65`, calibrated on the tier-1 index (Organon
  + Boericke + Allen, 6,448 chunks) and cross-validated against a full
  generative eval.** The Gemini free-tier daily quota for `gemini-2.5-flash-
  lite` on this project was initially 20 requests/day (not the 1,000/day the
  Quickstart assumes — measured from a live 429,
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: "20"`),
  which blocked a full pass at first; billing was linked mid-project and a
  full `--repeats 3` run completed after. Two rounds of real data:

  1. **Retrieval-only** (free, local embeddings, no API calls) — top-1 cosine
     similarity split into overlapping clusters (answerable: 0.661-0.795,
     refuse: 0.584-0.714), no threshold separates them perfectly, 0.65 chosen
     as the best single cutoff.
  2. **Full pipeline** (`evaluation.run_eval --repeats 3`) confirmed it, then
     surfaced a real bug the retrieval-only pass couldn't see: single-line
     fields (Boericke's `Dose`, `Relationship`) were winning rank-0 on pure
     cosine similarity and getting served as the answer source despite having
     almost no propositional content — the generator would then invent an
     explanation the passage didn't support, and the verifier (correctly)
     refused it. Fixed by excluding sparse chunks from rank-0 eligibility
     (`rag/retriever.py`) while keeping them available as distractor material.
     Before/after on the same goldset:

     | metric | before | after |
     |---|---|---|
     | served | 15 | 22 |
     | hallucination_rate | 0.0 | 0.0 |
     | quote_validity | 1.0 | 1.0 |
     | refusal_accuracy | 0.6 (3/5) | 0.8333 (5/6) |
     | false_refusal_rate | 0.6061 | 0.30 (9/30) |

     One residual gap at the time: "monoclonal antibody dosing schedule" still
     got served (from `NATRIUM SALICYLICUM § Non-Homeopathic Uses`, an
     aspirin-dosing passage). See below for how this was eventually closed —
     not by one clean patch, but by three small fixes together plus an
     unrelated bug in the search layer itself.

- **The deterministic path (task 5) used to bypass the confidence gate
  entirely for out-of-corpus queries — found via a `refusal_accuracy`
  regression from 0.83 to 0.17.** `DETERMINISTIC_ROUTE_RATE`'s random roll
  used to fire regardless of topic; since `rag/deterministic.py` never
  touches retrieval or a similarity check, a fired roll always served
  *something*, confidently, no matter what was asked. Verified directly:
  routing "Bitcoin consensus mechanism" through it served a real question
  about Anantherum Muricatum's fever symptoms. Fixed by gating the roll on
  `subject in DETERMINISTIC_ELIGIBLE_SUBJECTS` (`config.py`) — the two
  subjects (`Repertory`, `Materia Medica`) the deterministic generator can
  actually speak to — so an unstructured query (`subject=None`, as every
  adversarial goldset probe is) or a subject it has no content for
  (`Homoeopathic Philosophy`) always goes through the real, retrieval-gated
  path instead. Restored `refusal_accuracy` to 0.8333 (5/6) with zero new
  false refusals.

  Two "close the last leak" ideas were tested against real data and both
  failed to separate cleanly, rather than being assumed to work:
  - *Top-1/top-2 margin check*: legitimately answerable topics ("chronic
    miasms" 0.0062, "Arsenicum album" 0.0052) measured a **smaller** margin
    than the "monoclonal" leak (0.0067) — a margin threshold tight enough to
    catch the leak refuses good topics too.
  - *Raw top-k pattern check* ("is the top-k dominated by one generic field
    across unrelated remedies?"): the leak's raw top-8 are all `§ Dose` from
    different remedies, but "Sulphur"'s (legitimately answerable) raw top-8
    are almost as uniformly `§ Relationship` from different remedies — same
    shape, opposite correctness. Even after excluding sparse fields,
    Sulphur's own content ranks *below* several unrelated remedies (0.652 vs.
    Psorinum's 0.71) — bare single-word remedy names just don't embed
    distinctively against long prose here, so a stricter check would newly
    refuse good topics rather than only catching bad ones.

  Given neither is free, the one residual leak stayed an accepted, documented
  gap at the time rather than trading it for new false refusals — closed
  properly in the follow-up round below.

- **The "monoclonal" leak and the false-refusal rate were closed together, by
  three small fixes plus one bug in the search layer itself — not one clean
  patch.** Re-investigated after the routing fix above, because neither
  problem was actually a single-threshold issue:

  1. **Exclude `Non-Homeopathic Uses` from rank-0 eligibility, alongside
     `Dose`/`Relationship`.** Only 3 of 688 Boericke records carry this field,
     so it's not a load-bearing rule, but it's principled rather than
     overfit: the field documents allopathic/conventional use of a
     substance, which a UPSC Homoeopathy question is never built from, and
     its generic clinical vocabulary is exactly what let "monoclonal antibody
     dosing schedule" cosine-match an unrelated remedy's allopathic-use note.
     **This alone did not close the leak** — the next-best chunk
     (`TUBERCULINUM § Fever`, containing the literal sentence "repeat dose
     every two hours") is genuine, legitimate homeopathic content that
     happens to share the word "dose". No chunk-eligibility rule can exclude
     real content; this is an honest limit of a pure-embedding-similarity
     gate, not a bug to patch further.
  2. **A second, orthogonal accept path: `SOFT_SIM_FLOOR` + `MIN_MARGIN`**
     (`config.py`, enforced in `rag/retriever.py::retrieve_for_question`).
     Accept if top-1 clears `MIN_RETRIEVAL_SIMILARITY` outright, **or** it
     clears the lower `SOFT_SIM_FLOOR` *and* leads the next-best non-sparse
     candidate by `MIN_MARGIN` — rescuing real topics that score just under
     the hard floor without lowering that floor (which would also re-admit
     already-refused leaks). Calibrated with a zero-cost retrieval-only sweep
     (no LLM calls) before touching the real eval:

     | `MIN_MARGIN` | refusal_accuracy | false_refusal_rate | newly rescued vs. hard-floor-only |
     |---|---|---|---|
     | 0.010 | 0.667 (4/6) | 0.200 | Nux vomica, **CRISPR gene editing protocol** (bad — a leak) |
     | **0.015–0.020** | **0.833 (5/6)** | **0.200** | Nux vomica only (clean) |
     | 0.030–0.050 | 0.833 (5/6) | 0.300 | nothing (too strict to help) |

     `SOFT_SIM_FLOOR` turned out not to matter in the 0.55–0.60 range tested
     — every real candidate there already sits above all three values, so it
     doesn't discriminate anything on this goldset. Settled on
     `SOFT_SIM_FLOOR=0.55`, `MIN_MARGIN=0.02`: a real, calibrated, but
     **partial** fix — Arsenicum album (margin 0.005) and Lycopodium (no
     second non-sparse candidate to measure a margin against at all) aren't
     rescuable this way, and stay refused.
  3. **Loosened `VERIFY_SYSTEM`** (`rag/prompts.py`) to reject only claims
     that are fabricated, unsupported, or *contradict* the passage — not
     ones that merely paraphrase it differently than the checker itself
     would phrase it. "law of similars" and "Sulphur" were failing
     verification on wording/emphasis, not fabrication.
  4. **Unplanned but load-bearing: Chroma's HNSW search was not
     reproducible for the same query.** Found while re-calibrating the
     margin above — `search("Nux vomica", subject="Materia Medica")`
     returned 1 non-sparse candidate in 3 of 4 repeated calls and 2 in the
     4th, purely from HNSW's default approximate recall (`hnsw:search_ef`
     defaults to a speed-optimized value meant for web-scale collections).
     At 6,448 vectors the cost of near-exhaustive recall is negligible, so
     `ingest/build_index.py` now sets `"hnsw:search_ef": 400` at collection
     creation. Verified: 5/5 repeated calls returned identical results after
     rebuilding, versus 3/4 before. This matters for the *entire* confidence
     gate, not just the margin check — a threshold is only as meaningful as
     the retrieval feeding it is reproducible.

  **Result, full goldset, `--repeats 3`, all four fixes together:**

  | metric | before this round | after |
  |---|---|---|
  | served | 23 | 26 |
  | hallucination_rate | 0.0 | 0.0 |
  | quote_validity | 1.0 | 1.0 |
  | refusal_accuracy | 0.833 (5/6) | **1.0 (6/6)** |
  | false_refusal_rate | 0.267 | **0.133** |
  | retrieval_hit_rate_at_k | 0.8 | 0.8 |

  `refusal_accuracy` hitting 1.0 include some run-to-run luck at "monoclonal"'s
  exact decision boundary (the HNSW fix makes retrieval reproducible, but the
  underlying embedding similarity for that query is still a near-tie between
  legitimate and irrelevant content) — treat 1.0 as "very good, verify again
  as the goldset grows" rather than "mathematically guaranteed forever."

- **`Homoeopathic Philosophy` subject has zero indexed content.** Organon
  chunks are tagged `subject: "Organon of Medicine"`; there is no philosophy-
  specific book in the current corpus. The goldset's one "Homoeopathic
  Philosophy" case ("direction of cure") is set to `expect: "refuse"`
  accordingly — it isn't a threshold problem, retrieval correctly finds
  nothing to filter on. The subject filter itself is intentionally still
  offered (see `config.SUBJECTS`, `core/session.py`) rather than removed,
  because the fix is to add content, not hide the gap.

  **To close it:** `extract_homeoint.py` already has a ready-to-run pipeline
  for this — `BOOKS["kentlect"]` (base
  `http://www.homeoint.org/books3/kentlect/`, `layer: "philosophy"`, one prose
  chunk per lecture, each carrying `organon_aphorisms` / `organon_mentions`
  cross-references) — it has just never been executed, so
  `out/kent_lectures_philosophy.json` doesn't exist yet. Steps to add it:
  1. Run `python3 extract_homeoint.py --book kentlect --limit 5` as a smoke
     test, then `--book kentlect` for the full crawl (polite ~1 req/s against
     a live site — budget the time, and run `--stats` after per the extractor's
     own guidance).
  2. Write a `chunk_kentlect()` builder in `ingest/build_chunks.py` — one
     chunk per lecture (per the handoff's "never split/merge" pattern used for
     Organon), applying the same sparse-field word-count floor.
  3. Add `"kentlect": "Homoeopathic Philosophy"` to `BOOK_SUBJECT` in
     `ingest/build_index.py` and include it in a tier (it's small — likely
     tier-1-sized).
  4. Rebuild the index, then flip the goldset's "direction of cure" row back
     to `expect: "answerable"` (and consider adding 1-2 more Philosophy cases)
     and re-run `evaluation.run_eval` to confirm it's actually retrievable
     before calling the gap closed.
- Free-tier prompts may be used to improve Google's products. Fine here (public-
  domain corpus, no student PII), but know the boundary before adding accounts.
- Distractor quality on Flash-Lite is the weakest link. Set `FREE_TIER_MODE=0`
  and it routes to Flash — except MCQ generation itself, which is hardcoded to
  Flash-Lite in `config.ROUTES` regardless of `FREE_TIER_MODE`, so a Flash-Lite
  outage or quota exhaustion (see above) blocks question generation either way.
