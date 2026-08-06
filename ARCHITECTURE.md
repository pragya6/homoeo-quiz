# Architecture

The system design rationale behind HomoeoQuiz — why each piece is built the
way it is, not just what it does. For current eval numbers and the working
request flow, see `README.md`. For the full calibration history (goldset
expansion, refusal-gate fixes, retrieval-quality fix), see `README_OLD.md`.

---

## 1. Scale and scope

Target: **2–10 concurrent students.** Sized to sit inside the Gemini free
tier — though "free tier" turned out to be per-project, not a fixed published
number: this project measured **10 RPM and a 20 req/day cap** on
`gemini-2.5-flash-lite` directly from live 429 responses, well under the
documented 15 RPM / 1,000 req/day (see `README.md` "Known limits").
`core/ratelimit.py` throttles at 8 RPM with a token bucket and retries 429s
with jittered backoff, so a burst degrades into a short wait rather than an
error — but no local throttle fixes a daily cap; that needs a linked billing
account (Gemini Tier 1) or patience.

A served question costs ~1 LLM call (`rag/deterministic.py`, Kent/Hering
facts) or ~3 (`rag/generator.py`, generate → distractors → verify), mixed
`DETERMINISTIC_ROUTE_RATE` : the rest — roughly 2.2 calls/question blended at
the default 0.4. "Repertory" always routes deterministic (Kent was never
embedded — see §2); everything else rolls the split.

Scoped to the homoeopathy core (Organon, Materia Medica, Repertory). General
medical subjects are excluded on purpose — they age, and they aren't public
domain. See `data/raw/SOURCES.md`. Homoeopathic Philosophy is in `config.
SUBJECTS` but has no indexed content yet — see `README.md` "Known limits"
for what's blocking it and the concrete steps to close it.

**Scaling path:** session state is in-process (`gr.State`). Beyond ~15 concurrent
students, move `Session` into Redis keyed by user id, put the Chroma index behind
a read-only service or swap for a managed vector DB, and flip `FREE_TIER_MODE=0`
to lift throttles and route the hard tasks to the smarter model.

## 2. Data and retrieval

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
  text. Boericke/Clarke split one chunk per `fields` entry, and each field's
  **embedded text is prefixed with an identity header** (`"Belladonna —
  Mind. <original text>"`) — field prose like "Vertigo, with falling..."
  almost never names the remedy itself, so a bare name query ("Belladonna")
  was scoring only 0.55–0.66 against its own chunks with the name living only
  in `locator` metadata, not in what actually gets vectorised. The header is
  added *after* sparsity is judged on the original text, so it doesn't change
  which chunks count as sparse. Hering splits one chunk per section per
  remedy (its fixed 1–48 schema), with the running "remedy name repeated as a
  symptom" artifact stripped and Allen-flagged near-duplicate symptoms
  suppressed (below). Allen has no `fields` (verified against the actual
  JSON, contrary to its schema doc) and Nash is prose-only, so both get
  paragraph-packed from `raw_text` instead — and Allen's prose already
  mentions the remedy by name, so it didn't need the header fix. A
  fixed-width splitter would bisect aphorism §17 and quietly destroy
  grounding; none of this does.
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
- **Out-of-domain veto, checked before retrieval** (`rag/retriever.py::
  out_of_domain_term`, `config.OUT_OF_DOMAIN_TERMS`). Some out-of-corpus
  queries genuinely clear the similarity floor outright — other CAM systems
  (acupuncture, Unani, naturopathy) and clinical anatomy/pathology
  terminology (`brachial plexus`, `histopathology`) cosine-match this
  corpus's remedy body-region fields and generic Organon aphorisms closely
  enough that no threshold separates them. That's a relevance problem, not a
  confidence problem, so it's a deterministic keyword check on the query
  text, not a similarity number — zero cost, checked before any embedding or
  LLM call. Every entry is a term that is *only* out-of-domain: generic
  words that appear in real remedy indications (`cirrhosis`, `cardiac`) are
  deliberately excluded so the veto can't cause a false refusal.
- **Confidence gate, two accept paths:** raise `NotInCorpus` and refuse unless
  EITHER top-1 cosine similarity clears `MIN_RETRIEVAL_SIMILARITY` outright,
  OR it clears a lower `SOFT_SIM_FLOOR` *and* leads the next-best non-sparse
  candidate by `MIN_MARGIN` (a real topic usually has one clear best match; a
  nonsense topic usually has several tied, so-so ones). Both are calibrated
  on the goldset, not guessed — see `README.md` "Known limits" for the
  current numbers and `README_OLD.md` for the full calibration history.
- **Sparse-source exclusion:** single-line fields (Boericke's `Dose`,
  `Relationship`/`Complementary`, Clarke's `Relations`/`Causation`) and fields
  explicitly labelled non-homoeopathic (`Non-Homeopathic Uses`) never win
  rank-0, nor does anything under a 15-word floor. None of these carry enough
  propositional content — or the right *kind* of content — to build a
  defensible exam question on; letting one win drove the generator to invent
  an explanation the passage didn't actually support. They're still eligible
  as neighbour/distractor material — exactly the kind of "related but
  different" content that makes a wrong option tempting. See
  `rag/retriever.py::retrieve_for_question`.
- **Retrieval itself has to be reproducible for a threshold to mean
  anything:** Chroma's HNSW index defaults to an approximate search tuned for
  speed at web scale, which measurably returned different top-k sets for the
  *same* query on different calls at this corpus's edge cases. Fixed by
  raising `hnsw:search_ef` to 400 at index-build time (`ingest/
  build_index.py`) — negligible cost at 6,448 vectors, and it's what made the
  margin calibration trustworthy in the first place.
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

## 3. Model and inference

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

## 4. Serving, scale and cost

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

## 5. Guardrails

1. Strict grounding instruction in the system prompt.
2. Schema forces a verbatim `supporting_quote` — hard to fabricate a quote for a
   fact that isn't there.
3. Deterministic fuzzy check that the quote really appears in the source. We do
   not take the judge's word for something a string match can settle.
4. A separate verifier model adversarially checks every claim. Fail → regenerate.
   Fail again → refuse.
5. Out-of-domain keyword veto + confidence gate on retrieval (above), with
   sparse single-line fields excluded from ever being the answer source.
6. The student's message is only ever parsed as an answer or a topic, never
   passed to the model as instructions — a small but real injection surface
   reduction.
7. ~40% of served questions (`rag/deterministic.py`) skip 1–6 entirely by not
   needing them — the answer is a database row, so there's structurally
   nothing for a model to hallucinate.

**Monitoring:** every call appends model, task, latency, and token counts to
`logs/metrics.jsonl`. That file feeds the cost model, so cost estimates and
production reality never drift apart.
