---
title: HomoeoQuiz
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: "3.13"
app_file: app.py
pinned: false
---

# 🌿 HomoeoQuiz

Grounded MCQ practice for Indian homoeopathy entrance exams (UPSC Homoeopathy,
SR-ship). Every question comes from the classical, public-domain canon and
gets checked against its source passage before a student ever sees it.

The bot doesn't answer from memory. If the corpus doesn't cover something, it
says so instead of guessing. That refusal is the actual product here, not a
fallback.

For the reasoning behind each design choice, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

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

Deploying to Hugging Face Spaces? Add `GEMINI_API_KEY` as a **Secret**, and
commit `data/chroma/`, `data/repertory.db`, and `data/remedy_aliases.json`
alongside the code.

---

## How it works

1. A student picks an exam, a subject, and (optionally) a topic in the
   Gradio UI (`app.py` → `core/session.py`).
2. That request gets routed (`rag/generator.py::make_question`):
   - If the subject is `"Repertory"`, or a Materia Medica request happens to
     roll the `DETERMINISTIC_ROUTE_RATE` dice, it takes the **deterministic
     path** (`rag/deterministic.py`). The question, options, and answer all
     come straight from a Kent/Hering database row — the only LLM call is
     for the mnemonic and exam tip.
   - Otherwise it takes the **generative path**.
3. On the generative path (`rag/retriever.py`, `rag/generator.py`):
   - First, a quick keyword check for known non-homoeopathy terms. A match
     means an instant refusal, no embedding call needed.
   - Then we retrieve the top-k chunks and run them past the confidence gate
     (a similarity floor, with a margin-based fallback). Nothing clears it →
     refuse (`NotInCorpus`).
   - If it passes, we generate the MCQ from the retrieved passage,
     regenerate the distractors from nearby chunks, and verify every claim
     against the actual source quote.
   - Verification fails → retry once. Fails again → refuse
     (`GenerationFailed`) rather than hand the student something ungrounded.
4. Either way — served or refused — the call's model, task, latency, and
   token count get logged to `logs/metrics.jsonl`, which is what feeds the
   cost model and the latency numbers below.

## Current eval results

Here's the latest full run — `python -m evaluation.run_eval --repeats 3`
across all 62 goldset cases (`evaluation/goldset.jsonl`: 35 answerable, 27
refuse):

| metric | value |
|---|---|
| served | 102 |
| hallucination_rate | **0.0** |
| quote_validity | **1.0** |
| refusal_accuracy | **1.0 (27/27)**, 95% CI 0.875–1.0 |
| false_refusal_rate | **0.0286 (3/105)**, 95% CI 0.010–0.081 |
| retrieval_hit_rate_at_k | 0.8667 (26/30) |
| latency (overall) | p50 1987 ms, p95 5481 ms |

Two fixes got us here from a rougher earlier round (`refusal_accuracy` 0.70,
`false_refusal_rate` 0.25 on the same goldset):

- **Out-of-domain keyword veto** (`config.OUT_OF_DOMAIN_TERMS`,
  `rag/retriever.py::out_of_domain_term`). We were leaking on 8 refuse
  cases — mostly other CAM systems and clinical anatomy/pathology terms that
  happened to clear the similarity floor outright. This closed all 8;
  `refusal_accuracy` went from 0.70 to 1.0.
- **Identity header on Boericke/Clarke chunk text**
  (`ingest/build_chunks.py::_chunk_fields`). Turns out a bare remedy-name
  query like "Belladonna" was only scoring 0.55–0.66 against its own chunks,
  because the field text never actually mentions the remedy by name — only
  the locator metadata does. Prefixing each chunk with `"<Remedy> —
  <Field>."` before embedding fixed that; `false_refusal_rate` dropped from
  0.25 to 0.03.

The 3 remaining failures aren't bugs: one genuine retrieval miss (§188), and
two cases where the verifier correctly rejected an ambiguous source passage
(Rhus Toxicodendron, "law of similars"). See `evaluation/last_report.json`
for the details.

The full before/after tables for both fixes — plus the goldset-expansion
story that came before them — are in `README_OLD.md`.

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
rag/retriever.py                filtered search, confidence gate, out-of-domain veto, distractor mining
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

- Session state lives in-process, so realistically this only works as a
  single replica right now (see `ARCHITECTURE.md` §1 for how we'd scale it
  past that).
- **The Homoeopathic Philosophy subject has no indexed content yet.** There's
  no philosophy-specific book in the corpus, so the one Philosophy case in
  the goldset (`direction of cure`) is marked `expect: "refuse"` on purpose.
  `extract_homeoint.py` already has a pipeline ready for this
  (`BOOKS["kentlect"]`) — see `README_OLD.md` for the exact steps to close
  the gap.
- Gemini's free-tier daily quota is per-project, and it turned out to be
  well under the documented limits (see `README_OLD.md` for the live 429
  evidence). `core/ratelimit.py` throttles proactively, but a daily cap
  needs either a linked billing account or patience.
- Free-tier prompts may be used to improve Google's products. That's fine
  here — the corpus is public domain and there's no student PII — but worth
  knowing before adding real accounts.
- Distractor quality on Flash-Lite is the weakest link right now. Setting
  `FREE_TIER_MODE=0` routes most tasks to Flash, except MCQ generation
  itself, which is hardcoded to Flash-Lite regardless — so a Flash-Lite
  outage or quota exhaustion still blocks question generation either way.
- `core/ratelimit.py::_is_retryable` only retries on
  `429`/`resource_exhausted`/`503`/`unavailable`/`deadline`. A raw
  connection-level error (we hit `httpx.ReadError [WinError 10053]` live)
  isn't retried, and it takes down the whole eval run. Still needs
  hardening.
