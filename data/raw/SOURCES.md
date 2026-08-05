# Corpus sources

**Rule: public domain or officially published only.** No scraped commercial
question banks, no recent copyrighted editions. This is a legal constraint and a
quality one — the exams test the classical canon, which is exactly the part that
is free.

## Where this corpus actually comes from

All content is extracted by `extract_homeoint.py` from **homeoint.org
(Médi-T)**, which serves clean HTML — not scanned PDFs. That's the point:
archive.org scans of these same books grade BAD/REVIEW for OCR damage;
nothing here needs OCR repair. Output lands in `out/*.json` — one file per
book, schemas documented in `CORPUS_HANDOFF.md`.

| File (`out/`) | Work | Author | Records |
|---|---|---|---:|
| `organon_6th.json` | Organon of Medicine, 6th ed. | Hahnemann (d. 1843) | 291 |
| `kent_repertory.json` | Repertory of the Homoeopathic Materia Medica | J. T. Kent | 69,991 |
| `clarke_dictionary.json` | A Dictionary of Practical Materia Medica | J. H. Clarke | 1,010 |
| `boericke_materia_medica.json` | Pocket Manual of Homoeopathic Materia Medica | Boericke, 1922 | 688 |
| `hering_guiding_symptoms.json` | The Guiding Symptoms of our Materia Medica | Constantine Hering | 413 |
| `nash_leaders.json` | Leaders in Homoeopathic Therapeutics | E. B. Nash | 210 |
| `allen_keynotes.json` (+ `.dedup.json`) | Keynotes and Characteristics | H. C. Allen | 180 |

Re-run a book with `python3 extract_homeoint.py --book <name>` (`organon`,
`kent`, `clarke`, `boericke`, `hering`, `nash`, `allen`, or `kentlect` — see
below). Every page fetched is cached, so a re-run only retries what actually
failed; `--stats` after a crawl is the extractor's own guidance for catching
a silently under-matched parser (an `EMPTY` row, a nonzero `gaps`, a stat
stuck at 100%).

**Do not substitute the Hpathy / homeopathybooks.in edition of Hering's
Guiding Symptoms.** Its colour-coded formatting is separately copyrighted —
homeoint's plain-text edition only.

## Not yet extracted: Homoeopathic Philosophy

`config.SUBJECTS` lists "Homoeopathic Philosophy" and the app offers it as a
topic, but there is currently **no indexed content for it** — see the
README's "Known limits" for the measured effect (a goldset case for it is
set to `expect: "refuse"` accordingly). `extract_homeoint.py` already has a
ready-to-run pipeline for the fix — `BOOKS["kentlect"]` (Kent's *Lectures on
Homoeopathic Philosophy*, one prose chunk per lecture, each cross-referencing
the Organon aphorisms it discusses) — it has simply never been run. See the
README for the full close-the-gap checklist.

**Past papers:** pull from official portals only (UPSC publishes previous papers;
state PSCs and institutes publish SR-ship papers). Use them to calibrate question
*style*, not as content to reproduce.

## Why old editions are fine here

Counter-intuitive but load-bearing: in most of medicine an old textbook is
dangerous. In classical homoeopathy the canon is fixed. The UPSC Organon
syllabus is §1–§291 of the 6th edition — that *is* the public-domain text
(§292–294 exist only in the 5th edition and are excluded by design, not an
oversight; zero gaps in §1–§291). Sulphur's drug picture in Boericke (1922)
is the same one tested in 2026. Old means canonical, not stale.

## Scope note

The exams also cover anatomy, pathology, and practice of medicine. Those *do* go
out of date and are *not* public domain. This MVP deliberately scopes to the
homoeopathy core (Organon, Materia Medica, Repertory, and — once `kentlect`
is extracted — Philosophy) where the grounding guarantee is strongest.
Extending to general subjects requires a licensed corpus — treat that as a
v2 decision, not an oversight.
