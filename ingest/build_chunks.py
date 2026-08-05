"""Turn the seven corpus JSON files into flat chunk records ready to embed.

One function per schema, each returning a list of dicts with a common shape:

    {"text", "source", "layer", "locator", "url", "remedy"}

plus a few book-specific extras (e.g. Hering's `section_no`) that
`ingest/build_index.py` folds into Chroma metadata alongside the common keys.

Kent has no builder here — task 4 treats it as a database, not a retrieval
corpus, so it never becomes a vector chunk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import CHUNK_OVERLAP_CHARS, MAX_CHUNK_CHARS, ROOT
from core import hering_text
from core.remedies import canonical

OUT_DIR = ROOT / "out"

SOURCE_LABELS = {
    "organon": "Organon 6th ed.",
    "boericke": "Boericke MM",
    "allen": "Allen Keynotes",
    "clarke": "Clarke Dictionary",
    "hering": "Hering Guiding Symptoms",
    "nash": "Nash Leaders",
}


def _load(book_file: str) -> list[dict]:
    with (OUT_DIR / f"{book_file}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# Fields excluded from ever being the *sole* rank-0 answer source — not
# because they're wrong to index, but because neither is examinable
# homoeopathic material on its own. Two distinct reasons land here:
#   - structurally cross-reference lists ("Complementary: Sulph.",
#     "Antidoted by: Camph.") rather than propositional content, even when
#     they clear the word-count floor below — the eval caught the generator
#     over-reaching on these to invent an explanation the passage doesn't
#     support. Hering section 48 ("Relations") is the same content under a
#     different book's schema.
#   - explicitly *non*-homoeopathic by the source's own labelling
#     ("Non-Homeopathic Uses" documents allopathic/conventional use of the
#     substance) — a UPSC Homoeopathy question is never built from this, and
#     its generic clinical vocabulary is exactly what let an out-of-corpus
#     query like "monoclonal antibody dosing schedule" cosine-match a
#     completely unrelated remedy's allopathic-use note. Only 3/688 Boericke
#     records carry it, so this isn't a load-bearing exclusion — it's
#     correct regardless of whether it moves the eval.
SPARSE_FIELD_NAMES = {"dose", "relationship", "relations", "causation", "non-homeopathic uses"}
SPARSE_WORD_THRESHOLD = 15  # anything shorter has too little propositional content to build a defensible question on


def _is_sparse(text: str, field_name: str | None) -> bool:
    if field_name and field_name.strip().lower() in SPARSE_FIELD_NAMES:
        return True
    return len(text.split()) < SPARSE_WORD_THRESHOLD


def _record(text: str, source: str, layer: str, locator: str, url: str,
            remedy_raw: str | None = None, field_name: str | None = None, **extra) -> dict | None:
    text = text.strip()
    if not text:
        return None  # empty fields are common (Clarke's sparse Causation etc.) and carry nothing to embed
    rec = {
        "text": text,
        "source": source,
        "layer": layer,
        "locator": locator,
        "url": url,
        "remedy": canonical(remedy_raw) if remedy_raw else None,
        # Sparse chunks (single-line fields like Dose/Relationship, or just
        # short) are still fine as neighbouring/distractor material, but too
        # thin to be the passage a question is generated *from* — see
        # rag/retriever.py, which is where this actually gets enforced.
        "sparse": _is_sparse(text, field_name),
    }
    rec.update(extra)
    return rec


def _pack_paragraphs(text: str) -> list[str]:
    """Paragraph-packed fallback for prose that has no per-field structure.

    Same trade-off as the old chunker: pack paragraphs up to MAX_CHUNK_CHARS,
    carry CHUNK_OVERLAP_CHARS of trailing context into the next part so a
    fact straddling the cut survives in one piece somewhere.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    parts, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > MAX_CHUNK_CHARS and buf:
            parts.append(buf.strip())
            buf = buf[-CHUNK_OVERLAP_CHARS:] + " "
        buf += p + "\n\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


# ---------------------------------------------------------------- Organon
def chunk_organon() -> list[dict]:
    """One chunk per aphorism, never split, never merged.

    Footnotes become sibling child chunks (not appended text) so a footnote
    that's the actual examinable fact is independently retrievable — but each
    carries `parent_locator` back to its aphorism, and its own locator keeps
    the "§ N, footnote k" citation exact.
    """
    label = SOURCE_LABELS["organon"]
    out: list[dict] = []
    for rec in _load("organon_6th"):
        n = rec["aphorism_no"]
        locator = f"§ {n}"
        c = _record(rec["text"], label, "organon", locator, rec["url"])
        if c:
            out.append(c)
        for fn in rec.get("footnotes", []):
            fn_locator = f"§ {n}, footnote {fn['marker']}"
            fc = _record(fn["text"], label, "organon", fn_locator, rec["url"],
                         parent_locator=locator)
            if fc:
                out.append(fc)
    return out


# --------------------------------------------------- Boericke / Allen / Clarke
def _strip_field_artifact(value: str) -> str:
    # Clarke's field text is preceded by a heading-separator glyph the
    # extractor's normalize() produces from the source's typeset dash; it's
    # formatting, not content, so it doesn't belong in an embedded chunk.
    if value.startswith("─"):
        value = value[1:].lstrip()
    return value


def _chunk_fields(book_file: str, book_key: str, layer: str) -> list[dict]:
    label = SOURCE_LABELS[book_key]
    out: list[dict] = []
    for rec in _load(book_file):
        remedy_raw = rec["remedy"]
        for field_name, value in rec.get("fields", {}).items():
            if not isinstance(value, str):
                continue
            value = _strip_field_artifact(value)
            locator = f"{remedy_raw} § {field_name}"
            c = _record(value, label, layer, locator, rec["url"], remedy_raw, field_name=field_name)
            if c:
                # Identity header on the *embedded* text only -- locator/
                # metadata above are built from the unprefixed value, and
                # sparsity (_record -> _is_sparse) was already judged against
                # it too. Field prose ("Vertigo, with falling...") almost
                # never names the remedy itself -- verified directly:
                # bare-name queries like "Belladonna" scored only 0.55-0.66
                # against Belladonna's own 25 field chunks, because the name
                # lived only in `locator`, never in what gets vectorised.
                # See README "Known limits" for the measured before/after.
                c["text"] = f"{remedy_raw} — {field_name}. {c['text']}"
                out.append(c)
    return out


def chunk_boericke() -> list[dict]:
    return _chunk_fields("boericke_materia_medica", "boericke", "")


def chunk_allen() -> list[dict]:
    """Allen has no `fields` dict — verified against the actual JSON, which
    only carries `keynotes`/`keynotes_text` (heuristic, per the handoff),
    `modalities` and `relations` (structured but derived), and `raw_text`.
    The handoff's own caveat for this book is that `raw_text` holds the
    complete, non-heuristic content, so that's what gets paragraph-packed —
    same treatment as Nash, not the field-per-chunk treatment Boericke/Clarke
    get, because Allen genuinely doesn't have fields to split on."""
    label = SOURCE_LABELS["allen"]
    out: list[dict] = []
    for rec in _load("allen_keynotes"):
        remedy_raw = rec["remedy"]
        parts = _pack_paragraphs(rec.get("raw_text", ""))
        for i, part in enumerate(parts, start=1):
            locator = remedy_raw if len(parts) == 1 else f"{remedy_raw} (part {i})"
            c = _record(part, label, "condensed-keynotes", locator, rec["url"], remedy_raw)
            if c:
                out.append(c)
    return out


def chunk_clarke() -> list[dict]:
    return _chunk_fields("clarke_dictionary", "clarke", "")


# ---------------------------------------------------------------- Hering
def _normalize_symptom(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split()).strip().lower()


def _load_dedup_suppressions() -> dict[str, set[str]]:
    """canonical remedy -> normalized Hering symptom texts to drop.

    allen_keynotes.dedup.json flags individual Allen keynotes as near-
    duplicates of a specific Hering symptom (containment >= 0.7 -- see the
    handoff's dedup section; 230 of 5,228 comparable keynotes, min score
    among flagged pairs is 0.71, verified against the actual file). Per the
    handoff's recommended policy: index both books, but when a keynote is
    flagged, generate from Allen and suppress the Hering twin so the same
    fact isn't askable from both. The threshold is precomputed data, not
    something to retune here -- the handoff is explicit that chasing a higher
    flag count by lowering it starts pairing unrelated symptoms.
    """
    path = OUT_DIR / "allen_keynotes.dedup.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))

    out: dict[str, set[str]] = {}
    for rec in records:
        dedup = rec.get("dedup") or {}
        if dedup.get("status") != "matched":
            continue
        hering_remedy_raw = rec.get("hering_match")
        if not hering_remedy_raw:
            continue
        # Bridge Allen's and Hering's own spelling of the remedy through the
        # task-1 alias table rather than assuming either string matches the
        # other verbatim.
        canon = canonical(hering_remedy_raw) or canonical(rec["remedy"])
        if not canon:
            continue
        for kn in dedup.get("keynotes", []):
            if kn.get("duplicate") and kn.get("hering_text"):
                out.setdefault(canon, set()).add(_normalize_symptom(kn["hering_text"]))
    return out


def chunk_hering() -> list[dict]:
    """One chunk per section per remedy — whole-remedy is too large, a single
    symptom is too small to retrieve on. `no` (Hering's fixed 1-48 schema) and
    the `important`/`clinical` grade signals ride along as facets so the
    deterministic path (task 5) doesn't have to re-derive them.

    Symptoms flagged as Allen/Hering near-duplicates (task 6) are dropped
    from the joined section text here, at the individual-symptom level —
    Hering's own chunking stays section-granularity either way, this just
    keeps one specific sentence from being indexed twice under two authors.
    """
    label = SOURCE_LABELS["hering"]
    suppressions = _load_dedup_suppressions()
    out: list[dict] = []
    for rec in _load("hering_guiding_symptoms"):
        remedy_raw = rec["remedy"]
        markers = hering_text.markers_for(remedy_raw, rec.get("abbrev", ""))
        suppressed = suppressions.get(canonical(remedy_raw) or "", set())

        for sec in rec.get("sections", []):
            raw_symptoms = sec.get("symptoms", [])
            if not raw_symptoms:
                continue
            # The running-header marker, when present, is only ever the
            # section's first entry.
            cleaned = [hering_text.strip_marker(raw_symptoms[0]["text"], markers)] if raw_symptoms else []
            kept = [(cleaned[0], raw_symptoms[0])] if cleaned and cleaned[0] else []
            kept += [(s["text"], s) for s in raw_symptoms[1:] if s.get("text")]
            if suppressed:
                kept = [(t, s) for t, s in kept if _normalize_symptom(t) not in suppressed]

            text = " ".join(t for t, _ in kept)
            if not text.strip():
                continue
            locator = f"{remedy_raw} § {sec['name']}"
            clinicals = sorted({s["clinical"] for _, s in kept if s.get("clinical")})
            c = _record(
                text, label, "exhaustive", locator, rec["url"], remedy_raw,
                field_name=sec["name"],
                section_no=sec["no"],
                has_important=any(s.get("important") for _, s in kept),
                clinical=clinicals,
            )
            if c:
                out.append(c)
    return out


# ---------------------------------------------------------------- Nash
def chunk_nash() -> list[dict]:
    label = SOURCE_LABELS["nash"]
    out: list[dict] = []
    for rec in _load("nash_leaders"):
        remedy_raw = rec["remedy"]
        parts = _pack_paragraphs(rec.get("raw_text", ""))
        for i, part in enumerate(parts, start=1):
            locator = remedy_raw if len(parts) == 1 else f"{remedy_raw} (part {i})"
            c = _record(part, label, "", locator, rec["url"], remedy_raw)
            if c:
                out.append(c)
    return out


BUILDERS = {
    "organon": chunk_organon,
    "boericke": chunk_boericke,
    "allen": chunk_allen,
    "clarke": chunk_clarke,
    "hering": chunk_hering,
    "nash": chunk_nash,
}


def build(books: list[str]) -> dict[str, list[dict]]:
    unknown = set(books) - set(BUILDERS)
    if unknown:
        raise ValueError(f"Unknown book(s): {sorted(unknown)}. Known: {sorted(BUILDERS)}")
    return {book: BUILDERS[book]() for book in books}


def main() -> int:
    import sys

    books = sys.argv[1:] or list(BUILDERS)
    chunks_by_book = build(books)
    total = 0
    for book, chunks in chunks_by_book.items():
        print(f"  {book}: {len(chunks)} chunks")
        total += len(chunks)
    print(f"\nTotal: {total} chunks across {len(chunks_by_book)} book(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
