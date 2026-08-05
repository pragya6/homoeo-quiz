"""Offline builder for the Kent repertory store. Run once (or whenever the
corpus changes):

    python -m ingest.build_repertory

Kent is 77% of the corpus (69,991 rubrics, 394,352 graded remedy entries) and
the worst possible text for semantic embedding — terse, abbreviation-heavy,
no sentences ("WAX, black > tympanum"). Treated as a database, not a
retrieval corpus: SQLite with an FTS5 index for keyword/BM25 rubric lookup,
plus an indexed reverse mapping (remedy -> grade-3 rubrics), and it never
touches the Chroma collection.
"""

from __future__ import annotations

import json
import sqlite3
import sys

from config import REPERTORY_DB, ROOT
from core.remedies import canonical

SOURCE = ROOT / "out" / "kent_repertory.json"

SCHEMA = """
CREATE TABLE rubrics (
    id      INTEGER PRIMARY KEY,
    chapter TEXT NOT NULL,
    rubric  TEXT NOT NULL,
    page    TEXT,
    url     TEXT
);

CREATE TABLE rubric_remedies (
    rubric_id        INTEGER NOT NULL REFERENCES rubrics(id),
    remedy_raw       TEXT NOT NULL,     -- Kent's own abbreviation, e.g. "Merc" -- the citable form
    remedy_canonical TEXT,              -- resolved via core.remedies; NULL if unresolved (~42% of Kent abbreviations, see task 1)
    grade            INTEGER NOT NULL   -- 3 = bold, 2 = italic, 1 = plain
);
CREATE INDEX idx_rr_rubric ON rubric_remedies(rubric_id);
CREATE INDEX idx_rr_remedy_grade ON rubric_remedies(remedy_canonical, grade);

-- Keyword/BM25 lookup -- not a vector index. "external content" table so the
-- indexed text lives once, in `rubrics`.
CREATE VIRTUAL TABLE rubrics_fts USING fts5(
    chapter, rubric, content='rubrics', content_rowid='id'
);
"""


def build(con: sqlite3.Connection) -> dict:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))

    con.executescript(SCHEMA)
    cur = con.cursor()

    resolved = 0
    total_remedy_rows = 0
    for rec in records:
        cur.execute(
            "INSERT INTO rubrics (id, chapter, rubric, page, url) VALUES (?, ?, ?, ?, ?)",
            (None, rec["chapter"], rec["rubric"], rec.get("page"), rec.get("url")),
        )
        rubric_id = cur.lastrowid
        grade3_names = set(rec.get("grade3", []))
        for r in rec["remedies"]:
            canon = canonical(r["name"])
            if canon:
                resolved += 1
            total_remedy_rows += 1
            # grade3 is precomputed in the source but derivable from grade==3
            # too; assert they agree rather than silently trusting either.
            if r["grade"] == 3:
                assert r["name"] in grade3_names, f"grade3 mismatch: {rec['rubric']!r} {r}"
            cur.execute(
                "INSERT INTO rubric_remedies (rubric_id, remedy_raw, remedy_canonical, grade) "
                "VALUES (?, ?, ?, ?)",
                (rubric_id, r["name"], canon, r["grade"]),
            )

    cur.execute(
        "INSERT INTO rubrics_fts(rowid, chapter, rubric) SELECT id, chapter, rubric FROM rubrics"
    )
    con.commit()

    return {
        "rubrics": len(records),
        "remedy_rows": total_remedy_rows,
        "remedy_rows_resolved": resolved,
    }


def main() -> int:
    if not SOURCE.exists():
        print(f"Missing {SOURCE}. Run the extractor first (see CORPUS_HANDOFF.md).")
        return 1

    REPERTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild from scratch, same idempotency reasoning as ingest/build_index.py.
    if REPERTORY_DB.exists():
        REPERTORY_DB.unlink()

    con = sqlite3.connect(REPERTORY_DB)
    try:
        stats = build(con)
    finally:
        con.close()

    grade3_total = None
    con = sqlite3.connect(REPERTORY_DB)
    try:
        grade3_total = con.execute(
            "SELECT COUNT(*) FROM rubric_remedies WHERE grade = 3"
        ).fetchone()[0]
        chapters = con.execute("SELECT COUNT(DISTINCT chapter) FROM rubrics").fetchone()[0]
    finally:
        con.close()

    print(f"Rubrics: {stats['rubrics']}")
    print(f"Chapters: {chapters}")
    print(f"Graded remedy entries: {stats['remedy_rows']}")
    print(f"  grade-3 entries: {grade3_total}")
    pct = 100 * stats["remedy_rows_resolved"] / stats["remedy_rows"] if stats["remedy_rows"] else 0
    print(f"  resolved to a canonical remedy: {stats['remedy_rows_resolved']} ({pct:.1f}%)")
    print(f"Built {REPERTORY_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
