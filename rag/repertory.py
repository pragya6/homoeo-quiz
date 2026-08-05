"""Kent repertory lookups: keyword/BM25 rubric search plus the remedy -> grade-3
rubric reverse index. Deliberately not a vector search — see
ingest/build_repertory.py and the handoff for why Kent embeds badly.

Nothing here calls an LLM; it's plain SQL against data/repertory.db.
"""

from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass

from config import REPERTORY_DB

_con: sqlite3.Connection | None = None


def connection() -> sqlite3.Connection:
    global _con
    if _con is None:
        if not REPERTORY_DB.exists():
            raise RuntimeError(f"{REPERTORY_DB} not found. Run `python -m ingest.build_repertory` first.")
        _con = sqlite3.connect(REPERTORY_DB, check_same_thread=False)
        _con.row_factory = sqlite3.Row
    return _con


@dataclass
class RemedyGrade:
    remedy_raw: str             # Kent's own abbreviation -- the citable form
    remedy_canonical: str | None
    grade: int


@dataclass
class Rubric:
    id: int
    chapter: str
    rubric: str
    page: str | None
    url: str | None
    remedies: list[RemedyGrade]

    @property
    def grade3(self) -> list[RemedyGrade]:
        return [r for r in self.remedies if r.grade == 3]


def _load_remedies(cur: sqlite3.Cursor, rubric_id: int) -> list[RemedyGrade]:
    rows = cur.execute(
        "SELECT remedy_raw, remedy_canonical, grade FROM rubric_remedies "
        "WHERE rubric_id = ? ORDER BY grade DESC, remedy_raw",
        (rubric_id,),
    ).fetchall()
    return [RemedyGrade(r["remedy_raw"], r["remedy_canonical"], r["grade"]) for r in rows]


def get_rubric(rubric_id: int) -> Rubric | None:
    con = connection()
    row = con.execute(
        "SELECT id, chapter, rubric, page, url FROM rubrics WHERE id = ?", (rubric_id,)
    ).fetchone()
    if row is None:
        return None
    return Rubric(row["id"], row["chapter"], row["rubric"], row["page"], row["url"],
                  _load_remedies(con.cursor(), rubric_id))


_FTS_UNSAFE = re.compile(r'["*^:()\-]')


def _fts_query(text: str) -> str:
    """Turn free-text into a safe FTS5 MATCH expression: each word as its own
    quoted phrase, ANDed together. Kent rubrics contain punctuation FTS5's
    query syntax treats specially (">" , "," , "-"); quoting per-token avoids
    a MATCH syntax error on arbitrary input rather than trying to escape it."""
    tokens = [t for t in re.split(r"\s+", _FTS_UNSAFE.sub(" ", text).strip()) if t]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def search_rubrics(query: str, chapter: str | None = None, k: int = 10) -> list[Rubric]:
    """Keyword/BM25 rubric lookup (lower bm25() score = better match)."""
    con = connection()
    match = _fts_query(query)
    sql = (
        "SELECT r.id, r.chapter, r.rubric, r.page, r.url, bm25(rubrics_fts) AS score "
        "FROM rubrics_fts JOIN rubrics r ON r.id = rubrics_fts.rowid "
        "WHERE rubrics_fts MATCH ?"
    )
    params: list = [match]
    if chapter:
        sql += " AND r.chapter = ?"
        params.append(chapter)
    sql += " ORDER BY score LIMIT ?"
    params.append(k)

    cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    return [
        Rubric(row["id"], row["chapter"], row["rubric"], row["page"], row["url"],
               _load_remedies(cur, row["id"]))
        for row in rows
    ]


def grade3_rubrics_for_remedy(remedy_canonical: str, chapter: str | None = None, k: int = 50) -> list[Rubric]:
    """Reverse index: rubrics where `remedy_canonical` is graded 3 (bold).

    Most repertory questions run this direction, not rubric-first (per the
    handoff) -- this is why rubric_remedies is indexed on (remedy_canonical,
    grade) rather than relying on a table scan.
    """
    con = connection()
    sql = (
        "SELECT DISTINCT r.id, r.chapter, r.rubric, r.page, r.url "
        "FROM rubric_remedies rr JOIN rubrics r ON r.id = rr.rubric_id "
        "WHERE rr.remedy_canonical = ? AND rr.grade = 3"
    )
    params: list = [remedy_canonical]
    if chapter:
        sql += " AND r.chapter = ?"
        params.append(chapter)
    sql += " ORDER BY r.chapter, r.rubric LIMIT ?"
    params.append(k)

    cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    return [
        Rubric(row["id"], row["chapter"], row["rubric"], row["page"], row["url"],
               _load_remedies(cur, row["id"]))
        for row in rows
    ]


def chapters() -> list[str]:
    con = connection()
    return [r[0] for r in con.execute("SELECT DISTINCT chapter FROM rubrics ORDER BY chapter").fetchall()]


def random_rubrics_with_grade3(k: int = 1) -> list[Rubric]:
    """Random rubrics that have at least one grade-3 remedy -- the pool
    rag/deterministic.py draws "rubric -> remedy" questions from."""
    con = connection()
    cur = con.cursor()
    rows = cur.execute(
        "SELECT id, chapter, rubric, page, url FROM rubrics "
        "WHERE id IN (SELECT DISTINCT rubric_id FROM rubric_remedies WHERE grade = 3) "
        "ORDER BY RANDOM() LIMIT ?",
        (k,),
    ).fetchall()
    return [
        Rubric(row["id"], row["chapter"], row["rubric"], row["page"], row["url"],
               _load_remedies(cur, row["id"]))
        for row in rows
    ]


def remedies_with_grade3(k: int | None = None) -> list[str]:
    """Distinct canonical remedies that are grade-3 somewhere -- the pool
    rag/deterministic.py draws "remedy -> rubric" questions from."""
    con = connection()
    sql = "SELECT DISTINCT remedy_canonical FROM rubric_remedies WHERE grade = 3 AND remedy_canonical IS NOT NULL"
    rows = con.execute(sql).fetchall()
    names = [r[0] for r in rows]
    if k is not None:
        names = random.sample(names, min(k, len(names)))
    return names


def sample_rubrics(chapter: str, exclude_ids: list[int], k: int) -> list[Rubric]:
    """Random rubrics from `chapter` excluding specific ids -- used to build
    plausible-but-wrong distractor options for "which rubric" questions:
    same chapter keeps them topically adjacent rather than obviously absurd."""
    con = connection()
    cur = con.cursor()
    exclude_ids = exclude_ids or [-1]
    placeholders = ",".join("?" * len(exclude_ids))
    rows = cur.execute(
        f"SELECT id, chapter, rubric, page, url FROM rubrics "
        f"WHERE chapter = ? AND id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT ?",
        [chapter, *exclude_ids, k],
    ).fetchall()
    return [
        Rubric(row["id"], row["chapter"], row["rubric"], row["page"], row["url"],
               _load_remedies(cur, row["id"]))
        for row in rows
    ]
