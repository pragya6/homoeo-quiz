"""Generate the deterministic-repertory-path goldset from the real database.

    python -m evaluation.build_repertory_goldset

Hand-writing Kent rubric strings would be fabricated data, so this samples
directly from data/repertory.db (built by ingest/build_repertory.py) — every
case is guaranteed to actually exist and to already satisfy "answerable"
(>=1 grade-3 remedy, >=3 grade-1 remedies for a clean 4-option distractor
pool) before it's ever handed to the generator.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from rag.repertory import connection, grade3_rubrics_for_remedy, remedies_with_grade3

OUT = Path(__file__).parent / "repertory_goldset.jsonl"

N_FORWARD = 100
N_REVERSE = 10
MAX_PER_CHAPTER = 4  # stratification cap -- don't let one big chapter (Mind, Extremities) dominate; raised from 3 alongside N_FORWARD so 100 still spreads reasonably across 39 chapters


def _eligible_rubrics() -> list[dict]:
    """Rubrics with >=1 grade-3 remedy and >=3 grade-1 remedies -- the exact
    minimum a clean "1 correct + 3 distractor" MCQ needs. One grouped pass
    over rubric_remedies rather than a per-rubric COUNT subquery."""
    con = connection()
    rows = con.execute(
        """
        SELECT r.id, r.chapter, r.rubric, r.page, r.url
        FROM rubrics r
        JOIN (
            SELECT rubric_id,
                   SUM(CASE WHEN grade = 3 THEN 1 ELSE 0 END) AS grade3_n,
                   SUM(CASE WHEN grade = 1 THEN 1 ELSE 0 END) AS grade1_n
            FROM rubric_remedies
            GROUP BY rubric_id
            HAVING grade3_n >= 1 AND grade1_n >= 3
        ) g ON g.rubric_id = r.id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _remedies_for_rubric(rubric_id: int) -> list[dict]:
    con = connection()
    rows = con.execute(
        "SELECT remedy_raw, grade FROM rubric_remedies WHERE rubric_id = ?", (rubric_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def _stratified_sample(eligible: list[dict], n: int, max_per_chapter: int) -> list[dict]:
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_chapter[row["chapter"]].append(row)
    for rows in by_chapter.values():
        random.shuffle(rows)

    chapters = list(by_chapter)
    random.shuffle(chapters)
    pools = {c: iter(rows) for c, rows in by_chapter.items()}
    per_chapter_count: dict[str, int] = defaultdict(int)

    picked: list[dict] = []
    remaining = list(chapters)
    while len(picked) < n and remaining:
        progressed = False
        for c in list(remaining):
            if len(picked) >= n:
                break
            if per_chapter_count[c] >= max_per_chapter:
                remaining.remove(c)
                continue
            row = next(pools[c], None)
            if row is None:
                remaining.remove(c)
                continue
            picked.append(row)
            per_chapter_count[c] += 1
            progressed = True
        if not progressed:
            break
    return picked


def build_forward(n: int, max_per_chapter: int) -> list[dict]:
    eligible = _eligible_rubrics()
    picked = _stratified_sample(eligible, n, max_per_chapter)

    cases = []
    for i, row in enumerate(picked, start=1):
        remedies = _remedies_for_rubric(row["id"])
        grade3_all = [r["remedy_raw"] for r in remedies if r["grade"] == 3]
        grade1_pool = [r["remedy_raw"] for r in remedies if r["grade"] == 1]
        cases.append({
            "id": f"rep-{i:04d}",
            "rubric_id": row["id"],
            "chapter": row["chapter"],
            "rubric": row["rubric"],
            "correct_remedy": random.choice(grade3_all),
            "grade3_all": grade3_all,
            "distractor_pool": grade1_pool,
            "page": row["page"],
            "expect": "answerable",
            "qtype": "repertory-grade3",
            "path": "deterministic",
        })
    return cases


def build_reverse(n: int) -> list[dict]:
    names = remedies_with_grade3()
    random.shuffle(names)
    names = names[:n]

    cases = []
    for i, remedy in enumerate(names, start=1):
        rubrics = grade3_rubrics_for_remedy(remedy, k=5000)  # 5000 comfortably exceeds any remedy's real grade-3 count
        cases.append({
            "id": f"rep-rev-{i:02d}",
            "remedy": remedy,
            "qtype": "repertory-reverse",
            "expect_rubrics": [r.rubric for r in rubrics],
            "path": "deterministic",
        })
    return cases


def main() -> int:
    forward = build_forward(N_FORWARD, MAX_PER_CHAPTER)
    reverse = build_reverse(N_REVERSE)

    with OUT.open("w", encoding="utf-8") as fh:
        for case in forward + reverse:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    chapter_counts: dict[str, int] = defaultdict(int)
    for c in forward:
        chapter_counts[c["chapter"]] += 1

    print(f"Forward (repertory-grade3): {len(forward)} cases across {len(chapter_counts)} chapters")
    for chapter, n in sorted(chapter_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {chapter:20s} {n}")
    print(f"Reverse (repertory-reverse): {len(reverse)} cases")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
