"""Cross-book remedy name canonicalisation.

Remedy naming is inconsistent across the six remedy-bearing books: Kent uses
terse abbreviations (`Merc`, `kali-bi`), Hering carries both a full Latin name
and its own abbreviation, Boericke is ALL-CAPS Latin, Clarke/Allen are
mixed-case Latin, Nash is prose with its own remedy mentions. Every cross-book
feature (distractor mining across books, Allen<->Hering suppression, joining
the Kent reverse index to Materia Medica) needs one canonical key per remedy,
built once so it doesn't have to be re-derived — and re-guessed — everywhere
that needs it.

Hering's {remedy, abbrev} pairs seed the map (the handoff calls this the
cleanest source: one canonical Latin name and one abbreviation, paired, for
every entry). Every other book's remedy identifiers are resolved against that
seed by, in order:

  1. exact match on a normalised name key
  2. token-prefix match against an abbreviation (Kent's `kali-bi`, Hering's
     own `abbrev`, Nash's `remedy_mentions` are all "first N letters of each
     word" once you strip separators) — this is guarded against ambiguity: a
     match must be unique across canonical entries or it doesn't count
  3. fuzzy match (difflib) on full names only, for classical spelling drift
     (Hering's "Actea" vs Allen/Clarke's "Actaea") — high threshold, unique
     best match only

Anything left over is unresolved and reported, never dropped. A silently
dropped remedy is a silent hole in every cross-book feature built on top of
this table later.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import ROOT

OUT_DIR = ROOT / "out"
ALIASES_PATH = ROOT / "data" / "remedy_aliases.json"

# Books other than Hering that carry remedy identifiers we need to resolve.
# Kent is included here (task 1 extends the map "from the other five books");
# it's excluded only from vector indexing (task 4), not from aliasing.
OTHER_BOOKS = ["kent_repertory", "clarke_dictionary", "boericke_materia_medica",
               "nash_leaders", "allen_keynotes"]

_FUZZY_THRESHOLD = 0.90
_FUZZY_MARGIN = 0.05


def _load(name: str) -> list[dict]:
    with (OUT_DIR / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _tokens(raw: str) -> list[str]:
    """Split any remedy identifier into lowercase word tokens.

    Works uniformly across styles because it doesn't care what separates the
    words — hyphen (Kent's `kali-bi`), period+nbsp (Hering's `Act.\xa0r.`),
    plain space (Nash's `Kali bi`) all reduce to the same token list.
    """
    s = raw.replace("\xa0", " ").lower()
    return re.findall(r"[a-z]+", s)


def _name_key(raw: str) -> str:
    return " ".join(_tokens(raw))


def _split_variants(raw: str) -> list[str]:
    """Pull separately-matchable name variants out of one raw string.

    "Actea Racemosa. (Cimicifuga.)" -> ["Actea Racemosa", "Cimicifuga"]
    "ABRUS PRECATORIUS -- JEQUIRITY" -> ["ABRUS PRECATORIUS", "JEQUIRITY"]

    Parens aren't always trailing — "China (Cinchona) Boliviana" has one mid-
    string. Stripping every paren group out of the full string (rather than
    just keeping the text before the first one) is what keeps "China
    Boliviana" and "China Officinalis" from both collapsing to bare "China".
    """
    s = raw.replace("\xa0", " ").strip()
    parens = re.findall(r"\(([^)]*)\)", s)
    main = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    main = re.sub(r"\s+", " ", main).strip().rstrip(".").strip()
    bases = [main] + [p.strip().rstrip(".").strip() for p in parens]
    bases = list(dict.fromkeys(b for b in bases if b))  # dedupe, keep order

    out: list[str] = []
    for b in bases:
        parts = re.split(r"\s*--\s*|\s+—\s+", b)
        out.extend(p.strip().rstrip(".").strip() for p in parts if p.strip())
    return out or [s]


@dataclass
class Entry:
    canonical: str
    aliases: set[str] = field(default_factory=set)   # raw strings, as seen
    name_keys: set[str] = field(default_factory=set)  # normalised name keys
    tokens_list: list[list[str]] = field(default_factory=list)  # token lists for prefix match
    sources: set[str] = field(default_factory=set)

    def add(self, raw: str, source: str) -> None:
        self.aliases.add(raw)
        self.sources.add(source)
        key = _name_key(raw)
        if key:
            self.name_keys.add(key)
        toks = _tokens(raw)
        if toks and toks not in self.tokens_list:
            self.tokens_list.append(toks)


class AliasTable:
    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self._by_name_key: dict[str, Entry] = {}
        self._by_first_token: dict[str, list[Entry]] = {}

    # ------------------------------------------------------------ building
    def _index(self, entry: Entry) -> None:
        for key in entry.name_keys:
            self._by_name_key[key] = entry
        for toks in entry.tokens_list:
            self._by_first_token.setdefault(toks[0], []).append(entry)

    def seed_from_hering(self, records: list[dict]) -> None:
        for rec in records:
            variants = _split_variants(rec["remedy"])
            canonical = variants[0]
            entry = Entry(canonical=canonical)
            for v in variants:
                entry.add(v, "hering")
            if rec.get("abbrev"):
                entry.add(rec["abbrev"], "hering")
            self.entries.append(entry)
            self._index(entry)

    def _find_exact(self, raw: str) -> Entry | None:
        for v in _split_variants(raw):
            key = _name_key(v)
            if key in self._by_name_key:
                return self._by_name_key[key]
        return None

    def _find_by_prefix(self, raw: str) -> Entry | None:
        toks = _tokens(raw)
        if not toks or len(toks[0]) < 3:
            return None
        candidates = self._by_first_token.get(toks[0], [])
        matched: set[int] = set()
        for entry in candidates:
            for name_toks in entry.tokens_list:
                if len(toks) > len(name_toks):
                    continue
                if all(name_toks[i].startswith(toks[i]) for i in range(len(toks))):
                    matched.add(id(entry))
                    break
        if len(matched) == 1:
            eid = next(iter(matched))
            return next(e for e in candidates if id(e) == eid)
        return None  # zero or ambiguous -> unresolved, not guessed

    def _find_fuzzy(self, raw: str) -> Entry | None:
        key = _name_key(raw)
        if len(key) < 5:
            return None  # too short to fuzzy-match safely
        scored = []
        for entry in self.entries:
            best = max((difflib.SequenceMatcher(None, key, k).ratio() for k in entry.name_keys), default=0.0)
            if best > 0:
                scored.append((best, entry))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] < _FUZZY_THRESHOLD:
            return None
        if len(scored) > 1 and scored[0][0] - scored[1][0] < _FUZZY_MARGIN:
            return None  # too close to call -> unresolved rather than a guess
        return scored[0][1]

    def resolve(self, raw: str) -> Entry | None:
        return self._find_exact(raw) or self._find_by_prefix(raw) or self._find_fuzzy(raw)

    def extend_from(self, book: str, records: list[dict]) -> tuple[int, int, list[str]]:
        """Resolve every distinct remedy identifier `book` uses. Returns
        (total distinct identifiers, resolved count, sorted unresolved list)."""
        idents: set[str] = set()
        if book == "kent_repertory":
            for rec in records:
                for r in rec["remedies"]:
                    idents.add(r["name"])
        elif book == "nash_leaders":
            for rec in records:
                idents.add(rec["remedy"])
                idents.update(rec.get("remedy_mentions", []))
        else:
            for rec in records:
                idents.add(rec["remedy"])

        unresolved: list[str] = []
        resolved = 0
        for ident in sorted(idents):
            entry = self.resolve(ident)
            if entry is None:
                unresolved.append(ident)
                continue
            resolved += 1
            entry.add(ident, book)
            self._index(entry)  # new alias/abbrev may help resolve later identifiers
        return len(idents), resolved, unresolved

    # ------------------------------------------------------------ persistence
    def save(self, path: Path = ALIASES_PATH) -> None:
        payload = {
            entry.canonical: {
                "aliases": sorted(entry.aliases),
                "sources": sorted(entry.sources),
            }
            for entry in sorted(self.entries, key=lambda e: e.canonical)
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build() -> tuple[AliasTable, dict]:
    table = AliasTable()
    table.seed_from_hering(_load("hering_guiding_symptoms"))

    report = {"hering_seed_entries": len(table.entries), "books": {}}
    for book in OTHER_BOOKS:
        total, resolved, unresolved = table.extend_from(book, _load(book))
        report["books"][book] = {
            "distinct_identifiers": total,
            "resolved": resolved,
            "unresolved_count": len(unresolved),
            "unresolved": unresolved,
        }
    return table, report


# ---------------------------------------------------------------- runtime API
# Built once per process and reused — callers here are ingest/build_chunks.py
# (tens of thousands of lookups per run) and later the generator's distractor
# mining, so rebuilding the matcher per call would make both O(n^2).
_runtime: dict | None = None
_runtime_table: AliasTable | None = None
_by_name_key: dict[str, str] | None = None  # name_key -> canonical, fast path


def _load_runtime() -> tuple[dict, AliasTable, dict[str, str]]:
    global _runtime, _runtime_table, _by_name_key
    if _runtime is None:
        if not ALIASES_PATH.exists():
            raise RuntimeError(
                f"{ALIASES_PATH} not found. Run `python -m core.remedies` to build it."
            )
        _runtime = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        table = AliasTable()
        by_key: dict[str, str] = {}
        for canon, data in _runtime.items():
            entry = Entry(canonical=canon)
            for a in data["aliases"]:
                entry.add(a, "persisted")
                by_key[_name_key(a)] = canon
            table.entries.append(entry)
            table._index(entry)
        _runtime_table = table
        _by_name_key = by_key
    return _runtime, _runtime_table, _by_name_key


@lru_cache(maxsize=None)
def canonical(name: str) -> str | None:
    """Best-effort canonical form for any remedy identifier from any book.

    Returns None rather than guessing when the name isn't in the table and
    doesn't resolve by the same exact/prefix/fuzzy rules used at build time.

    Cached: the fuzzy fallback is O(canonical entries) and callers (chunk
    building, distractor mining) ask about the same handful of thousand
    distinct remedy strings, often the same one many times per remedy record.
    """
    _, table, by_key = _load_runtime()
    key = _name_key(name)
    if key in by_key:
        return by_key[key]
    match = table._find_by_prefix(name) or table._find_fuzzy(name)
    return match.canonical if match else None


def aliases(canonical_name: str) -> list[str]:
    raw, _, _ = _load_runtime()
    data = raw.get(canonical_name)
    return sorted(data["aliases"]) if data else []


def main() -> int:
    table, report = build()
    table.save()

    print(f"Hering seed entries: {report['hering_seed_entries']}")
    total_all = total_resolved = 0
    for book, stats in report["books"].items():
        total_all += stats["distinct_identifiers"]
        total_resolved += stats["resolved"]
        pct = 100 * stats["resolved"] / stats["distinct_identifiers"] if stats["distinct_identifiers"] else 0
        print(f"\n{book}: {stats['resolved']}/{stats['distinct_identifiers']} resolved ({pct:.1f}%)")
        if stats["unresolved"]:
            print(f"  unresolved ({stats['unresolved_count']}):")
            for name in stats["unresolved"]:
                print(f"    - {name}")

    pct_all = 100 * total_resolved / total_all if total_all else 0
    print(f"\nOverall (excluding Hering, which seeds the map): {total_resolved}/{total_all} ({pct_all:.1f}%)")
    print(f"Canonical entries: {len(table.entries)}")
    print(f"Saved to {ALIASES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
