"""Shared Hering symptom-text cleaning.

Every section's symptom list opens with a running-header repeat of the
remedy's abbreviation or full name (e.g. "Abies. n. ] Pain after a hearty
meal." or a bare "Act. sp."). It isn't a symptom, and the source marks it
`important: true` regardless. Used by ingest/build_chunks.py (section-level
chunking) and rag/deterministic.py (individual-symptom deterministic
questions) alike, so it's factored out here instead of duplicated.
"""

from __future__ import annotations

import re

_CACHE: dict[tuple, re.Pattern | None] = {}


def _marker_pattern(candidates: list[str]) -> re.Pattern | None:
    key = tuple(candidates)
    if key in _CACHE:
        return _CACHE[key]
    parts = []
    for c in candidates:
        c = c.replace("\xa0", " ").strip()
        if not c:
            continue
        esc = re.escape(c)
        esc = re.sub(r"(?:\\ )+", r"\\s+", esc)  # tolerate space/nbsp drift
        parts.append(esc)
    pat = re.compile(r"^\s*(?:" + "|".join(parts) + r")\.?\s*\]?\s*", re.IGNORECASE) if parts else None
    _CACHE[key] = pat
    return pat


def markers_for(remedy_raw: str, abbrev: str) -> list[str]:
    main_name = re.sub(r"\s*\([^)]*\)\s*", " ", remedy_raw).strip().rstrip(".").strip()
    return [m for m in (abbrev, main_name) if m]


def strip_marker(text: str, markers: list[str]) -> str | None:
    """Strip the running-header marker from one symptom's text.

    Returns None if the marker was the entire "symptom" (nothing else to keep).
    """
    t = text.replace("\xa0", " ")
    pat = _marker_pattern(markers)
    if not pat:
        return text
    m = pat.match(t)
    if not m:
        return text
    rest = t[m.end():].strip()
    return rest or None
