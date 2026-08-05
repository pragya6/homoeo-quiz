"""Deterministic grounding checks.

Deliberately dependency-free: no SDK, no network, no API key. Where a string
match can settle a question, we don't ask an LLM for its opinion. This also
means the check is unit-testable in CI without credentials.
"""

from __future__ import annotations

import difflib


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def quote_present(quote: str, source: str, threshold: float = 0.88) -> bool:
    """Is `quote` a verbatim span of `source`?

    Exact substring matching is too brittle — models normalise whitespace, smart
    quotes and ligatures. Too loose and a paraphrase slips through as a "quote",
    which would defeat the entire mechanism. A difflib ratio against the best
    matching window is the pragmatic middle: it forgives typography, not content.
    """
    q, s = normalize(quote), normalize(source)
    if not q or not s:
        return False
    if q in s:
        return True

    window = len(q)
    step = max(1, window // 4)
    for i in range(0, max(1, len(s) - window + 1), step):
        if difflib.SequenceMatcher(None, q, s[i : i + window]).ratio() >= threshold:
            return True
    return False
