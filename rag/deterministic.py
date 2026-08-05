"""Deterministic MCQs — zero LLM calls for the question, options, or answer.

Kent's grade-3 marking and Hering's `important`/`clinical` flags already ARE
the answer; the fact doesn't need generating, retrieving, or verifying, only
citing. The LLM is used for exactly one thing here — the mnemonic and exam
tip — which are genuinely creative and have no "correct" value to get wrong
(see rag/prompts.py's MNEMONIC_SYSTEM). Verification is skipped entirely:
there is nothing to verify.

Emits the same `MCQ` schema as rag/generator.py's LLM path, and the same
`(MCQ, Chunk, Verdict)` return shape as `rag.generator.make_question` — Chunk
and Verdict stand in for "there was no retrieval or verification step" so
app.py and evaluation/run_eval.py don't need to branch on which path served
a question.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from config import ROOT
from core import hering_text
from core.remedies import canonical
from rag import prompts
from rag.llm import call
from rag.repertory import (
    Rubric,
    get_rubric,
    grade3_rubrics_for_remedy,
    random_rubrics_with_grade3,
    remedies_with_grade3,
    sample_rubrics,
)
from rag.schemas import MCQ, Chunk, MnemonicTip, Verdict

OUT_DIR = ROOT / "out"

MAX_ATTEMPTS = 8  # random draws can dead-end (too few distractors); retry with a fresh draw before giving up


class DeterministicUnavailable(Exception):
    """No deterministic question could be constructed after MAX_ATTEMPTS tries."""


@dataclass
class _Draft:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    supporting_quote: str
    difficulty: str
    locator: str
    book: str
    url: str | None


# ---------------------------------------------------------- Kent: rubric -> grade-3 remedy
def _rubric_to_remedy(rubric_id: int | None = None) -> _Draft | None:
    if rubric_id is not None:
        rubric = get_rubric(rubric_id)
        if rubric is None:
            return None
    else:
        hits = random_rubrics_with_grade3(k=1)
        if not hits:
            return None
        rubric = hits[0]

    grade3 = rubric.grade3
    grade1_unique: dict[str, object] = {}
    for r in rubric.remedies:
        if r.grade == 1:
            grade1_unique.setdefault(r.remedy_raw, r)
    if not grade3 or len(grade1_unique) < 3:
        return None

    correct = random.choice(grade3)
    pool = [r for name, r in grade1_unique.items() if name != correct.remedy_raw]
    if len(pool) < 3:
        return None
    distractors = random.sample(pool, 3)

    options_data = [correct] + distractors
    random.shuffle(options_data)
    correct_index = options_data.index(correct)

    return _Draft(
        question=f'Which remedy is graded 3 (bold) under the rubric "{rubric.rubric}" ({rubric.chapter}) '
                 f"in Kent's Repertory?",
        options=[r.remedy_raw for r in options_data],
        correct_index=correct_index,
        explanation=(
            f'"{rubric.rubric}" lists {correct.remedy_raw} at grade 3 (bold); '
            f'{", ".join(d.remedy_raw for d in distractors)} are listed at grade 1 (plain) under the same rubric.'
        ),
        supporting_quote=rubric.rubric,
        difficulty="moderate",
        locator=f'{rubric.chapter} — "{rubric.rubric}"' + (f", p.{rubric.page}" if rubric.page else ""),
        book="Kent, Repertory",
        url=rubric.url,
    )


# ---------------------------------------------------------- Kent: remedy -> grade-3 rubric (reverse)
def _remedy_to_rubric(remedy: str | None = None) -> _Draft | None:
    if remedy is None:
        names = remedies_with_grade3(k=1)
        if not names:
            return None
        remedy = names[0]

    hits = grade3_rubrics_for_remedy(remedy, k=300)
    if not hits:
        return None

    # Kent's flattened schema repeats bare sub-rubric labels ("after") across
    # many unrelated parents; dedupe by rubric *text* so two options never
    # look identical regardless of which row backs them.
    by_text: dict[str, Rubric] = {}
    for r in hits:
        by_text.setdefault(r.rubric, r)
    unique_hits = list(by_text.values())
    correct = random.choice(unique_hits)

    exclude_ids = [r.id for r in hits]  # every rubric where this remedy is grade-3, not just the chosen one
    pool_rows = sample_rubrics(correct.chapter, exclude_ids, k=20)
    pool: dict[str, Rubric] = {}
    for r in pool_rows:
        if r.rubric != correct.rubric:
            pool.setdefault(r.rubric, r)
    if len(pool) < 3:
        return None
    distractors = random.sample(list(pool.values()), 3)

    options_data = [correct] + distractors
    random.shuffle(options_data)
    correct_index = options_data.index(correct)

    display = next(g.remedy_raw for g in correct.remedies if g.remedy_canonical == remedy and g.grade == 3)

    return _Draft(
        question=f"Under which rubric is {display} ({remedy}) graded 3 (bold) in Kent's Repertory?",
        options=[r.rubric for r in options_data],
        correct_index=correct_index,
        explanation=(
            f'{display} is listed at grade 3 (bold) under "{correct.rubric}" ({correct.chapter}); '
            f"the other options are rubrics in the same chapter where it is not graded 3."
        ),
        supporting_quote=correct.rubric,
        difficulty="hard",
        locator=f'{correct.chapter} — "{correct.rubric}"' + (f", p.{correct.page}" if correct.page else ""),
        book="Kent, Repertory",
        url=correct.url,
    )


# ---------------------------------------------------------------- Hering
_hering_cache: list[dict] | None = None


def _hering_symptoms() -> list[dict]:
    """Flatten Hering into individual, marker-stripped symptom records.
    Cached in-process: 413 remedies, not large, and every deterministic draw
    from Hering needs this same flat pool."""
    global _hering_cache
    if _hering_cache is not None:
        return _hering_cache

    with (OUT_DIR / "hering_guiding_symptoms.json").open(encoding="utf-8") as fh:
        records = json.load(fh)

    out: list[dict] = []
    for rec in records:
        remedy_raw = rec["remedy"]
        remedy = canonical(remedy_raw) or remedy_raw
        markers = hering_text.markers_for(remedy_raw, rec.get("abbrev", ""))
        for sec in rec.get("sections", []):
            for i, s in enumerate(sec.get("symptoms", [])):
                text = s.get("text", "")
                if i == 0:
                    text = hering_text.strip_marker(text, markers)
                if not text:
                    continue
                out.append({
                    "remedy": remedy,
                    "section_no": sec["no"],
                    "section_name": sec["name"],
                    "text": text,
                    "important": bool(s.get("important")),
                    "clinical": s.get("clinical"),
                    "url": rec["url"],
                })
    _hering_cache = out
    return out


def _hering_keynote(remedy: str | None = None) -> _Draft | None:
    """Hering `important: true` (19% of symptoms) -> high-yield keynote question."""
    symptoms = _hering_symptoms()
    important = [s for s in symptoms if s["important"]]
    if not important:
        return None
    if remedy is not None:
        candidates = [s for s in important if s["remedy"] == remedy]
        if not candidates:
            return None
        correct = random.choice(candidates)
    else:
        correct = random.choice(important)

    same_section: dict[str, dict] = {}
    for s in important:
        if s["section_no"] == correct["section_no"] and s["remedy"] != correct["remedy"]:
            same_section.setdefault(s["remedy"], s)
    if len(same_section) < 3:
        return None
    distractors = random.sample(list(same_section.values()), 3)

    options_data = [correct] + distractors
    random.shuffle(options_data)
    correct_index = options_data.index(correct)

    return _Draft(
        question=f'Which remedy is characterized by this keynote symptom: "{correct["text"]}"?',
        options=[s["remedy"] for s in options_data],
        correct_index=correct_index,
        explanation=(
            f'This is recorded as an important (bold) symptom for {correct["remedy"]} under the '
            f'"{correct["section_name"]}" section of Hering\'s Guiding Symptoms.'
        ),
        supporting_quote=correct["text"],
        difficulty="moderate",
        locator=f'{correct["remedy"]} § {correct["section_name"]}',
        book="Hering Guiding Symptoms",
        url=correct["url"],
    )


def _hering_clinical() -> _Draft | None:
    """Hering `clinical` (2%, the theta marker) -> "in which condition is symptom X indicated"."""
    symptoms = _hering_symptoms()
    clinical_syms = [s for s in symptoms if s["clinical"]]
    if not clinical_syms:
        return None
    correct = random.choice(clinical_syms)

    def _pool(candidates: list[dict]) -> dict[str, dict]:
        by_clinical: dict[str, dict] = {}
        for s in candidates:
            if s["clinical"] != correct["clinical"]:
                by_clinical.setdefault(s["clinical"], s)
        return by_clinical

    pool = _pool([s for s in clinical_syms if s["section_no"] == correct["section_no"]])
    if len(pool) < 3:
        pool = _pool(clinical_syms)  # section too thin -- widen corpus-wide rather than fail outright
    if len(pool) < 3:
        return None
    distractors = random.sample(list(pool.values()), 3)

    options_data = [correct] + distractors
    random.shuffle(options_data)
    correct_index = options_data.index(correct)

    return _Draft(
        question=f'In which condition is this symptom of {correct["remedy"]} clinically indicated: '
                 f'"{correct["text"]}"?',
        options=[s["clinical"] for s in options_data],
        correct_index=correct_index,
        explanation=(
            f'Hering marks this exact symptom with the clinical (θ) annotation "{correct["clinical"]}" '
            f'for {correct["remedy"]}.'
        ),
        supporting_quote=correct["text"],
        difficulty="hard",
        locator=f'{correct["remedy"]} § {correct["section_name"]}',
        book="Hering Guiding Symptoms",
        url=correct["url"],
    )


_BUILDERS = [_rubric_to_remedy, _remedy_to_rubric, _hering_keynote, _hering_clinical]
_REMEDY_TARGETABLE = {"remedy_to_rubric", "hering_keynote"}  # the two kinds that can be biased toward a specific remedy


def _draw(kinds: list[str] | None = None, remedy: str | None = None, rubric_id: int | None = None) -> _Draft | None:
    """Draw a random deterministic question.

    `remedy`: if given (a canonical remedy name that appeared in the
    student's requested topic), bias toward the two kinds that can target a
    specific remedy rather than drawing something unrelated to what was
    asked for. Falls back to a fully random draw if that remedy has no
    eligible material (e.g. never grade-3, or no `important` symptoms).

    `rubric_id`: if given, go straight to that exact rubric via
    `_rubric_to_remedy` — used by evaluation/run_eval.py's `--repertory` mode
    to verify a specific goldset row rather than draw a random one. No
    retry loop here: a fixed rubric_id either has enough grade-1 remedies to
    build a question or it doesn't, and retrying wouldn't change that.
    """
    if rubric_id is not None:
        return _rubric_to_remedy(rubric_id)

    builders = _BUILDERS if not kinds else [b for b in _BUILDERS if b.__name__.lstrip("_") in kinds]

    if remedy is not None:
        targeted = [b for b in builders if b.__name__.lstrip("_") in _REMEDY_TARGETABLE]
        for _ in range(MAX_ATTEMPTS):
            if not targeted:
                break
            fn = random.choice(targeted)
            draft = fn(remedy)
            if draft is not None:
                return draft

    for _ in range(MAX_ATTEMPTS):
        fn = random.choice(builders)
        draft = fn()
        if draft is not None:
            return draft
    return None


def _add_mnemonic(draft: _Draft) -> MCQ:
    tip: MnemonicTip = call(
        task="coaching",
        system=prompts.MNEMONIC_SYSTEM,
        prompt=prompts.MNEMONIC_USER.format(
            question=draft.question,
            options=json.dumps(draft.options),
            correct=draft.options[draft.correct_index],
            explanation=draft.explanation,
        ),
        schema=MnemonicTip,
        temperature=0.6,  # this is the one genuinely creative call in the whole pipeline
        path="deterministic",
    )
    return MCQ(
        question=draft.question,
        options=draft.options,
        correct_index=draft.correct_index,
        explanation=draft.explanation,
        supporting_quote=draft.supporting_quote,
        mnemonic=tip.mnemonic,
        exam_tip=tip.exam_tip,
        difficulty=draft.difficulty,
    )


def make_question(
    kinds: list[str] | None = None, remedy: str | None = None, rubric_id: int | None = None,
) -> tuple[MCQ, Chunk, Verdict]:
    """Full deterministic pipeline: draw -> mnemonic -> package.

    `remedy`: canonical remedy name to bias the draw toward, if the caller
    has one (e.g. the student's requested topic resolved to a real remedy).
    Falls back to a fully random draw if that remedy has no eligible
    material, so this never fails just because a specific remedy was asked
    for.

    `rubric_id`: target one exact rubric (see `_draw`) — used to verify a
    specific goldset row, not for normal serving.

    Raises DeterministicUnavailable if nothing could be constructed (e.g. a
    freshly-built, near-empty repertory db) -- callers should fall back to
    the generative path rather than crash.
    """
    draft = _draw(kinds, remedy, rubric_id)
    if draft is None:
        raise DeterministicUnavailable("No deterministic question could be constructed.")

    mcq = _add_mnemonic(draft)

    # Stand-ins so callers can treat this identically to the generative path's
    # (MCQ, Chunk, Verdict) return -- similarity=1.0 and supported=True are
    # sentinels for "not applicable", not measured values.
    source = Chunk(
        chunk_id=f"deterministic:{draft.book}:{draft.locator}",
        text=draft.supporting_quote,
        book=draft.book,
        subject="Repertory" if draft.book == "Kent, Repertory" else "Materia Medica",
        locator=draft.locator,
        similarity=1.0,
        sparse=False,
    )
    verdict = Verdict(
        supported=True,
        unsupported_claims=[],
        quote_found_in_source=True,
        reason="deterministic — question, options, and answer come directly from a database row; nothing to verify.",
    )
    return mcq, source, verdict


__all__ = ["make_question", "DeterministicUnavailable"]


def _demo() -> int:
    """python -m rag.deterministic -- prints one question of each kind, proving
    zero LLM calls produced the question/options/answer (only the mnemonic/tip
    line comes from a model)."""
    kinds = ["rubric_to_remedy", "remedy_to_rubric", "hering_keynote", "hering_clinical"]
    for kind in kinds:
        print("=" * 70)
        print("KIND:", kind)
        try:
            mcq, source, verdict = make_question(kinds=[kind])
        except DeterministicUnavailable as exc:
            print("  unavailable:", exc)
            continue
        print("Q:", mcq.question)
        for i, opt in enumerate(mcq.options):
            marker = "*" if i == mcq.correct_index else " "
            print(f"  [{marker}] {opt}")
        print("Explanation:", mcq.explanation)
        print("Quote:", mcq.supporting_quote)
        print("Citation:", f"{source.book} — {source.locator}")
        print("Mnemonic:", mcq.mnemonic)
        print("Exam tip:", mcq.exam_tip)
        print("path: deterministic | verified:", verdict.supported, "-", verdict.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
