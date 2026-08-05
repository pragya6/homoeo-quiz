"""Generate-then-verify pipeline.

    retrieve -> generate MCQ -> (optionally) regenerate distractors -> verify -> serve

The verify step is what lets us claim "does not hallucinate" with a straight
face. It costs one extra call per question. At Flash-Lite prices that is roughly
$0.0004 — cheap insurance against teaching a student a fabricated aphorism.

If verification fails we retry with a fresh generation; if it fails repeatedly
we refuse. Refusing is a correct outcome, not an error.
"""

from __future__ import annotations

import json
import random

from config import (
    DETERMINISTIC_ELIGIBLE_SUBJECTS,
    DETERMINISTIC_ROUTE_RATE,
    GENERATION_TEMPERATURE,
    VERIFY_TEMPERATURE,
)
from core.grounding import quote_present
from core.remedies import canonical
from rag import deterministic, prompts
from rag.llm import call
from rag.retriever import Chunk, NotInCorpus, retrieve_for_question
from rag.schemas import MCQ, Distractors, Verdict

MAX_GENERATION_ATTEMPTS = 2


class GenerationFailed(Exception):
    """Every attempt produced an ungrounded question. Do not serve it."""


def _generate_mcq(
    exam: str, subject: str, topic: str, source: Chunk, neighbours: list[Chunk], asked: list[str]
) -> MCQ:
    neigh_text = "\n\n".join(f"[{c.locator}] {c.text[:900]}" for c in neighbours) or "(none)"
    asked_text = "\n".join(f"- {q}" for q in asked[-8:]) or "(none yet)"

    return call(
        task="mcq_generation",
        system=prompts.MCQ_SYSTEM,
        prompt=prompts.MCQ_USER.format(
            exam=exam,
            subject=subject,
            topic=topic,
            locator=source.locator,
            source=source.text,
            neighbours=neigh_text,
            asked=asked_text,
        ),
        schema=MCQ,
        temperature=GENERATION_TEMPERATURE,
    )


def _regenerate_distractors(mcq: MCQ, neighbours: list[Chunk]) -> MCQ:
    """Replace the three wrong options with corpus-mined ones.

    Routed to the smarter model when not on free tier: writing a wrong answer
    that is *tempting but defensibly wrong* is the hardest reasoning step in this
    whole system, and it's where a small model most visibly stumbles.
    """
    if not neighbours:
        return mcq

    correct = mcq.options[mcq.correct_index]
    neigh_text = "\n\n".join(f"[{c.locator}] {c.text[:900]}" for c in neighbours)

    d: Distractors = call(
        task="distractor_generation",
        system=prompts.DISTRACTOR_SYSTEM,
        prompt=prompts.DISTRACTOR_USER.format(
            question=mcq.question, correct=correct, neighbours=neigh_text
        ),
        schema=Distractors,
        temperature=GENERATION_TEMPERATURE,
    )
    if len(d.options) != 3:
        return mcq  # fall back to the original options rather than serving 3 or 5

    # Deterministic placement by hash keeps the correct index unpredictable to a
    # student but reproducible for our eval harness.
    slot = abs(hash(mcq.question)) % 4
    options = d.options[:]
    options.insert(slot, correct)
    return mcq.model_copy(update={"options": options, "correct_index": slot})


def _verify(mcq: MCQ, source: Chunk) -> Verdict:
    verdict: Verdict = call(
        task="grounding_verification",
        system=prompts.VERIFY_SYSTEM,
        prompt=prompts.VERIFY_USER.format(
            source=source.text,
            question=mcq.question,
            options=json.dumps(mcq.options),
            correct=mcq.options[mcq.correct_index],
            explanation=mcq.explanation,
            quote=mcq.supporting_quote,
        ),
        schema=Verdict,
        temperature=VERIFY_TEMPERATURE,
    )

    # Don't trust the judge's self-report on the quote — check it ourselves.
    # A cheap deterministic check beats an LLM opinion where one is available.
    verdict.quote_found_in_source = quote_present(mcq.supporting_quote, source.text)
    if not verdict.quote_found_in_source:
        verdict.supported = False
        verdict.unsupported_claims.append("supporting_quote not found verbatim in source")
    return verdict


def make_question(
    exam: str,
    subject: str | None,
    topic: str,
    asked: list[str] | None = None,
) -> tuple[MCQ, Chunk, Verdict]:
    """Full pipeline. Raises NotInCorpus or GenerationFailed — both are refusals.

    Routes some questions through rag.deterministic instead — Kent is never
    in the vector index (task 4), so "Repertory" has no other way to be
    served at all; everything else in DETERMINISTIC_ELIGIBLE_SUBJECTS gets a
    DETERMINISTIC_ROUTE_RATE chance, biased toward the requested topic if it
    resolves to a real remedy. Falls through to the generative pipeline below
    if the deterministic draw can't find eligible material
    (DeterministicUnavailable), so this never fails just because the dice
    landed on the smaller pool.

    The random roll is deliberately gated on `subject` being one the
    deterministic path can actually speak to: it never touches retrieval or
    the confidence gate, so rolling it for an arbitrary/unstructured query
    (subject=None, or a subject it has no content for, like "Homoeopathic
    Philosophy") would confidently serve an unrelated fact instead of
    refusing — verified directly, this used to happen.
    """
    if subject == "Repertory" or (
        subject in DETERMINISTIC_ELIGIBLE_SUBJECTS and random.random() < DETERMINISTIC_ROUTE_RATE
    ):
        try:
            return deterministic.make_question(remedy=canonical(topic))
        except deterministic.DeterministicUnavailable:
            pass

    asked = asked or []
    source, neighbours = retrieve_for_question(topic, subject=subject)

    last: Verdict | None = None
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mcq = _generate_mcq(exam, subject or source.subject, topic, source, neighbours, asked)
        mcq = _regenerate_distractors(mcq, neighbours)

        # Structural sanity before we spend a verification call.
        if len(mcq.options) != 4 or not (0 <= mcq.correct_index < 4):
            continue
        if len(set(o.strip().lower() for o in mcq.options)) != 4:
            continue  # duplicate options

        verdict = _verify(mcq, source)
        last = verdict
        if verdict.supported:
            return mcq, source, verdict

    raise GenerationFailed(
        f"Could not produce a grounded question for '{topic}'. "
        f"Last verdict: {last.reason if last else 'n/a'}"
    )


__all__ = ["make_question", "GenerationFailed", "NotInCorpus"]
