"""Typed contracts between the LLM and the app.

These double as Gemini `response_schema` values, so the model is constrained at
decode time rather than us regex-scraping prose afterwards.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class MCQ(BaseModel):
    """One exam-format question, fully grounded in a single source chunk."""

    question: str = Field(description="Exam-style stem, one sentence, no preamble.")
    options: List[str] = Field(description="Exactly 4 options, no A/B/C/D prefixes.")
    correct_index: int = Field(description="0-based index of the correct option.")
    explanation: str = Field(description="Why the correct option is correct, from the source only.")
    supporting_quote: str = Field(
        description="Verbatim span (<=25 words) from the source that proves the answer."
    )
    mnemonic: str = Field(description="A short, memorable hook for retaining this fact.")
    exam_tip: str = Field(
        description="How a smart test-taker reaches this answer: elimination, option clues, etc."
    )
    difficulty: Literal["easy", "moderate", "hard"]


class MnemonicTip(BaseModel):
    """The only thing the LLM contributes to a deterministic question: a
    memory hook and an exam strategy note. The question, options, correct
    answer and explanation are already decided from a database row before
    this is ever called — there's nothing here for the model to get wrong
    that would make the *fact* incorrect."""

    mnemonic: str = Field(description="A short, memorable hook for retaining this fact.")
    exam_tip: str = Field(
        description="How a smart test-taker reaches this answer: elimination, option clues, etc."
    )


class Distractors(BaseModel):
    """Wrong-but-believable options mined from adjacent corpus material."""

    options: List[str] = Field(description="Exactly 3 plausible but incorrect options.")
    rationale: List[str] = Field(description="Why each is tempting yet wrong.")


class Verdict(BaseModel):
    """Runtime faithfulness check. Runs before the student ever sees the question."""

    supported: bool = Field(description="True only if every claim traces to the source.")
    unsupported_claims: List[str] = Field(default_factory=list)
    quote_found_in_source: bool = Field(description="Is supporting_quote actually present?")
    reason: str


class Chunk(BaseModel):
    """A retrieved passage plus everything needed to cite it."""

    chunk_id: str
    text: str
    book: str
    subject: str
    locator: str          # e.g. "Organon §17" or "SULPHUR"
    similarity: float
    sparse: bool = False  # single-line field (Dose, Relationship, ...) or just short — ok as a distractor source, not as the answer source


class GradedAnswer(BaseModel):
    """What we hand the UI after the student responds."""

    correct: bool
    chosen_index: Optional[int]
    mcq: MCQ
    citation: str
