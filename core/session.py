"""Per-student quiz state.

Two things live here that are easy to get wrong:

1. The answer is generated *with* the question but never rendered until the
   student responds. Generation and presentation are separate concerns. One LLM
   call per question, not two.

2. Weak-topic tracking. Every miss increments a topic's weight; the next topic
   is drawn from that weighted pool. This is a crude Leitner system — enough to
   make the bot feel like it's paying attention, cheap enough to keep in memory.

State is in-process. That is correct for 2-10 students and wrong for 200: see
README "Scaling" for the swap to Redis / a keyed store.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto

from config import SUBJECTS
from rag.schemas import MCQ

# Seed topics per subject, used when the student picks "general practice".
GENERAL_TOPICS = {
    "Organon of Medicine": [
        "law of similars", "vital force", "chronic miasms", "drug proving",
        "homoeopathic aggravation", "palliation and suppression", "susceptibility",
        "the ideal cure", "one-sided diseases", "posology",
    ],
    "Materia Medica": [
        "Sulphur", "Pulsatilla", "Lycopodium", "Nux vomica", "Belladonna",
        "Arsenicum album", "Natrum muriaticum", "Bryonia", "Phosphorus", "Sepia",
    ],
    "Repertory": [
        "rubric construction", "gradation of symptoms", "generals versus particulars",
        "repertorial totality", "cross-referencing",
    ],
    "Homoeopathic Philosophy": [
        "simple substance", "direction of cure", "prognosis after the remedy",
        "the sick individual", "signs of improvement",
    ],
}

LETTERS = ["A", "B", "C", "D"]


class State(Enum):
    CONFIG_EXAM = auto()
    CONFIG_MODE = auto()
    ASKING = auto()
    AWAITING_ANSWER = auto()


@dataclass
class Session:
    state: State = State.CONFIG_EXAM
    exam: str | None = None
    subject: str | None = None
    topic: str | None = None
    general_mode: bool = False

    current: MCQ | None = None
    current_citation: str | None = None

    asked: list[str] = field(default_factory=list)
    answered: int = 0
    correct: int = 0
    # topic -> miss count. Higher weight = more likely to resurface.
    weak_topics: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ config
    def set_exam(self, text: str) -> bool:
        t = text.strip().lower()
        if "upsc" in t or t == "1":
            self.exam = "UPSC (Homoeopathy)"
        elif "sr" in t or "resident" in t or t == "2":
            self.exam = "SR-ship Entrance"
        elif "other" in t or "general" in t or t == "3":
            self.exam = "Other / General"
        else:
            return False
        self.state = State.CONFIG_MODE
        return True

    def set_mode(self, text: str) -> bool:
        t = text.strip()
        low = t.lower()
        if low in {"general", "2", "practice", "mixed"}:
            self.general_mode = True
            self.subject = None
            self.topic = None
        else:
            # Try to read a subject name; otherwise treat the whole string as a topic.
            matched = next((s for s in SUBJECTS if s.lower() in low), None)
            self.subject = matched
            self.topic = t if not matched else None
            self.general_mode = False
        self.state = State.ASKING
        return True

    # ------------------------------------------------------------------ topics
    def next_topic(self) -> tuple[str, str | None]:
        """Return (topic, subject_filter).

        30% of the time we revisit a topic the student has missed. Not more —
        drilling only weaknesses is demoralising and narrows coverage.
        """
        if self.weak_topics and random.random() < 0.30:
            pool = [t for t, n in self.weak_topics.items() for _ in range(n)]
            return random.choice(pool), self.subject

        if not self.general_mode and self.topic:
            return self.topic, self.subject

        subject = self.subject or random.choice(SUBJECTS)
        return random.choice(GENERAL_TOPICS[subject]), subject

    # ------------------------------------------------------------------ answers
    def serve(self, mcq: MCQ, citation: str) -> None:
        self.current = mcq
        self.current_citation = citation
        self.asked.append(mcq.question)
        self.state = State.AWAITING_ANSWER

    @staticmethod
    def parse_choice(text: str, options: list[str]) -> int | None:
        """Accept 'B', 'b)', '2', or the option text itself."""
        t = text.strip().lower().rstrip(").:")
        if len(t) == 1 and t.upper() in LETTERS:
            return LETTERS.index(t.upper())
        if re.fullmatch(r"[1-4]", t):
            return int(t) - 1
        for i, opt in enumerate(options):
            if t and t in opt.strip().lower():
                return i
        return None

    def grade(self, choice: int | None, topic: str) -> bool:
        self.answered += 1
        is_correct = choice is not None and choice == self.current.correct_index
        if is_correct:
            self.correct += 1
        else:
            self.weak_topics[topic] = self.weak_topics.get(topic, 0) + 1
        self.state = State.ASKING
        return is_correct

    def score_line(self) -> str:
        if not self.answered:
            return ""
        pct = 100 * self.correct / self.answered
        weak = sorted(self.weak_topics.items(), key=lambda kv: -kv[1])[:3]
        weak_s = ", ".join(t for t, _ in weak)
        tail = f" · Revisiting: {weak_s}" if weak else ""
        return f"Score: {self.correct}/{self.answered} ({pct:.0f}%){tail}"
