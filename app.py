"""HomeoQuiz — grounded MCQ practice for homoeopathy entrance exams.

Entry point for Hugging Face Spaces. `python app.py` locally.

The student can do exactly two things, by design:
  - answer the current question
  - state which exam / topic they want

Everything else (revealing answers, picking the next topic, refusing when the
corpus is thin) is the app's decision, not a prompt the student can steer.
That constraint is also our injection surface: a student's message never reaches
the model as instructions, only as an answer to parse.
"""

from __future__ import annotations

import sys

from config import CHROMA_DIR, REPERTORY_DB

def ensure_index() -> None:
    """Build the Chroma index and repertory DB on first boot if either is
    missing, then return. Idempotent — a container that already has both
    (a warm restart, not a fresh deploy) does nothing.

    Both are normally committed via Git LFS (see .gitattributes) so a fresh
    deploy already has them and this is a no-op. This function only builds
    from the small committed source files (data/remedy_aliases.json,
    out/*.json) as a fallback -- e.g. a fresh clone without LFS data, or CI
    before the index has been committed.

    Must run before rag.retriever / rag.repertory open their DB handles —
    both are lazy singletons that raise if the file isn't there yet.
    """
    repertory_ready = REPERTORY_DB.exists()
    chroma_ready = (CHROMA_DIR / "chroma.sqlite3").exists()

    if repertory_ready and chroma_ready:
        print("[ensure_index] index ready (found existing data/chroma and data/repertory.db)")
        return

    if not repertory_ready:
        print("[ensure_index] repertory.db missing, building (reads out/kent_repertory.json, ~10s, no API calls)...")
        from ingest.build_repertory import main as build_repertory

        if build_repertory() != 0:
            raise RuntimeError("ingest.build_repertory failed — see output above")
        print("[ensure_index] repertory.db ready")

    if not chroma_ready:
        print("[ensure_index] chroma index missing, building the full corpus "
              "locally with sentence-transformers, no API calls, ~30min on CPU...")
        from ingest.build_index import main as build_chroma_index

        # build_index.main() reads --books from sys.argv (it's a CLI entry
        # point); reusing it here instead of duplicating its logic means
        # driving it the same way its own CLI does. No --books passed, so it
        # falls through to its own default (ALL_BOOKS).
        old_argv = sys.argv
        sys.argv = ["ingest.build_index"]
        try:
            rc = build_chroma_index()
        finally:
            sys.argv = old_argv
        if rc != 0:
            raise RuntimeError("ingest.build_index failed — see output above")
        print("[ensure_index] chroma index ready")


ensure_index()

import gradio as gr

from config import EXAMS, SUBJECTS
from core.session import LETTERS, Session, State
from rag.generator import GenerationFailed, NotInCorpus, make_question

GREETING = (
    "**Namaste — welcome to HomeoQuiz.**\n\n"
    "Every question here is generated from the classical public-domain canon "
    "(Hahnemann, Boericke, Kent, Allen) and checked against its source before "
    "you see it. If the books don't cover something, I'll say so rather than guess.\n\n"
    "**Which exam are you preparing for?**\n"
    + "\n".join(f"{i+1}. {e}" for i, e in enumerate(EXAMS))
)

MODE_PROMPT = (
    "Good. Now — **a specific topic, or general practice?**\n\n"
    "- Type a topic or subject (e.g. `Sulphur`, `chronic miasms`, `Repertory`)\n"
    "- Or type `general` for mixed practice across "
    + ", ".join(SUBJECTS)
)


def _render_question(sess: Session, topic: str) -> str:
    mcq = sess.current
    lines = [f"**Q{sess.answered + 1}.** {mcq.question}", ""]
    lines += [f"**{LETTERS[i]}.** {opt}" for i, opt in enumerate(mcq.options)]
    lines += ["", f"_{topic} · {mcq.difficulty}_", "", "Your answer? (A/B/C/D)"]
    return "\n".join(lines)


def _render_reveal(sess: Session, correct: bool, choice: int | None) -> str:
    mcq = sess.current
    right = mcq.options[mcq.correct_index]

    if correct:
        head = f"✅ **Correct.** {LETTERS[mcq.correct_index]} — {right}"
    elif choice is None:
        head = f"🤔 Couldn't read that as an option. The answer is **{LETTERS[mcq.correct_index]}. {right}**"
    else:
        head = (
            f"❌ **Not quite.** You chose {LETTERS[choice]}. "
            f"The answer is **{LETTERS[mcq.correct_index]}. {right}**"
        )

    return "\n\n".join(
        [
            head,
            f"**Why:** {mcq.explanation}",
            f"> {mcq.supporting_quote}\n> — *{sess.current_citation}*",
            f"🧠 **Mnemonic:** {mcq.mnemonic}",
            f"🎯 **Exam tip:** {mcq.exam_tip}",
            f"_{sess.score_line()}_",
        ]
    )


MAX_TOPIC_RETRIES = 5


def _next_question(sess: Session) -> str:
    """Pick a topic and generate a question for it.

    A topic the student typed themselves is a request we owe them an answer
    about — if it fails, say so. A topic we auto-picked (general practice, or
    the weak-topic reroll) is our own implementation detail; if the corpus is
    thin on it, that's not the student's problem, so swap in another pick
    silently rather than dumping them back to config mode.
    """
    tried: set[str] = set()
    topic = subject = last_topic = None
    last_exc: NotInCorpus | GenerationFailed | None = None

    for _ in range(MAX_TOPIC_RETRIES):
        topic, subject = sess.next_topic()
        if topic in tried:
            continue
        tried.add(topic)
        is_explicit_ask = topic == sess.topic and not sess.general_mode

        try:
            mcq, source, _verdict = make_question(
                exam=sess.exam, subject=subject, topic=topic, asked=sess.asked
            )
        except (NotInCorpus, GenerationFailed) as exc:
            last_topic, last_exc = topic, exc
            if is_explicit_ask:
                break
            continue

        sess.serve(mcq, f"{source.book} — {source.locator}")
        sess._last_topic = topic  # noqa: SLF001 - simple carry for grading
        return _render_question(sess, topic)

    sess.state = State.CONFIG_MODE
    if isinstance(last_exc, NotInCorpus):
        return (
            f"I don't have solid source material for **{last_topic}** in the indexed "
            f"canon, so I won't invent a question about it.\n\n_({last_exc})_\n\n"
            "Try another topic, or type `general`."
        )
    return (
        f"I drafted a question on **{last_topic}** but it failed my own grounding "
        f"check, so I'm discarding it rather than showing you something I "
        f"can't source.\n\n_({last_exc})_\n\nPick another topic, or type `general`."
    )


def respond(message: str, history: list, sess: Session):
    msg = (message or "").strip()

    if sess.state is State.CONFIG_EXAM:
        if not sess.set_exam(msg):
            return "Pick 1, 2 or 3 — or name the exam.", sess
        return MODE_PROMPT, sess

    if sess.state is State.CONFIG_MODE:
        sess.set_mode(msg)
        return _next_question(sess), sess

    if sess.state is State.AWAITING_ANSWER:
        topic = getattr(sess, "_last_topic", sess.topic or "general")
        choice = Session.parse_choice(msg, sess.current.options)
        correct = sess.grade(choice, topic)
        reveal = _render_reveal(sess, correct, choice)
        return reveal + "\n\n---\n\n" + _next_question(sess), sess

    return _next_question(sess), sess


def build() -> gr.Blocks:
    # Gradio 6 moved `theme` from the Blocks constructor to .launch(), dropped
    # Chatbot's `show_copy_button` (copy-to-clipboard is default now), and
    # removed the old tuples chat format entirely -- messages
    # ({"role", "content"} dicts) is the only format left. requirements.txt
    # pins gradio>=5.9.0 with no upper bound, so this needs to work against
    # whatever 5.x/6.x actually resolves.
    with gr.Blocks(title="HomeoQuiz") as demo:
        gr.Markdown("# 🌿 HomeoQuiz\n*Grounded MCQ practice · UPSC Homoeopathy*")

        # gr.State deep-copies this value for each browser session, so students
        # never share a quiz. Swap for a keyed store when you outgrow one process.
        sess = gr.State(Session())
        chat = gr.Chatbot(value=[{"role": "assistant", "content": GREETING}], height=560)
        box = gr.Textbox(placeholder="Type your answer or a topic…", show_label=False, autofocus=True)

        def turn(message, history, session):
            reply, session = respond(message, history, session)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            return history, session, ""

        box.submit(turn, [box, chat, sess], [chat, sess, box])

        gr.Markdown(
            "_Sources: public-domain classical texts. Questions are verified "
            "against their source passage before display._"
        )
    return demo


if __name__ == "__main__":
    build().launch(theme=gr.themes.Soft())
