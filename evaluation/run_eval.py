"""Offline evaluation. Run before every deploy:

    python -m evaluation.run_eval --repeats 3

What we measure, and why each one exists:

  hallucination_rate    fraction of served questions whose claims aren't in the
                        source. This is the headline number. It should be ~0
                        because the verifier gates on it — a non-zero value means
                        the verifier itself is leaking.

  quote_validity        fraction of supporting_quotes actually found verbatim.
                        Deterministic, no LLM opinion involved. A regression here
                        is the earliest signal that generation has drifted.

  refusal_accuracy      on out-of-corpus probes, did we refuse? Measures the
                        confidence gate. A bot that answers "CRISPR protocol"
                        from a homoeopathy corpus has a broken gate.

  false_refusal_rate    on in-corpus topics, did we wrongly refuse? The gate's
                        other failure mode. Tuning MIN_RETRIEVAL_SIMILARITY is a
                        trade between these two columns — that's the whole point
                        of measuring both.

  retrieval_top1_sim    distribution of top-1 similarity. Use it to calibrate the
                        threshold rather than guessing.

  retrieval_hit_rate_at_k  fraction of goldset queries (those with an
                        `expect_locator`) where the expected chunk actually
                        appears somewhere in the top-k results. Separates
                        retrieval quality from generation quality — a topic
                        can fail either one for completely different reasons,
                        and only this number tells you which.

  latency_p50_ms / p95_ms  from logs/metrics.jsonl, overall and split by task
                        and by generative-vs-deterministic path — the
                        deterministic path (rag/deterministic.py) only ever
                        logs task="coaching", so its latency profile (one
                        cheap call) is naturally comparable against the
                        generative path's three-call pipeline.

Cost: repeats * len(goldset) * ~3 calls. Keep the goldset small.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from config import METRICS_LOG, RETRIEVE_K
from rag import deterministic
from rag.generator import GenerationFailed, make_question
from rag.retriever import NotInCorpus, effective_top

GOLDSET = Path(__file__).parent / "goldset.jsonl"
REPERTORY_GOLDSET = Path(__file__).parent / "repertory_goldset.jsonl"


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion.

    Preferred over the normal approximation (`p +/- z*sqrt(p(1-p)/n)`) at the
    small n this goldset still has: the normal approximation can produce
    bounds outside [0, 1] and is a poor fit near p=0 or p=1, exactly where
    refusal_accuracy tends to sit. See CLAUDE_CODE_EVAL_EXPANSION.md for why
    this matters -- 0.833 and 1.0 were not distinguishable at n=6.
    """
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return round(max(0.0, (center - margin) / denom), 4), round(min(1.0, (center + margin) / denom), 4)


def _rate_report(successes: int, n: int) -> dict | None:
    """A rate plus the raw counts and a Wilson 95% CI -- so the printed
    number always carries its own uncertainty instead of reading as exact."""
    if n == 0:
        return None
    return {
        "rate": round(successes / n, 4),
        "n": n,
        "count": successes,
        "ci95": _wilson_ci(successes, n),
    }


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _latency_report() -> dict:
    if not METRICS_LOG.exists():
        return {}
    rows = [json.loads(ln) for ln in METRICS_LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("latency_ms") is not None]
    if not rows:
        return {}

    def stats(subset: list[dict]) -> dict:
        lat = [r["latency_ms"] for r in subset]
        return {"p50_ms": round(_percentile(lat, 0.5), 1), "p95_ms": round(_percentile(lat, 0.95), 1), "n": len(lat)}

    by_task: dict[str, list[dict]] = {}
    by_path: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
        # Entries logged before rag/llm.py grew the `path` field default to
        # "generative" -- that was the only path that existed at the time.
        by_path.setdefault(r.get("path", "generative"), []).append(r)

    return {
        "overall": stats(rows),
        "by_task": {t: stats(v) for t, v in by_task.items()},
        "by_path": {p: stats(v) for p, v in by_path.items()},
    }


def run(repeats: int) -> dict:
    # Explicit encoding matters here: this file's default (locale-dependent,
    # cp1252 on Windows) silently mangled "§" into "Â§" -- harmless while
    # only topic/subject strings were read (a slightly-off embedding query
    # goes unnoticed), but exact-match-sensitive once expect_locator needed
    # byte-for-byte fidelity against Chroma's UTF-8 metadata.
    cases = [json.loads(ln) for ln in GOLDSET.read_text(encoding="utf-8").splitlines() if ln.strip()]

    served = 0
    unsupported = 0
    quote_ok = 0
    correct_refusals = 0
    refuse_cases = 0
    false_refusals = 0
    answerable_cases = 0
    top1_sims: list[float] = []
    hit_rate_hits = 0
    hit_rate_total = 0
    failures: list[dict] = []

    for case in cases:
        topic, subject, expect = case["topic"], case["subject"], case["expect"]
        expect_locator = case.get("expect_locator")

        # `top` is the sparse-aware pick the real confidence gate would use
        # (not gated on the threshold here — we need the value even below it
        # to calibrate that threshold); `hits` is the raw ranking, used
        # as-is for hit-rate@k since that's meant to measure retrieval
        # quality independent of the sparse-source policy overlay.
        top, hits = effective_top(topic, subject=subject)
        if top is not None:
            top1_sims.append(top.similarity)
        if expect_locator:
            hit_rate_total += 1
            if any(expect_locator.lower() in h.locator.lower() for h in hits[:RETRIEVE_K]):
                hit_rate_hits += 1

        for _ in range(repeats if expect == "answerable" else 1):
            if expect == "refuse":
                refuse_cases += 1
            else:
                answerable_cases += 1

            try:
                mcq, source, verdict = make_question("UPSC (Homoeopathy)", subject, topic)
            except (NotInCorpus, GenerationFailed) as exc:
                if expect == "refuse":
                    correct_refusals += 1
                else:
                    false_refusals += 1
                    failures.append({"topic": topic, "kind": "false_refusal", "why": str(exc)[:200]})
                continue

            # We got a question. For a "refuse" case that is itself the failure.
            if expect == "refuse":
                failures.append({"topic": topic, "kind": "answered_out_of_corpus"})

            served += 1
            if not verdict.supported:
                unsupported += 1  # should be unreachable: verifier gates on this
            if verdict.quote_found_in_source:
                quote_ok += 1
            else:
                failures.append({"topic": topic, "kind": "quote_not_found"})

    return {
        "served": served,
        "hallucination_rate": round(unsupported / served, 4) if served else None,
        "quote_validity": round(quote_ok / served, 4) if served else None,
        # Both carry raw counts + a Wilson 95% CI, not just a point estimate --
        # see _rate_report / CLAUDE_CODE_EVAL_EXPANSION.md for why that matters
        # at goldset sizes where a couple of flipped cases swing the number a lot.
        "refusal_accuracy": _rate_report(correct_refusals, refuse_cases),
        "false_refusal_rate": _rate_report(false_refusals, answerable_cases),
        "retrieval_top1_sim_median": round(statistics.median(top1_sims), 4) if top1_sims else None,
        "retrieval_top1_sim_min": round(min(top1_sims), 4) if top1_sims else None,
        "retrieval_hit_rate_at_k": round(hit_rate_hits / hit_rate_total, 4) if hit_rate_total else None,
        "retrieval_hit_rate_at_k_n": hit_rate_total,
        "latency": _latency_report(),
        "failures": failures,
    }


def run_repertory() -> dict:
    """Verify rag/deterministic.py's Kent-backed path against
    evaluation/repertory_goldset.jsonl (generated by
    evaluation.build_repertory_goldset, never hand-written).

    Every case is targeted at an exact rubric_id / remedy from the goldset —
    not a random draw — so this is a correctness check on the deterministic
    generator itself, not a sampling exercise. answer_correctness should be
    1.0: the answer is a database row look-up, so a wrong answer here means
    the generator has an actual bug, not bad luck.
    """
    cases = [json.loads(ln) for ln in REPERTORY_GOLDSET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    forward = [c for c in cases if c["qtype"] == "repertory-grade3"]
    reverse = [c for c in cases if c["qtype"] == "repertory-reverse"]

    correct = 0
    evaluated = 0  # cases that actually produced an MCQ to check -- excludes errors below
    forward_evaluated = 0
    distractor_clean = 0
    errors = 0
    failures: list[dict] = []

    for case in forward:
        try:
            mcq, source, verdict = deterministic.make_question(kinds=["rubric_to_remedy"], rubric_id=case["rubric_id"])
        except deterministic.DeterministicUnavailable as exc:
            errors += 1
            failures.append({"id": case["id"], "kind": "unavailable", "why": str(exc)[:200]})
            continue
        except ValueError as exc:
            # The mnemonic/exam-tip call (the only LLM call on this path) can
            # still fail after rag/llm.py's own retries -- e.g. a repetition-
            # loop response that broke schema decoding twice in a row. That's
            # a model hiccup on a low-stakes field, not the fact being wrong,
            # so it's excluded from answer_correctness rather than counted
            # against it -- but still recorded, not silently dropped.
            errors += 1
            failures.append({"id": case["id"], "kind": "generation_error", "why": str(exc)[:200]})
            continue

        evaluated += 1
        forward_evaluated += 1
        grade3_all = set(case["grade3_all"])
        answer = mcq.options[mcq.correct_index]
        distractors = [o for i, o in enumerate(mcq.options) if i != mcq.correct_index]

        if answer in grade3_all:
            correct += 1
        else:
            failures.append({"id": case["id"], "kind": "wrong_answer", "answer": answer, "expected_any_of": sorted(grade3_all)})

        bad = [d for d in distractors if d in grade3_all]
        if not bad:
            distractor_clean += 1
        else:
            failures.append({"id": case["id"], "kind": "distractor_overlaps_grade3", "distractors": bad})

    for case in reverse:
        try:
            mcq, source, verdict = deterministic.make_question(kinds=["remedy_to_rubric"], remedy=case["remedy"])
        except deterministic.DeterministicUnavailable as exc:
            errors += 1
            failures.append({"id": case["id"], "kind": "unavailable", "why": str(exc)[:200]})
            continue
        except ValueError as exc:
            errors += 1
            failures.append({"id": case["id"], "kind": "generation_error", "why": str(exc)[:200]})
            continue

        evaluated += 1
        expect_rubrics = set(case["expect_rubrics"])
        answer = mcq.options[mcq.correct_index]
        if answer in expect_rubrics:
            correct += 1
        else:
            failures.append({"id": case["id"], "kind": "reverse_wrong_answer", "answer": answer})

    return {
        "n_forward": len(forward),
        "n_reverse": len(reverse),
        "n_errors": errors,  # mnemonic-call/model hiccups -- not fact errors, excluded from the metrics below
        "answer_correctness": round(correct / evaluated, 4) if evaluated else None,
        "distractor_validity": round(distractor_clean / forward_evaluated, 4) if forward_evaluated else None,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="generations per answerable topic")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--repertory", action="store_true",
        help="verify the deterministic Kent path against repertory_goldset.jsonl instead of the main eval",
    )
    args = ap.parse_args()

    if args.repertory:
        out = args.out or Path("evaluation/last_repertory_report.json")
        report = run_repertory()
        out.write_text(json.dumps(report, indent=2))
        summary = {k: v for k, v in report.items() if k != "failures"}
        print(json.dumps(summary, indent=2))
        if report["failures"]:
            print(f"\n{len(report['failures'])} failure(s) — see {out}")
        return

    out = args.out or Path("evaluation/last_report.json")
    report = run(args.repeats)
    out.write_text(json.dumps(report, indent=2))

    summary = {k: v for k, v in report.items() if k != "failures"}
    print(json.dumps(summary, indent=2))
    for key in ("refusal_accuracy", "false_refusal_rate"):
        r = report[key]
        if r:
            lo, hi = r["ci95"]
            print(f"{key}: {r['rate']} ({r['count']}/{r['n']}, 95% CI {lo}-{hi})")
    if report["failures"]:
        shown = report["failures"][:20]
        print(f"\n{len(report['failures'])} failure(s) — see {out}" +
              (f" (showing first {len(shown)} above the fold)" if len(report["failures"]) > 20 else ""))


if __name__ == "__main__":
    main()
