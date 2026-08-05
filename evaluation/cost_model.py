"""Quantify the routing decision instead of asserting it.

    python -m evaluation.cost_model --questions 50000

Reads real token counts from logs/metrics.jsonl when available, so the estimate
is grounded in what the app actually sends, not in a guess. Prices are USD per
1M tokens, June 2026, standard (non-batch) tier. Verify before quoting.
"""

from __future__ import annotations

import argparse
import json

from config import DETERMINISTIC_ROUTE_RATE, METRICS_LOG

PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.5": (5.00, 30.00),
}

# Fallbacks if no metrics have been logged yet.
DEFAULT_IN, DEFAULT_OUT = 3000, 700


def observed_tokens() -> tuple[int, int]:
    if not METRICS_LOG.exists():
        return DEFAULT_IN, DEFAULT_OUT
    rows = [json.loads(l) for l in METRICS_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("input_tokens")]
    if not rows:
        return DEFAULT_IN, DEFAULT_OUT

    # A served question is either the generative pipeline (mcq_generation +
    # distractor_generation + grounding_verification -- 3 calls) or
    # rag/deterministic.py (task="coaching" only -- 1 call), mixed roughly
    # DETERMINISTIC_ROUTE_RATE : (1 - DETERMINISTIC_ROUTE_RATE). Blending the
    # two path's own token profiles (rather than one calls_per_question=3
    # constant applied to the average of everything) matters because a
    # deterministic call's prompt is much smaller than a generative one's.
    det_rows = [r for r in rows if r.get("path") == "deterministic"]
    gen_rows = [r for r in rows if r.get("path", "generative") == "generative"]

    def per_question(subset: list[dict], calls_per_question: int) -> tuple[float, float]:
        if not subset:
            return 0.0, 0.0
        avg_in = sum(r["input_tokens"] for r in subset) / len(subset)
        avg_out = sum((r.get("output_tokens") or 0) for r in subset) / len(subset)
        return avg_in * calls_per_question, avg_out * calls_per_question

    det_in, det_out = per_question(det_rows, 1)
    gen_in, gen_out = per_question(gen_rows or rows, 3)

    rate = DETERMINISTIC_ROUTE_RATE
    avg_in = rate * det_in + (1 - rate) * gen_in
    avg_out = rate * det_out + (1 - rate) * gen_out
    return int(avg_in), int(avg_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=50_000, help="MCQs served per month")
    args = ap.parse_args()

    tin, tout = observed_tokens()
    src = "logs/metrics.jsonl" if METRICS_LOG.exists() else "defaults"
    print(f"Tokens per question ({src}): {tin} in / {tout} out")
    print(f"Volume: {args.questions:,} questions/month\n")

    in_m = args.questions * tin / 1e6
    out_m = args.questions * tout / 1e6

    rows = []
    for model, (pin, pout) in PRICES.items():
        rows.append((model, in_m * pin + out_m * pout))
    rows.sort(key=lambda r: r[1])

    baseline = dict(rows)["gpt-5.5"]
    width = max(len(m) for m, _ in rows)
    for model, cost in rows:
        saving = f"  ({100 * (1 - cost / baseline):.0f}% vs gpt-5.5)" if cost < baseline else ""
        print(f"  {model:<{width}}  ${cost:>9,.2f}/mo{saving}")


if __name__ == "__main__":
    main()
