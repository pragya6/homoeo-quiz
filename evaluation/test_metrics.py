"""Pure-function tests for evaluation/run_eval.py's statistics helpers.

Scope, deliberately narrow: `_wilson_ci`, `_rate_report`, and `_percentile`
are the only functions in run_eval.py that are already pure and isolated --
no GEMINI_API_KEY, no network, no disk reads. Everything else that produces
a headline metric (hallucination_rate, refusal_accuracy, false_refusal_rate,
retrieval_hit_rate_at_k) is computed inline inside run()/run_repertory(),
fused to live make_question()/effective_top() calls, and is NOT covered
here -- testing those would require extracting them into pure functions
first, which is a refactor of the code under test and out of scope for
this suite. See CLAUDE_CODE_EVAL_EXPANSION.md for why Wilson intervals
(over the normal approximation) matter at this goldset's size.

Run: python -m pytest evaluation/test_metrics.py -q
Requires: nothing. No GEMINI_API_KEY, no network, no goldset.jsonl reads.
"""

from __future__ import annotations

from evaluation.run_eval import _percentile, _rate_report, _wilson_ci

# ---------------------------------------------------------------- _wilson_ci


def test_wilson_ci_none_when_n_is_zero():
    assert _wilson_ci(0, 0) is None


def test_wilson_ci_known_interval_3_of_4():
    assert _wilson_ci(3, 4) == (0.3006, 0.9544)


def test_wilson_ci_refusal_accuracy_case_5_of_6():
    # The exact case CLAUDE_CODE_EVAL_EXPANSION.md cites: refusal_accuracy
    # 0.833 (5/6) has a wide enough Wilson interval that it should not read
    # as confidently different from a perfect 6/6.
    assert _wilson_ci(5, 6) == (0.4365, 0.9699)


def test_wilson_ci_5_of_6_and_6_of_6_intervals_overlap():
    # This is the actual claim from the doc: 0.833 and 1.0 are "not
    # distinguishable" at n=6 -- i.e. their 95% CIs overlap.
    lo_a, hi_a = _wilson_ci(5, 6)
    lo_b, hi_b = _wilson_ci(6, 6)
    assert lo_b <= hi_a  # [lo_b, hi_b] and [lo_a, hi_a] intersect


def test_wilson_ci_perfect_record_upper_bound_is_exactly_one():
    # p_hat = 1.0 is exactly where the normal approximation can exceed 1;
    # Wilson's upper bound must clamp to 1.0 rather than overshoot.
    lo, hi = _wilson_ci(4, 4)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_ci_zero_successes_lower_bound_is_exactly_zero():
    lo, hi = _wilson_ci(0, 4)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


# --------------------------------------------------------------- _rate_report


def test_rate_report_none_when_n_is_zero():
    assert _rate_report(5, 0) is None


def test_rate_report_shape_and_values():
    report = _rate_report(3, 4)
    assert report == {
        "rate": 0.75,
        "n": 4,
        "count": 3,
        "ci95": (0.3006, 0.9544),
    }


def test_rate_report_refusal_accuracy_case_5_of_6():
    report = _rate_report(5, 6)
    assert report == {
        "rate": 0.8333,
        "n": 6,
        "count": 5,
        "ci95": (0.4365, 0.9699),
    }


def test_rate_report_full_success_ci_still_below_certainty_at_low_n():
    report = _rate_report(6, 6)
    assert report["rate"] == 1.0
    assert report["ci95"][0] < 1.0  # lower bound still short of 1.0 at n=6


# --------------------------------------------------------------- _percentile


def test_percentile_median_odd_length():
    assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0


def test_percentile_median_even_length_interpolates():
    assert _percentile([1, 2, 3, 4], 0.5) == 2.5


def test_percentile_single_value_short_circuits():
    assert _percentile([42.0], 0.5) == 42.0


def test_percentile_p0_is_min():
    assert _percentile([5, 1, 3, 2, 4], 0.0) == 1.0


def test_percentile_p1_is_max():
    assert _percentile([5, 1, 3, 2, 4], 1.0) == 5


def test_percentile_unsorted_input_is_sorted_first():
    assert _percentile([4, 2, 5, 1, 3], 0.5) == 3.0
