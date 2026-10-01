"""Contract tests for the arm gate -- the predicate that decides shipping.

Two real bugs live behind these assertions, both found in production code:

1. The ARM GUARD opened on the COUNT of judged commits (`n_commits >= 25`)
   regardless of what they said. `commit_judge.py` exists to prove arms
   harmful, so recording 27 judgements mechanically unlocked the feature its
   own audit disproved, and a rebuild silently shipped arms ON.

2. When the win-unit evidence was one-sided -- the first two `score_delta`
   rows were both WRONG commits, each turning a win into a loss -- `e_gain_raw`
   fell to 0 and the `max(1.0, ...)` clamps turned "no measured upside" into
   "upside 1", driving the threshold to 1/(1+1) = 0.5, the LOOSEST available,
   on evidence that says committing is catastrophic.

Both share one shape: a gate that fails PERMISSIVE when evidence is absent or
negative. Every case below asks the same question -- what happens when the
evidence arrives and is bad?

    python tests/test_gates.py
"""
from kaggriculture.paths import ROOT
import os
import sys

import kaggriculture.train.train_gates as TG  # noqa: E402


def test_blocks_on_insufficient_evidence():
    ok, why = TG.arms_allowed({"n_commits": 3, "arms_positive_value": True})
    assert not ok, why
    assert "need >=" in why, why
    print(f"few commits -> blocked ({why})")


def test_blocks_on_negative_value_even_with_many_commits():
    """The bug that shipped arms: plenty of evidence, all of it bad."""
    ok, why = TG.arms_allowed({
        "n_commits": 1000, "arms_positive_value": False,
        "e_gain_correct_raw": -52939.6, "e_cost_false_raw": 39405.4})
    assert not ok, why
    assert "NEGATIVE-VALUE" in why, why
    print(f"1000 commits but negative value -> blocked ({why[:60]}...)")


def test_allows_only_when_count_and_sign_are_both_good():
    ok, why = TG.arms_allowed({
        "n_commits": 30, "arms_positive_value": True,
        "e_gain_correct_raw": 500.0, "e_cost_false_raw": 100.0})
    assert ok, why
    print(f"enough commits AND favourable -> allowed ({why})")


def test_legacy_gates_without_the_verdict_field():
    """Older gates.json has only the CLAMPED values; must still fail safe."""
    ok, why = TG.arms_allowed({"n_commits": 30, "e_gain_correct": 1.0,
                               "e_cost_false": 41314.6})
    assert not ok, why
    ok2, _ = TG.arms_allowed({"n_commits": 30, "e_gain_correct": 900.0,
                              "e_cost_false": 100.0})
    assert ok2
    print("legacy clamped-only gates: blocked when gain <= cost, allowed when not")


def test_unreadable_gates_blocks():
    ok, why = TG.arms_allowed(path=os.path.join(ROOT, "does-not-exist.json"))
    assert not ok and "unreadable" in why, why
    print(f"missing gates.json -> blocked ({why})")


def test_live_gates_file_is_consistent():
    """Whatever the real file says, the verdict must match its own numbers."""
    import json
    p = os.path.join(ROOT, "models", "v22", "identifier", "gates.json")
    if not os.path.exists(p):
        print("SKIP: no live gates.json")
        return
    g = json.load(open(p, encoding="utf-8"))
    ok, why = TG.arms_allowed(g)
    raw_gain = g.get("e_gain_correct_raw")
    if raw_gain is not None and raw_gain <= 0:
        assert not ok, "gate allows arms while measured upside is <= 0"
        # and the threshold must be at its most conservative, not its loosest
        assert g["match_prob"] >= 0.94, (
            f"no measured upside but match_prob is {g['match_prob']} -- a "
            f"missing-upside estimate must fail conservative")
    print(f"live gates.json: match_prob {g.get('match_prob')}, "
          f"raw gain {raw_gain}, allowed={ok}")


def _policy_file(tmpname, rows):
    import json
    import tempfile
    p = os.path.join(tempfile.gettempdir(), tmpname)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_policy_guard_blocks_with_no_evidence():
    ok, why = TG.policy_allowed(path=os.path.join(ROOT, "does-not-exist.jsonl"))
    assert not ok and "no policy judgments" in why, why
    print(f"policy guard, no file -> blocked ({why})")


def test_policy_guard_blocks_on_volume_alone():
    """The ARM GUARD bug must not be recreated: 1000 judged games, all bad."""
    p = _policy_file("pj_neg.jsonl", [{"score_delta": -0.5}] * 1000)
    ok, why = TG.policy_allowed(path=p)
    assert not ok, why
    assert "NEGATIVE-VALUE" in why, why
    print(f"policy guard, 1000 negative judgments -> blocked ({why[:60]}...)")


def test_policy_guard_blocks_below_floor_even_when_positive():
    p = _policy_file("pj_few.jsonl", [{"score_delta": 1.0}] * 5)
    ok, why = TG.policy_allowed(path=p)
    assert not ok and "need >=" in why, why
    print(f"policy guard, 5 positive judgments -> blocked ({why})")


def test_policy_guard_opens_only_on_count_and_sign():
    p = _policy_file("pj_good.jsonl", [{"score_delta": 0.25}] * 30)
    ok, why = TG.policy_allowed(path=p)
    assert ok, why
    print(f"policy guard, 30 positive judgments -> allowed ({why})")


def test_generation_scoping_lets_rehabilitated_evidence_win():
    """2026-08-14 decision: the daily rehab job re-judges retrained
    arms/policy under a fresh generation tag. The gate prices the NEWEST
    generation when it has enough rows -- otherwise 52 dead v25.1 rows would
    veto every future policy forever -- but a thin new generation must not
    escape the pooled (conservative) view."""
    p = _policy_file("pj_gen_suite.jsonl",
                     [{"score_delta": -0.3, "judged_pair": "old-gen"}] * 52
                     + [{"score_delta": 0.2, "judged_pair": "new-gen"}] * 30)
    ok, why = TG.policy_allowed(path=p)
    assert ok and "new-gen" in why, why
    p2 = _policy_file("pj_gen_thin.jsonl",
                      [{"score_delta": -0.3, "judged_pair": "old-gen"}] * 52
                      + [{"score_delta": 1.0, "judged_pair": "new-gen"}] * 5)
    ok2, why2 = TG.policy_allowed(path=p2)
    assert not ok2, why2
    print("generation scoping: strong new generation opens, thin one stays "
          "pooled-blocked")


def test_preranker_blocks_without_measurement():
    ok, why = TG.preranker_allowed(path=os.path.join(ROOT, "nope.json"))
    assert not ok and "no recall measurement" in why, why
    print(f"pre-ranker, no evidence -> gated ({why})")


def test_preranker_blocks_on_bad_recall():
    """Speed is not the criterion: a fast pre-ranker that ranks badly makes
    the funnel wider AND worse."""
    import json
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "recall_bad.json")
    json.dump({"recall_at_n": 0.40, "n": 4, "candidates": 10},
              open(p, "w", encoding="utf-8"))
    ok, why = TG.preranker_allowed(path=p)
    assert not ok and "losing the official winners" in why, why
    json.dump({"recall_at_n": 1.0, "n": 2, "candidates": 3},
              open(p, "w", encoding="utf-8"))
    ok2, why2 = TG.preranker_allowed(path=p)
    assert not ok2, why2                    # perfect recall, tiny sample
    json.dump({"recall_at_n": 0.85, "n": 4, "candidates": 8},
              open(p, "w", encoding="utf-8"))
    ok3, why3 = TG.preranker_allowed(path=p)
    assert ok3, why3
    print("pre-ranker: bad recall gated, small sample gated, good opens")


def test_policy_guard_zero_mean_blocks():
    """Exactly zero measured value is not evidence FOR shipping."""
    p = _policy_file("pj_zero.jsonl",
                     [{"score_delta": 1.0}] * 15 + [{"score_delta": -1.0}] * 15)
    ok, why = TG.policy_allowed(path=p)
    assert not ok, why
    print(f"policy guard, zero mean -> blocked ({why[:60]})")


if __name__ == "__main__":
    for fn in (test_blocks_on_insufficient_evidence,
               test_blocks_on_negative_value_even_with_many_commits,
               test_allows_only_when_count_and_sign_are_both_good,
               test_legacy_gates_without_the_verdict_field,
               test_unreadable_gates_blocks,
               test_live_gates_file_is_consistent,
               test_policy_guard_blocks_with_no_evidence,
               test_policy_guard_blocks_on_volume_alone,
               test_policy_guard_blocks_below_floor_even_when_positive,
               test_policy_guard_opens_only_on_count_and_sign,
               test_policy_guard_zero_mean_blocks,
               test_generation_scoping_lets_rehabilitated_evidence_win,
               test_preranker_blocks_without_measurement,
               test_preranker_blocks_on_bad_recall):
        fn()
    print("\nall gate checks passed")
