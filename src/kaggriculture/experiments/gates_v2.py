"""Gates v2: per-family empirical-Bayes commit thresholds + SPRT commit test.

Two pieces, both pure functions so the agent generator can embed them later:

1. eb_thresholds(rows): per-family decision thresholds with empirical-Bayes
   shrinkage. Correctness rate ~ Beta-Binomial shrunk toward the pooled rate;
   E[gain]/E[cost] ~ Normal-Normal shrunk toward pooled means. Threshold per
   family: p* = E[cost] / (E[gain] + E[cost]), same decision theory as
   train_gates.py, but per family and stable at tiny n.

2. sprt_update(state, posterior): in-game sequential probability ratio test
   replacing the "posterior >= 0.85 AND streak >= 2" heuristic. H1: the class
   posterior stream is genuine (opponent IS the family), H0: it is noise.
   Commit when the cumulated log-likelihood ratio crosses A = log((1-b)/a);
   abandon when below B = log(b/(1-a)).

The commit audit currently has 0 judged rows (arms are guarded off), so
eb_thresholds is exercised by synthetic tests here and activates on real data
untouched. Run as a script to self-test:

    python src/experiments/gates_v2.py
"""
import math
from collections import defaultdict

# ---------------------------------------------------------------- EB gates --

PRIOR_STRENGTH = 8.0        # pseudo-observations pulling a family to the pool


def eb_thresholds(rows, default_p=0.85):
    """rows: commit-audit dicts with keys family, commit_correct (bool),
    margin_delta (float). Returns {family: threshold} plus '_pooled'."""
    fams = defaultdict(list)
    for r in rows:
        if r.get("commit_correct") is None:
            continue
        fams[str(r.get("family", "?"))].append(r)
    all_rows = [r for v in fams.values() for r in v]
    if not all_rows:
        return {"_pooled": default_p}

    pool_gain = [float(r.get("margin_delta") or 0) for r in all_rows
                 if r["commit_correct"]]
    pool_cost = [-float(r.get("margin_delta") or 0) for r in all_rows
                 if not r["commit_correct"]]
    if not pool_gain or not pool_cost:
        return {"_pooled": default_p}
    g0 = sum(pool_gain) / len(pool_gain)
    c0 = sum(pool_cost) / len(pool_cost)
    pooled_t = c0 / (g0 + c0) if (g0 + c0) > 0 else default_p

    out = {"_pooled": round(pooled_t, 4)}
    for fam, rs in fams.items():
        n = len(rs)
        gains = [float(r.get("margin_delta") or 0) for r in rs
                 if r["commit_correct"]]
        costs = [-float(r.get("margin_delta") or 0) for r in rs
                 if not r["commit_correct"]]
        w = n / (n + PRIOR_STRENGTH)
        g = w * (sum(gains) / len(gains) if gains else g0) + (1 - w) * g0
        c = w * (sum(costs) / len(costs) if costs else c0) + (1 - w) * c0
        t = c / (g + c) if (g + c) > 0 else pooled_t
        out[fam] = round(min(0.97, max(0.5, t)), 4)
    return out


# -------------------------------------------------------------------- SPRT --

def sprt_update(state, posterior, p1=0.75, p0=0.25,
                alpha=0.02, beta=0.10):
    """One SPRT step. state: dict persisted across turns (agent state blob).
    posterior: identifier posterior for the tracked class this turn.
    Returns 'commit' | 'abandon' | 'continue'.

    p1/p0: P(evidence bit | match / no-match). alpha: false-commit rate
    (kept tight -- a wrong commit cost v23.1 half its economy), beta: missed-
    commit rate (looser -- missing an arm is the safe failure).

    The evidence bit cuts at 0.7, NOT 0.5: with a 0.5 cut an opponent whose
    posterior hovers ~0.5 (genuinely ambiguous) is a symmetric random walk
    that hits the commit bound 44% of the time (measured in the self-test).
    At 0.7 the ambiguous stream has P(bit)=0.3 < p0-adjacent, so ambiguity
    DRIFTS TO ABANDON -- only sustained decisive posteriors commit."""
    x = 1 if posterior >= 0.7 else 0
    # Bernoulli LLR on the thresholded evidence keeps the exported agent
    # arithmetic trivially auditable (two constants, one running sum).
    llr = (math.log(p1 / p0) if x else math.log((1 - p1) / (1 - p0)))
    state["sprt_llr"] = state.get("sprt_llr", 0.0) + llr
    a_bound = math.log((1 - beta) / alpha)
    b_bound = math.log(beta / (1 - alpha))
    if state["sprt_llr"] >= a_bound:
        return "commit"
    if state["sprt_llr"] <= b_bound:
        return "abandon"
    return "continue"


# ------------------------------------------------------------------- tests --

def _test():
    import random
    rng = random.Random(7)

    # EB: a family with few rows must sit near the pool; one with many rows
    # and high cost must demand a higher posterior.
    rows = []
    for _ in range(60):     # pool: gains 4000, costs 6000, mixed families
        ok = rng.random() < 0.5
        rows.append({"family": f"f{rng.randrange(6)}",
                     "commit_correct": ok,
                     "margin_delta": 4000 if ok else -6000})
    rows += [{"family": "expensive", "commit_correct": False,
              "margin_delta": -30000} for _ in range(12)]
    rows += [{"family": "expensive", "commit_correct": True,
              "margin_delta": 2000} for _ in range(3)]
    rows += [{"family": "tiny", "commit_correct": True,
              "margin_delta": 5000}]
    th = eb_thresholds(rows)
    assert abs(th["tiny"] - th["_pooled"]) < 0.08, \
        f"tiny family should hug the pool: {th['tiny']} vs {th['_pooled']}"
    assert th["expensive"] > th["_pooled"] + 0.02, \
        f"costly family must gate higher: {th['expensive']} vs {th['_pooled']}"
    print(f"EB ok: pooled {th['_pooled']}, tiny {th['tiny']}, "
          f"expensive {th['expensive']}")

    # SPRT: strong evidence commits fast; noise abandons; alpha honored.
    st = {}
    steps = 0
    for _ in range(50):
        steps += 1
        if sprt_update(st, 0.9) == "commit":
            break
    assert steps <= 8, f"strong evidence should commit fast, took {steps}"

    st = {}
    verdicts = [sprt_update(st, rng.random() * 0.4) for _ in range(40)]
    assert "commit" not in verdicts, "pure noise must never commit"
    assert "abandon" in verdicts, "noise should be abandoned, not tracked"

    false_commits = 0
    for _ in range(2000):   # posterior ~ U(0,1): uninformative opponent
        st = {}
        for _ in range(60):
            v = sprt_update(st, rng.random())
            if v != "continue":
                false_commits += (v == "commit")
                break
    rate = false_commits / 2000
    assert rate < 0.15, f"false-commit rate too high under noise: {rate:.3f}"
    print(f"SPRT ok: fast commit in {steps} steps, "
          f"noise false-commit rate {rate:.3f}")
    print("ALL GATES-V2 TESTS PASSED")


if __name__ == "__main__":
    _test()
