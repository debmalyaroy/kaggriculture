"""P3.6 -- PLANNER GUARD + sim-to-real check (fresh implementation).

PLANNER GUARD (the sign-tested shape used everywhere in this project,
re-implemented here because Track P shares no code):
  * >= MIN_JUDGED paired official-engine games (planner vs incumbent,
    same seeds, both seats)
  * positive mean score edge
  * one-sided sign test on discordant pairs, p <= ALPHA
  * newest-generation scoped: only rows tagged with the CURRENT generation
    count once that generation has >= MIN_JUDGED rows; stale positive
    evidence from an older build can never open the gate

No evidence or negative evidence means NO. The gate is keyed on evidence
QUALITY, never on volume alone.

Sim-to-real: a Rust-trained candidate must re-validate on the official
engine -- its official-engine anchor score must not fall more than
SIM2REAL_DROP below its serve-mode score on the same opponents.

Judgment rows live in models/trackp/planner_judgments.jsonl:
  {"generation": ..., "seed": ..., "seat": ..., "bank_me": ..,
   "bank_opp": .., "opponent": ..., "ts": ...}
"""
from __future__ import annotations

import json
import math
import os

try:
    from . import common
except ImportError:
    import sys
    from kaggriculture.trackp import common

MIN_JUDGED = 25
ALPHA = 0.05
SIM2REAL_DROP = 0.15
JUDGMENTS = os.path.join(common.MODELS, "planner_judgments.jsonl")


def record(rows: list):
    with open(JUDGMENTS, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def load_rows() -> list:
    if not os.path.exists(JUDGMENTS):
        return []
    out = []
    with open(JUDGMENTS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def newest_generation(rows: list) -> tuple:
    tags = [r.get("generation") for r in rows if r.get("generation")]
    if not tags:
        return rows, "pooled (no generation tags)"
    newest = tags[-1]
    gen = [r for r in rows if r.get("generation") == newest]
    if len(gen) >= MIN_JUDGED:
        return gen, f"generation '{newest}'"
    return gen, f"generation '{newest}' (still gathering)"


def _binom_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5) -- one-sided sign test."""
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i)
    return total / (2.0 ** n)


def planner_allowed(verbose: bool = True) -> dict:
    rows_all = load_rows()
    rows, scope = newest_generation(rows_all)
    n = len(rows)
    verdict = {"allowed": False, "scope": scope, "n": n,
               "min_judged": MIN_JUDGED}
    if n < MIN_JUDGED:
        verdict["reason"] = f"only {n} judged games (< {MIN_JUDGED})"
        if verbose:
            print(json.dumps(verdict))
        return verdict
    wins = sum(1 for r in rows if r["bank_me"] > r["bank_opp"])
    losses = sum(1 for r in rows if r["bank_me"] < r["bank_opp"])
    draws = n - wins - losses
    score = (wins + 0.5 * draws) / n
    p = _binom_tail(wins, wins + losses) if wins + losses else 1.0
    verdict.update({"wins": wins, "losses": losses, "draws": draws,
                    "score": round(score, 4), "sign_p": round(p, 5)})
    if score <= 0.5:
        verdict["reason"] = "no positive edge"
    elif p > ALPHA:
        verdict["reason"] = f"sign test p={p:.4f} > {ALPHA}"
    else:
        verdict["allowed"] = True
        verdict["reason"] = "positive, sign-tested, generation-scoped"
    if verbose:
        print(json.dumps(verdict))
    return verdict


def expected_score(scores) -> float:
    """Mean win-score of a list of per-cell results (win 1, draw 0.5, loss 0).

    Mirrors `win_metric.expected_score`. See `paired_test` for why the lane
    carries its own copy.
    """
    s = list(scores)
    return sum(s) / len(s) if s else 0.0


def paired_test(a_scores, b_scores) -> dict:
    """McNemar exact on discordant pairs, in win units. A == candidate.

    **This is a deliberate re-implementation of `win_metric.paired_test`, not
    a wrapper.** `win_metric` is a bandit/route module and
    `tests/test_trackp.py::test_isolation` forbids Track P from importing one
    -- the lane rule in CLAUDE.md is that Track P shares zero code with that
    lane, so the verdicts of the two lanes cannot be silently coupled by an
    edit to a shared file. The arithmetic is pinned identical to
    `win_metric.paired_test` by `tests/test_trackp.py::test_paired_test`,
    which imports both when `win_metric` is importable and skips otherwise.

    Same seeds, same seats, same opponents, so the games pair up; only the
    DISCORDANT pairs carry information. Returns the score difference, the
    discordant counts, the two-sided exact p-value and a normal-approximation
    95% CI on the paired difference.
    """
    a, b = list(a_scores), list(b_scores)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win_a = sum(1 for x, y in zip(a, b) if x > y)
    win_b = sum(1 for x, y in zip(a, b) if y > x)
    disc = win_a + win_b
    if disc == 0:
        p = 1.0
    else:
        k = min(win_a, win_b)
        # two-sided exact binomial: 2 * P(X <= k), X ~ Bin(disc, 0.5)
        p = min(1.0, 2.0 * _binom_tail(disc - k, disc))
    diff = (sum(a) - sum(b)) / n if n else 0.0
    if n > 1:
        var = sum(((x - y) - diff) ** 2 for x, y in zip(a, b)) / (n - 1)
        half = 1.959964 * math.sqrt(var / n)
    else:
        half = float("nan")
    return {"n_pairs": n, "score_a": expected_score(a),
            "score_b": expected_score(b), "score_diff": diff,
            "better_a": win_a, "better_b": win_b, "discordant": disc,
            "p_value": p, "ci95": (diff - half, diff + half),
            "significant": bool(p < 0.05)}


def sim_to_real(serve_score: float, official_score: float) -> dict:
    drop = serve_score - official_score
    ok = drop <= SIM2REAL_DROP
    return {"serve_score": serve_score, "official_score": official_score,
            "drop": round(drop, 4), "max_drop": SIM2REAL_DROP, "ok": ok}


if __name__ == "__main__":
    planner_allowed()
