"""Contract tests for the tournament's successive-halving schedule.

This replaced a fixed full-roster budget (1,120+ games, ~51 min at 3 workers),
so a control-flow bug here would silently mis-crown: the crown reads whatever
`run_halving` returns. The play step is injected, so every property below is
checked without running a single episode.

    python tests/test_tourney_halving.py
"""
import os
import sys

import kaggriculture.pipeline.refresh_cycle as RC  # noqa: E402

SCREEN = ["s1.py", "s2.py", "s3.py", "s4.py"]
ROSTER = [f"r{i}.py" for i in range(12)]
SEEDS = (60000, 150000)


def make_play(strength, calls):
    """A stub play(): each agent wins in proportion to `strength[name]`."""
    def play(names, opps, seeds, n):
        games = len(opps) * n * 2 * len(seeds)      # both seats
        calls.append({"names": list(names), "n_opps": len(opps),
                      "seeds": len(seeds), "n": n,
                      "games": games * len(names)})
        return {nm: (int(round(strength.get(nm, 0.5) * games)), games,
                     strength.get(nm, 0.5) * 1000.0) for nm in names}
    return play


def test_returns_final_round_only():
    """The crown must see a full-roster, all-seed-set comparison."""
    cands = [f"c{i}" for i in range(14)]
    strength = {c: 0.4 + 0.03 * i for i, c in enumerate(cands)}
    strength["__incumbent__"] = 0.5
    calls = []
    final, finalists = RC.run_halving(cands, SCREEN, ROSTER, SEEDS,
                                      make_play(strength, calls),
                                      log=lambda *a: None)
    assert calls[-1]["n_opps"] == len(ROSTER), "final round must use the FULL roster"
    assert calls[-1]["seeds"] == len(SEEDS), "final round must use ALL seed sets"
    assert calls[-1]["n"] == 2, "final round must use the full match count"
    expect = set(finalists) | {"__incumbent__"}
    assert set(final) == expect, (set(final), expect)
    print(f"final round: {len(finalists)} finalists + incumbent, "
          f"{len(ROSTER)} referees, {len(SEEDS)} seed sets")


def test_incumbent_never_eliminated():
    """The incumbent is the control; a weak incumbent must still be measured."""
    cands = [f"c{i}" for i in range(10)]
    strength = {c: 0.9 for c in cands}
    strength["__incumbent__"] = 0.01              # worst agent in the field
    calls = []
    final, finalists = RC.run_halving(cands, SCREEN, ROSTER, SEEDS,
                                      make_play(strength, calls),
                                      log=lambda *a: None)
    for c in calls:
        assert "__incumbent__" in c["names"], "incumbent dropped from a round"
    assert "__incumbent__" in final
    print("incumbent present in all "
          f"{len(calls)} rounds despite being the weakest agent")


def test_strongest_candidate_survives():
    """Elimination must be monotone in strength -- the best cannot be cut."""
    cands = [f"c{i}" for i in range(14)]
    strength = {c: 0.30 + 0.01 * i for i, c in enumerate(cands)}
    strength["c13"] = 0.99                        # clearly the best
    strength["__incumbent__"] = 0.5
    _final, finalists = RC.run_halving(cands, SCREEN, ROSTER, SEEDS,
                                       make_play(strength, []),
                                       log=lambda *a: None)
    assert "c13" in finalists, (finalists, "strongest candidate was eliminated")
    print(f"strongest candidate survived to the final: {finalists}")


def test_cheaper_than_the_fixed_budget():
    """The whole point: fewer games than every candidate playing everything."""
    cands = [f"c{i}" for i in range(14)]
    strength = {c: 0.4 + 0.03 * i for i, c in enumerate(cands)}
    strength["__incumbent__"] = 0.5
    calls = []
    RC.run_halving(cands, SCREEN, ROSTER, SEEDS, make_play(strength, calls),
                   log=lambda *a: None)
    spent = sum(c["games"] for c in calls)
    old = (len(cands) + 1) * len(ROSTER) * 2 * 2 * len(SEEDS)
    assert spent < old, (spent, old)
    print(f"games: {spent:,} vs {old:,} for the old fixed budget "
          f"({old / spent:.2f}x cheaper)")


def test_rank_uses_score_not_margin():
    """Margin is a tie-break only; a huge margin must not outrank a better
    score. This is the currency rule the whole repo now runs on."""
    res = {"a": (5, 10, 1_000_000.0),      # score 0.50, colossal margin
           "b": (9, 10, 1.0)}              # score 0.90, trivial margin
    assert RC.rank_by_score(res, ["a", "b"])[0] == "b"
    # ...but it DOES break a genuine tie
    res2 = {"a": (5, 10, 10.0), "b": (5, 10, 999.0)}
    assert RC.rank_by_score(res2, ["a", "b"])[0] == "b"
    print("rank_by_score: score dominates, margin only breaks ties")


def test_no_candidates_is_not_a_crash():
    final, finalists = RC.run_halving([], SCREEN, ROSTER, SEEDS,
                                      make_play({}, []),
                                      log=lambda *a: None)
    assert finalists == [] and "__incumbent__" in final
    print("empty candidate pool handled")


if __name__ == "__main__":
    for fn in (test_returns_final_round_only, test_incumbent_never_eliminated,
               test_strongest_candidate_survives,
               test_cheaper_than_the_fixed_budget,
               test_rank_uses_score_not_margin,
               test_no_candidates_is_not_a_crash):
        fn()
    print("\nall halving checks passed")
