"""Tournament win-rate-vs-rating gate.

The final is rated pairwise (BT/Elo), so the objective is a PROFILE, not an
average: dominate everyone below 2500 (a loss there tanks rating), and hold a
70-90% win rate against >=2500. This gate makes that an explicit pass/fail.

    python -m kaggriculture.measure.tournament_gate <agent.py|policy.pt> [--seeds 8]

Reuses the crown panel (banded refs) + eval_harness.play_two (both seats).
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import os

import kaggriculture.measure.crown_panel as CP
import kaggriculture.measure.eval_harness as EH

SPLIT_RATING = 2500.0
LOW_BAR = 0.98          # < 2500: must ~never lose (any loss flagged)
HIGH_LO, HIGH_HI = 0.70, 0.90     # >= 2500: maintain 70-90%


def _refs():
    """(name, path, rating) for every playable crown-panel ref."""
    try:
        man = CP.load()
        bands = man.get("bands") if isinstance(man, dict) else None
    except Exception:
        bands = None
    out = []
    if bands:
        for _b, lst in bands.items():
            for r in lst:
                path = r.get("tape") or r.get("path") or r.get("origin")
                if not path:
                    continue
                p = path if os.path.isabs(path) else os.path.join(ROOT, path)
                if os.path.exists(p):
                    out.append((r.get("name", os.path.basename(p)),
                                p, float(r.get("rating", 0) or 0)))
    return out


def _agent_from(x, rails=None):
    """Load a candidate: a .pt policy (via policy_agent) or a .py/callable agent."""
    if isinstance(x, str) and x.endswith(".pt"):
        from kaggriculture.trackp import policy_agent as PA
        return PA.make_agent(x, rail_config=rails, sample=False)
    return EH._as_agent(x)


def evaluate(agent, seeds=8, rails=None):
    A = _agent_from(agent, rails)
    bank = EH.gate_seeds(seeds)
    per_ref = {}
    for name, path, rating in _refs():
        try:
            R = EH._as_agent(path)
        except Exception:
            continue
        cells = []
        for s in bank:
            b0, b1 = EH.play_two(A, R, s)
            cells.append(1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0))
            b0b, b1b = EH.play_two(R, A, s)
            cells.append(1.0 if b1b > b0b else (0.5 if b1b == b0b else 0.0))
        per_ref[name] = {"rating": rating, "win": sum(cells) / len(cells),
                         "games": len(cells), "losses": cells.count(0.0)}
    return per_ref


def gate(agent, seeds=8, verbose=True, rails=None):
    per_ref = evaluate(agent, seeds, rails)
    below = [v for v in per_ref.values() if v["rating"] < SPLIT_RATING]
    above = [v for v in per_ref.values() if v["rating"] >= SPLIT_RATING]

    def wr(xs):
        return sum(v["win"] for v in xs) / len(xs) if xs else None
    below_wr, above_wr = wr(below), wr(above)
    below_losses = sum(v["losses"] for v in below)
    passed = True
    reasons = []
    if below:
        if below_wr < LOW_BAR or below_losses > 0:
            passed = False
            reasons.append(f"<2500 winrate {below_wr:.3f} (bar {LOW_BAR}), "
                           f"{below_losses} losses to weak refs")
    if above:
        if above_wr < HIGH_LO:
            passed = False
            reasons.append(f">=2500 winrate {above_wr:.3f} < {HIGH_LO}")
        elif above_wr > HIGH_HI:
            reasons.append(f">=2500 winrate {above_wr:.3f} > {HIGH_HI} "
                           f"(over-target -- ok, but check field)")
    res = {"below_2500_winrate": below_wr, "above_2500_winrate": above_wr,
           "below_losses": below_losses, "n_below": len(below), "n_above": len(above),
           "pass": passed, "reasons": reasons, "per_ref": per_ref}
    if verbose:
        print(f"[tourney-gate] <2500: winrate={below_wr} losses={below_losses} "
              f"(n={len(below)})")
        print(f"[tourney-gate] >=2500: winrate={above_wr} (n={len(above)})")
        print(f"[tourney-gate] PASS={passed}" + (f"  {reasons}" if reasons else ""))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help="agent .py or policy .pt")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    r = gate(a.agent, a.seeds)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
