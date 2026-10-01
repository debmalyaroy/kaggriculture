"""Which referees actually discriminate between candidates?

A tournament referee earns its ~4 games per candidate by SEPARATING candidates.
A tape that every candidate beats, or that beats every candidate, costs full
price and returns no information: it shifts every score equally and changes no
ranking. Measured over real per-opponent results, those tapes can be dropped
with no loss of decision quality.

Discriminating power here is the variance of per-candidate scores against that
referee, plus a rank-agreement term: a referee whose candidate ordering is
identical to another's is redundant even if its variance is high.

Reads whatever per-opponent evidence exists -- the model registry's recorded
evals and any ablation json under .local/ -- so it works without re-running a
tournament.

    python src/referee_power.py
    python src/referee_power.py --json .local/referee_power.json
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# A referee below this score-variance separates nobody; the value is in units
# of expected score (0..1), so 0.01 is a 10-point spread in win rate.
MIN_VARIANCE = 0.010
# Two referees whose candidate orderings agree at least this closely are
# redundant; keep the one with more variance.
MAX_RANK_AGREEMENT = 0.95
# A verdict from two near-identical candidates is worthless: they beat the same
# referees, so every referee looks non-discriminating. Measured 2026-08-13 on
# the timeON/timeOFF pair -- four field-strata tapes showed variance 0.0000
# purely because both agents beat them 100%. Require a real spread of
# candidates before believing any DROP.
MIN_CANDIDATES = 5
# Never prune below this many referees, whatever the variance says.
MIN_KEEP = 4
# Referees that exist for REGIME COVERAGE rather than discrimination. The field
# strata span the opponent sell-volume range (413..3,763 units) so that a
# price-regime change can be seen at all; dropping them for low variance would
# undo exactly the panel-bias fix they were added for.
PROTECTED_PREFIXES = ("tape_9258", "tape_9259")


def _from_ablation():
    """{referee: {agent: score}} from any ablation json in .local/."""
    out = {}
    for p in glob.glob(os.path.join(ROOT, ".local", "ablation*.json")):
        try:
            rows = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            opp = os.path.basename(str(r.get("opp") or ""))
            agent = str(r.get("agent") or "")
            g = float(r.get("games") or 0)
            if not opp or not agent or g <= 0:
                continue
            out.setdefault(opp, {}).setdefault(agent, []).append(
                float(r.get("wins") or 0) / g)
    return {opp: {a: statistics.mean(v) for a, v in d.items()}
            for opp, d in out.items()}


def _from_registry():
    """{referee: {agent: score}} from the model registry's recorded evals."""
    out = {}
    try:
        import kaggriculture.data.registry as registry
        data = registry.load() if hasattr(registry, "load") else None
    except Exception:                                           # noqa: BLE001
        data = None
    if not isinstance(data, dict):
        return out
    for name, rec in (data.get("models") or {}).items():
        for ev in (rec.get("evals") or []):
            opp = os.path.basename(str(ev.get("opponent") or ""))
            wr = ev.get("win_rate")
            if opp and wr is not None:
                out.setdefault(opp, {})[name] = float(wr)
    return out


def gather():
    merged = _from_ablation()
    for opp, d in _from_registry().items():
        merged.setdefault(opp, {}).update(d)
    return merged


def _agreement(a, b):
    """Fraction of candidate PAIRS both referees order the same way."""
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return 0.0
    same = tot = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            da, db = a[x] - a[y], b[x] - b[y]
            if da == 0 or db == 0:
                continue
            tot += 1
            if (da > 0) == (db > 0):
                same += 1
    return (same / tot) if tot else 0.0


def analyse(table, min_variance=MIN_VARIANCE,
            max_agreement=MAX_RANK_AGREEMENT, protected=()):
    """(rows, keep, drop). rows = [(referee, n, mean, variance, verdict)].

    Refuses to prune on thin evidence: with fewer than MIN_CANDIDATES distinct
    candidates every referee looks non-discriminating, and `protected` referees
    are kept regardless because they are in the panel for regime coverage.
    """
    rows = []
    for opp, d in table.items():
        if len(d) < 2:
            rows.append((opp, len(d), (statistics.mean(d.values())
                                       if d else 0.0), 0.0,
                         "too few candidates to judge"))
            continue
        vals = list(d.values())
        rows.append((opp, len(d), statistics.mean(vals),
                     statistics.pvariance(vals), ""))
    rows.sort(key=lambda r: -r[3])

    n_cands = max((r[1] for r in rows), default=0)
    if n_cands < MIN_CANDIDATES:
        return rows, list(table), [
            ("__all__", f"only {n_cands} candidate(s) in the evidence "
                        f"(need >={MIN_CANDIDATES}) -- every referee looks "
                        f"non-discriminating when candidates are alike; "
                        f"refusing to prune")]
    keep, drop = [], []
    for opp, n, mean, var, note in rows:
        if note:
            keep.append(opp)            # unjudged referees are kept, not cut
            continue
        if any(opp.startswith(px) for px in protected):
            keep.append(opp)            # regime coverage, not discrimination
            continue
        if var < min_variance:
            drop.append((opp, f"variance {var:.4f} < {min_variance} "
                              f"(mean score {mean:.2f}) -- separates nobody"))
            continue
        dup = None
        for kept in keep:
            if _agreement(table[opp], table.get(kept, {})) >= max_agreement:
                dup = kept
                break
        if dup:
            drop.append((opp, f"orders candidates like {dup} "
                              f"(agreement >= {max_agreement})"))
        else:
            keep.append(opp)
    return rows, keep, drop


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-variance", type=float, default=MIN_VARIANCE)
    ap.add_argument("--max-agreement", type=float, default=MAX_RANK_AGREEMENT)
    ap.add_argument("--json", dest="as_json")
    args = ap.parse_args()

    table = gather()
    if not table:
        print("no per-opponent evidence found (registry evals or "
              ".local/ablation*.json). Run a tournament or an ablation first.")
        return 0
    rows, keep, drop = analyse(table, args.min_variance, args.max_agreement,
                               protected=PROTECTED_PREFIXES)
    print(f"{len(table)} referee(s) with per-candidate results\n")
    print(f"{'referee':<40}{'cands':>6}{'mean':>8}{'variance':>10}  verdict")
    dropped = dict(drop)
    for opp, n, mean, var, note in rows:
        verdict = note or ("DROP: " + dropped[opp] if opp in dropped
                           else "keep")
        print(f"{opp:<40}{n:>6}{mean:>8.2f}{var:>10.4f}  {verdict}")
    print(f"\nkeep {len(keep)}, drop {len(drop)}")
    if len(drop) == 1 and drop[0][0] == "__all__":
        print(f"NO PRUNING: {drop[0][1]}")
        drop = []
    if drop:
        saving = len(drop) / max(1, len(table))
        print(f"pruning removes {100 * saving:.0f}% of referee games with no "
              f"ranking information lost")
    if args.as_json:
        json.dump({"keep": keep, "drop": dropped,
                   "rows": [{"referee": r[0], "candidates": r[1],
                             "mean": r[2], "variance": r[3]} for r in rows]},
                  open(args.as_json, "w", encoding="utf-8"), indent=1)
        print(f"-> {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
