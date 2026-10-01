"""Crown measure v2: per-strong-opponent win rate, many seeds, both seats,
sign-tested -- the measure the #1 player describes and the ladder actually
uses, replacing the pooled win% + 10pp gate that froze the crown.

WHY (kaggriculture discussion 736219, Ryo Hasegawa #1; 736214, Luka Duvanov):
  * per-seed money sd ~$13,747 -- a handful of seeds is pure noise; use dozens.
  * only W/L matters, matched near your rating -- optimize WIN RATE vs STRONG
    opponents, never pooled coin margin.
  * judge by win rate vs EACH strong opponent (a change that flips two wins
    against the #2 bot is a loss even if it adds coins on average).
  * a seat-0 edge exists -- pair both seats so it cancels.
  * treat sub-~50-rating (a few pp) moves as noise -> a SIGN TEST decides,
    not a fixed margin.

The direct measure (this file, decider). A per-turn win-probability control
variate (variance reducer) lives in crown_winprob.py and only ever tightens
seeds; it never overrides the direct sign test.

    python src/crown_eval.py --incumbent agents/v29.0_route.py \
        --candidates <route_id_or_agent> ... --seeds 40
    python src/crown_eval.py --incumbent agents/v29.0_route.py --auto-pool 30
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

OUT = os.path.join(ROOT, "models", "factory", "crown_eval.json")
TOP_TEAMS = ["Ryo Hasegawa", "tetsuya", "Arman Tuganbaev", "Crop Dusta",
             "カワシギ", "Izzoudine Mohamed KANTA", "Subramanya N",
             "Xiaowenhao404", "peikopon", "ReCurSiON", "Efe Can Celiksoy",
             "Thomas Tschinkel"]


def champion_panel(idx, size):
    from kaggriculture.measure.champion_screen import best_recent_win
    panel = []
    for t in TOP_TEAMS[:size]:
        rec = best_recent_win(idx, t)
        if rec:
            panel.append(rec)
    return panel


def sign_test(pos, neg):
    import math
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incumbent", required=True,
                    help="route id or agent .py of the live flagship")
    ap.add_argument("--candidates", nargs="*", default=[],
                    help="route ids or agent .py paths to test vs incumbent")
    ap.add_argument("--auto-pool", type=int, default=0,
                    help="also pull the top-N recent winning routes from the "
                         "index as candidates")
    ap.add_argument("--panel-size", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    import kaggriculture.data.routes as R
    import kaggriculture.pipeline.sell_search as SS
    idx = R.load_index()
    os.makedirs(SS.WORK, exist_ok=True)

    panel = champion_panel(idx, args.panel_size)
    if len(panel) < 3:
        raise SystemExit("need >=3 champion opponents; index too thin")
    panel_teams = [r.get("team") for r in panel]
    panel_tapes = [SS.write_tape(R.load_route(r["id"]),
                                 os.path.join(SS.WORK, "ce_p%d.tape" % i))
                   for i, r in enumerate(panel)]
    print("panel (%d strong opponents): %s" % (
        len(panel), [r["id"] for r in panel]))

    def resolve(ref, tag):
        if ref.endswith(".py"):
            # incumbent/candidate given as an agent: its embedded route id is
            # in the header -- reuse the mined route (serve plays the tape).
            import re
            head = open(os.path.join(ROOT, ref), encoding="utf-8").read(4000)
            m = re.search(r"(\d{6,}_s[01])", head)
            if m and m.group(1) in idx["routes"]:
                return SS.write_tape(R.load_route(m.group(1)),
                                     os.path.join(SS.WORK, tag + ".tape"))
            raise SystemExit("cannot resolve %s to a route tape" % ref)
        return SS.write_tape(R.load_route(ref),
                             os.path.join(SS.WORK, tag + ".tape"))

    cand_refs = list(args.candidates)
    if args.auto_pool:
        pool = sorted((r for r in idx["routes"].values()
                       if r.get("won") and r.get("team") not in panel_teams),
                      key=lambda r: -float(r.get("bank", 0)))[:args.auto_pool]
        cand_refs += [r["id"] for r in pool]
    # de-dup, keep incumbent first
    seen, ordered = set(), []
    for ref in [args.incumbent] + cand_refs:
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    print("evaluating %d agents (incl. incumbent) over %d seeds x 2 seats vs "
          "%d opponents" % (len(ordered), args.seeds, len(panel)))

    tapes = [resolve(ref, "ce_c%d" % i) for i, ref in enumerate(ordered)]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    evals = SS.batch_eval(tapes, panel_tapes, seeds, args.threads)
    inc = evals[0]

    def per_opp(ev):
        agg = {}
        for (oi, s), sc in ev["cells"].items():
            agg.setdefault(oi, []).append(sc)
        return {panel_teams[oi]: sum(v) / len(v) for oi, v in agg.items()}

    inc_wr = inc["score"]
    inc_by = per_opp(inc)
    print("\nINCUMBENT %s: win rate vs panel %.3f" % (ordered[0], inc_wr))
    for t, w in sorted(inc_by.items(), key=lambda kv: -kv[1]):
        print("   vs %-22s %.3f" % (t, w))

    rows = []
    for i, ref in enumerate(ordered[1:], start=1):
        ev = evals[i]
        common = sorted(set(ev["cells"]) & set(inc["cells"]))
        pos = sum(1 for k in common if ev["cells"][k] > inc["cells"][k])
        neg = sum(1 for k in common if ev["cells"][k] < inc["cells"][k])
        p = sign_test(pos, neg)
        wr = ev["score"]
        by = per_opp(ev)
        # per-opponent: does it FLIP any strong opponent negative vs incumbent?
        flips_down = [t for t in by if by[t] + 1e-9 < inc_by.get(t, 0)
                      and inc_by.get(t, 0) >= 0.5]
        better = wr > inc_wr and pos > neg and p < 0.05
        rows.append({"ref": ref, "win_rate": round(wr, 4),
                     "vs_incumbent_pp": round(100 * (wr - inc_wr), 2),
                     "discordant_pos": pos, "discordant_neg": neg,
                     "sign_p": round(p, 5),
                     "flips_strong_opponent_down": flips_down,
                     "genuine_improvement": bool(better and not flips_down)})
        tag = "  ** GENUINE IMPROVEMENT **" if rows[-1][
            "genuine_improvement"] else ("  (flips %s)" % flips_down
                                         if flips_down else "")
        print("  %-28s wr %.3f (%+.2fpp)  discordant %d-%d  p=%.4f%s" % (
            os.path.basename(ref), wr, 100 * (wr - inc_wr), pos, neg, p, tag))

    rows.sort(key=lambda r: -r["win_rate"])
    winners = [r for r in rows if r["genuine_improvement"]]
    verdict = ("%d genuine improvement(s); best = %s (+%.2fpp, p=%.4f)"
               % (len(winners), winners[0]["ref"],
                  winners[0]["vs_incumbent_pp"], winners[0]["sign_p"])
               if winners else
               "NO candidate genuinely beats the incumbent -- keep it aging")
    print("\nVERDICT: %s" % verdict)
    json.dump({"when": __import__("datetime").datetime.now()
               .isoformat(timespec="seconds"),
               "incumbent": {"ref": ordered[0], "win_rate": inc_wr,
                             "per_opponent": inc_by},
               "seeds": args.seeds, "panel": panel_teams,
               "rows": rows, "winners": winners, "verdict": verdict},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
