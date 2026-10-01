"""Simulation diff: your bot vs the top-20, on identical states.

Two complementary comparisons, over a whole folder of replays.

**A. Counterfactual policy diff.** For every turn of a top player's episode we
hand our agent that exact observation and record what it would have done. Same
board, same market, same shed. Disagreement is localised to a turn and an op,
not inferred from the final score.

**B. Aggregate strategy diff.** Their portfolio, hiring curve, land timing and
sell mix, against ours over the same span.

Together they answer "what do they do that we don't", at both resolutions.

    python -m kaggriculture.engine.diff_sim                            # everything under data/episodes
    python -m kaggriculture.engine.diff_sim --replays data/replays --top-n 20
    python -m kaggriculture.engine.diff_sim --agent agents/v2_tuned.py --limit 240
"""
from kaggriculture.paths import ROOT
import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.measure.action_diff as ad  # noqa: E402
import kaggriculture.data.replay_analysis as ra  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def find_replays(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        elif p.endswith(".json"):
            out.append(p)
    return [p for p in out if os.path.basename(p) != "manifest.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replays", nargs="*",
                    default=[os.path.join(ROOT, "data", "episodes")])
    ap.add_argument("--agent", default=None)
    ap.add_argument("--limit", type=int, default=None, help="turns per replay")
    ap.add_argument("--max-replays", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "diff-simulation.md"))
    args = ap.parse_args()

    agent = args.agent
    if not agent:
        c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        agent = c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")

    files = find_replays(args.replays)[: args.max_replays]
    if not files:
        sys.exit("no replays found.\n"
                 "  python download_data.py            (needs network + kaggle CLI)\n"
                 "  or point --replays at a folder of episode JSONs")

    print(f"comparing {os.path.basename(agent)} against {len(files)} replays\n")
    agree_num = agree_den = 0
    cm = collections.Counter()
    net = collections.Counter()
    market_net = collections.Counter()
    phase = collections.defaultdict(lambda: [0, 0])
    theirs_prof, ours_ref = [], []

    for f in files:
        try:
            res = ad.analyse(f, agent, "winner", args.limit)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {os.path.basename(f)}: {exc}")
            continue
        agree_num += res["agree"]["farmer"]
        agree_den += res["total"]["farmer"]
        for k, v in res["farmer_cm"].items():
            cm[k] += v
            net[k[1]] += v
            net[k[0]] -= v
        for k, v in res["market_cm"].items():
            market_net[k] += v
        for ph, (a, t) in res["by_phase"].items():
            phase[ph][0] += a
            phase[ph][1] += t
        prof = ra.profile_replay(f)
        if prof:
            p = res["focus_player"]
            theirs_prof.append(prof["players"][p])
        print(f"  {os.path.basename(f):<24} agreement "
              f"{100.0*res['agree']['farmer']/max(1,res['total']['farmer']):5.1f}%  "
              f"({res['focus_name']} ${res['focus_bank']:,.0f})", flush=True)

    L = []
    A = L.append
    A("# Simulation diff — our bot vs the top of the ladder\n")
    A(f"Agent: `{os.path.basename(agent)}` · {len(files)} episodes · "
      f"{agree_den:,} turns replayed\n")
    A(f"\n## A. Same board, different move\n")
    A(f"**Overall farmer-op agreement: {100.0*agree_num/max(1,agree_den):.1f}%**\n")
    A("\n| phase | agreement | turns |")
    A("|---|---:|---:|")
    for ph in ("early (d0-9)", "mid (d10-19)", "late (d20-29)"):
        if ph in phase:
            a, t = phase[ph]
            A(f"| {ph} | {100.0*a/max(1,t):.1f}% | {t:,} |")

    A("\n### Biggest divergences\n")
    A("| they did | we would | turns | share |")
    A("|---|---|---:|---:|")
    for (t_op, o_op), n in sorted(((k, v) for k, v in cm.items() if k[0] != k[1]),
                                  key=lambda kv: -kv[1])[:18]:
        A(f"| `{t_op}` | `{o_op}` | {n:,} | {100.0*n/max(1,agree_den):.1f}% |")

    A("\n### Net action budget — ours minus theirs\n")
    A("| op | net turns |")
    A("|---|---:|")
    for op, v in sorted(net.items(), key=lambda kv: -abs(kv[1]))[:14]:
        if v:
            A(f"| `{op}` | {v:+,} |")
    mv = sum(v for k, v in net.items() if k in MOVES)
    A(f"\nNet movement: **{mv:+,}** turns "
      f"({'we walk more than they do' if mv > 0 else 'we walk less than they do'}).\n")

    if market_net:
        A("\n### Market orders — ours minus theirs\n")
        A("| order | net |")
        A("|---|---:|")
        for op, v in sorted(market_net.items(), key=lambda kv: -abs(kv[1])):
            A(f"| `{op}` | {v:+,} |")

    if theirs_prof:
        A("\n## B. Aggregate strategy\n")
        A("| metric | top-ladder median |")
        A("|---|---:|")
        for lbl, fn in (("final bank", lambda p: p["final_bank"]),
                        ("peak herd", lambda p: p["peak_herd"]),
                        ("peak crop tiles", lambda p: p["peak_crops"]),
                        ("peak hands/day", lambda p: p["peak_hands"]),
                        ("first animal day", lambda p: p["first_animal_day"] or 0)):
            A(f"| {lbl} | {statistics.median([fn(p) for p in theirs_prof]):,.0f} |")
        A("\n| asset | their median peak tiles |")
        A("|---|---:|")
        for k in list(ra.CROPS) + list(ra.ANIMALS):
            A(f"| {k} | {statistics.median([p['peak'].get(k,0) for p in theirs_prof]):.0f} |")

    A("\n## How to read this\n")
    A("- States come from *their* trajectory, so our errors never compound — but "
      "we are being asked about boards we would never have built. A single "
      "disagreement means nothing; a **systematic skew** is the signal.\n")
    A("- A large positive net on movement, or an order type they issue "
      "constantly and we never do, is a concrete policy gap worth testing.\n")
    A("- Everything here is a hypothesis. It has to clear three independent "
      "seed sets in `src/kaggriculture/measure/evaluate.py` before it is believed.\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\noverall agreement {100.0*agree_num/max(1,agree_den):.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
