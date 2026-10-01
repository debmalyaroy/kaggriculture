"""Counterfactual judging of a policy-on build vs its policy-off twin.

Paired episodes (same opponent, seed, seat); each pair appends one row
{score_delta, judged_pair, ...} to models/lab/policy_judgments.jsonl -- the
evidence train_gates.policy_allowed() prices, scoped to the newest
generation. Promoted from .local/judge_policy.py (2026-08-14) for the daily
rehab job; opponents default to the newest refresh day's loss tapes + the
crowd guard.

    python src/experiments/judge_policy.py --on <policy_on.py> --off <twin.py>
        --label v25.2-policy-head
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
from kaggle_environments import make                             # noqa: E402

import kaggriculture.measure.win_metric as WM  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "policy_judgments.jsonl")
GUARD = os.path.join(ROOT, "data", "panel", "opp_r004_90558188_s0.py")


def newest_tapes():
    days = sorted(glob.glob(os.path.join(ROOT, "data", "refresh", "*",
                                         "loss_tapes")))
    if not days:
        return []
    return sorted(glob.glob(os.path.join(days[-1], "*.py")))


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(agent_path, opp_path, seed, seat, tag):
    a = load(agent_path, f"a{tag}")
    o = load(opp_path, f"o{tag}")
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": int(seed),
                              "actTimeout": 60, "runTimeout": 1000000})
    pair = [a.agent, o.agent] if seat == 0 else [o.agent, a.agent]
    env.run(pair)
    return (float(env.state[seat].reward or 0),
            float(env.state[1 - seat].reward or 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--on", required=True, help="policy-ON lab build")
    ap.add_argument("--off", required=True, help="policy-OFF twin")
    ap.add_argument("--label", required=True,
                    help="generation tag (judged_pair)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[91000, 91007])
    ap.add_argument("--max-opponents", type=int, default=12)
    args = ap.parse_args()

    opponents = newest_tapes()[:args.max_opponents]
    if os.path.exists(GUARD):
        opponents.append(GUARD)
    if not opponents:
        print("no opponents available -- nothing judged")
        return 0
    rows, n = [], 0
    for opp in opponents:
        for seed in args.seeds:
            for seat in (0, 1):
                n += 1
                b_on = play(args.on, opp, seed, seat, f"on{n}")
                b_off = play(args.off, opp, seed, seat, f"off{n}")
                s_on = WM.score(b_on[0], b_on[1])
                s_off = WM.score(b_off[0], b_off[1])
                rows.append({"score_delta": s_on - s_off,
                             "score_on": s_on, "score_off": s_off,
                             "bank_on": round(b_on[0], 1),
                             "bank_off": round(b_off[0], 1),
                             "opp": os.path.basename(opp), "seed": seed,
                             "seat": seat, "judged_pair": args.label,
                             "source": "daily-rehab"})
                print(f"[{n}] {os.path.basename(opp)[:26]} s{seed}/{seat}: "
                      f"on {s_on} off {s_off} d{s_on - s_off:+.1f}",
                      flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ds = [r["score_delta"] for r in rows]
    print(f"\n{len(rows)} judged; mean score_delta {statistics.mean(ds):+.4f} "
          f"(generation {args.label})")
    import kaggriculture.train.train_gates as TG
    print("policy_allowed:", TG.policy_allowed())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
