"""Held-out test of a reactive shell v2 candidate (never used by training, the screen or CMA-ES).

    python python/rshell/test.py --rshell F [--knobs K] [--chain-off-file C] [--it 790] [--name NAME] [--threads 24]

  closed loop  the bank's HELD-OUT split: 3 seeds per world (192) x 7 lineage opponents x both seats, paired vs the
               reference (i790 + big1); reported overall, per opponent and PER WORLD (which worlds got worse)
  open loop    the held-out ladder half (data/tapes/ladder_val), our real losses split B, and the band gate
               (991 real players' tapes) for the candidate AND the reference, both vs v63.1_rl
  latency      worst turn over 32 closed-loop games (must stay well under the 1 s budget)
Writes data/rshell/test_NAME.json.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402


def band(policy_it, cand_args, name, threads):
    pol = os.path.join(P.RUN, "snapshots", f"i{policy_it:06d}.bin")
    r = subprocess.run([sys.executable, os.path.join(P.RL, "python", "band_gate.py"), "--cand", pol, "--cand-args", cand_args, "--name", name,
                        "--ref-profiles", P.V64, "--ref-profile", "35", "--ref-group", "35,35,36", "--threads", str(threads)],
                       cwd=P.RL, capture_output=True, text=True, env={**os.environ, "KRL_BIN": P.BIN})
    line = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", line)
    return {"line": line, "losses_below_2500": int(m.group(1)) if m else None, "rate_2500plus": float(m.group(2)) if m else None,
            "paired_vs_v631": [int(m.group(3)), int(m.group(4))] if m else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rshell", default="", help="shell v2 config ('' = none: tests the v1 shell + knobs + chain-off only)")
    ap.add_argument("--knobs", default=None)
    ap.add_argument("--chain-off-file", default=None)
    ap.add_argument("--v1-shell", default=P.BIG1, help="keep the v1 sales shell under shell v2 ('' = none); diagnostic 2026-09-27: big1 is worth +253 net closed loop")
    ap.add_argument("--it", type=int, default=790)
    ap.add_argument("--name", default="cand")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    extra = f"--rshell {os.path.abspath(a.rshell)}" if a.rshell else ""
    if a.knobs:
        extra += f" --knob-over {os.path.abspath(a.knobs)}"
    off = open(a.chain_off_file).read().strip() if a.chain_off_file and os.path.exists(a.chain_off_file) else ""
    if off:
        extra += f" --chain-off {off}"
    cand = P.ppo_args(a.it, shell=a.v1_shell or None, extra=extra)
    seeds = P.bank_seeds("heldout", 3, 0)
    world = {s: w for w, s in seeds}
    rep = {"candidate": cand, "reference": P.REF}
    ref_cl, _ = P.closed_loop(P.REF, seeds, a.threads)
    cl, worlds = P.closed_loop(cand, seeds, a.threads)
    fit = lambda d: {k: v for k, v in d.items() if k[0] != "mirror"}  # noqa: E731
    rep["closed"] = P.compare(fit(cl), fit(ref_cl))
    rep["closed_incl_mirror"] = P.compare(cl, ref_cl)
    rep["world_mismatch"] = sum(str(w).startswith("MISMATCH") for w in worlds.values())
    rep["by_opponent"] = {o: P.compare({k: v for k, v in cl.items() if k[0] == o}, {k: v for k, v in ref_cl.items() if k[0] == o}) for o in P.OPPONENTS}
    by_w = defaultdict(lambda: [0, 0])
    for k in cl:
        if k[0] != "mirror" and k in ref_cl and cl[k] != ref_cl[k]:
            by_w[world[k[1]]][0 if cl[k] > ref_cl[k] else 1] += 1
    rep["by_world"] = {w: {"better": b, "worse": x} for w, (b, x) in sorted(by_w.items())}
    rep["worlds_worse"] = sorted(w for w, (b, x) in by_w.items() if x > b)
    print(f"[test] closed loop (held-out 64 worlds x 3 seeds x 6 lineage opp x 2 seats, mirror excluded): +{rep['closed']['better']}/-{rep['closed']['worse']} p {rep['closed']['p']}; "
          f"worlds worse {len(rep['worlds_worse'])}/64; mismatches {rep['world_mismatch']}", flush=True)
    for o, v in rep["by_opponent"].items():
        print(f"[test]   vs {o:9s}: +{v['better']}/-{v['worse']} (wins {v['wins']} vs {v['ref_wins']})", flush=True)
    sets = {"ladder_val": sorted(glob.glob(os.path.join(P.RL, "data", "tapes", "ladder_val", "**", "*.json"), recursive=True)),
            "losses_B": P.tape_sets("B")["losses"]}
    rol, col = P.open_loop(P.REF, sets, a.threads), P.open_loop(cand, sets, a.threads)
    rep["open"] = {k: P.compare(col[k], rol[k]) for k in sets}
    rep["losses_B_won"] = {"cand": int(sum(v == 1 for v in col["losses_B"].values())), "ref": int(sum(v == 1 for v in rol["losses_B"].values())), "of": len(sets["losses_B"])}
    for k, v in rep["open"].items():
        print(f"[test] {k}: +{v['better']}/-{v['worse']} (wins {v['wins']} vs {v['ref_wins']})", flush=True)
    rep["band"] = {"cand": band(a.it, ((f"--shell {a.v1_shell} " if a.v1_shell else "") + extra).strip(), f"rshell-{a.name}", a.threads), "ref": band(a.it, f"--shell {P.BIG1}", f"rshell-ref-i{a.it}", a.threads)}
    print(f"[test] band: cand {rep['band']['cand']['losses_below_2500']} losses <2500 ({rep['band']['cand']['paired_vs_v631']} vs v63.1_rl), "
          f"ref {rep['band']['ref']['losses_below_2500']} ({rep['band']['ref']['paired_vs_v631']})", flush=True)
    r = subprocess.run([P.SELFPLAY, "--seeds", ",".join(str(s) for _, s in seeds[:32]), "--threads", "4", "--a-args", cand, "--b-args", P.OPPONENTS["v63"]],
                       cwd=P.RL, capture_output=True, text=True)
    worst = max((float(x.split("\t")[3]) for x in r.stdout.splitlines() if x.count("\t") >= 4), default=None)
    rep["worst_turn_ms"] = worst / 1000 if worst else None
    print(f"[test] worst turn {rep['worst_turn_ms']} ms", flush=True)
    out = os.path.join(P.RL, "data", "rshell", f"test_{a.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(f"[test] -> {out}", flush=True)


if __name__ == "__main__":
    main()
