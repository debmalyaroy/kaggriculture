"""Paired test of a v63.5_rl fix (extra agent args) against v63.5_rl itself (RCA 2026-09-27 fixes F1-F5).

    python python/rshell/fix_test.py --name v219_2 --extra "--knob-over configs/fixes/v219_min2.json" [--threads 16] [--band]

  loss tapes   every public-25 loss (data/rshell/public25/rca/tapes, open loop: the public agent's recorded
               actions; our side reproduces the tournament exactly with no fix) -> how many turn into wins
  lineage      64 worlds x 3 held-out bank seeds x 6 lineage opponents x 2 seats, closed loop (mirror excluded)
  real tapes   training ladder half, 600 training band tapes, our real losses split B
  band         (--band) the 991-tape real-player gate vs v63.1_rl, for the fix and for v63.5_rl
Writes data/rshell/fixes/<name>.json.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402

V635 = P.ppo_args(790, extra="--chain-off r127,sm,r95")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--policy-iter", type=int, default=790, help="run B checkpoint of the reference (910 = v63.6_rl)")
    ap.add_argument("--seed-offset", type=int, default=0, help="seed block (3 seeds per world)")
    ap.add_argument("--seed-split", default="heldout", help="heldout (3/world) or train (131/world; CMA fitness only uses the first ~50 -> offset 40 = seeds 120-122, never tuned on)")
    ap.add_argument("--ref-extra", default="", help="extra args of the REFERENCE too (e.g. v63.7_rl's --rshell/--knob-over/--endg); the candidate gets --extra instead")
    a = ap.parse_args()
    global V635
    base = P.ppo_args(a.policy_iter, extra="--chain-off r127,sm,r95")
    V635 = f"{base} {a.ref_extra}".strip()
    cand = f"{base} {a.extra}"
    rep = {"name": a.name, "extra": a.extra, "ref": V635}
    losses = sorted(glob.glob(os.path.join(P.RL, "data", "rshell", "public25", "rca", "tapes", "*.json")))
    ref_l, cand_l = P.open_loop(V635, {"losses": losses}, a.threads)["losses"], P.open_loop(cand, {"losses": losses}, a.threads)["losses"]
    rep["public_losses"] = P.compare(cand_l, ref_l)
    print(f"[fix {a.name}] public-25 losses: {rep['public_losses']['wins']} of {len(losses)} now won (ref {rep['public_losses']['ref_wins']}), "
          f"+{rep['public_losses']['better']}/-{rep['public_losses']['worse']}", flush=True)
    seeds = P.bank_seeds(a.seed_split, 3, a.seed_offset)
    rc, _ = P.closed_loop(V635, seeds, a.threads, opponents=P.FIT)
    cc, worlds = P.closed_loop(cand, seeds, a.threads, opponents=P.FIT)
    rep["lineage"] = P.compare(cc, rc)
    rep["lineage_by_opp"] = {o: P.compare({k: v for k, v in cc.items() if k[0] == o}, {k: v for k, v in rc.items() if k[0] == o}) for o in P.FIT}
    print(f"[fix {a.name}] lineage closed loop: +{rep['lineage']['better']}/-{rep['lineage']['worse']} p {rep['lineage']['p']}", flush=True)
    sets = P.tape_sets("B")
    ro, co = P.open_loop(V635, sets, a.threads), P.open_loop(cand, sets, a.threads)
    rep["real"] = {k: P.compare(co[k], ro[k]) for k in sets}
    # the real-ladder yardstick (python/ladder_study.py): our own ladder games vs the players who beat us
    ls = os.path.join(P.RL, "data", "ladder_study", "sets.json")
    if os.path.exists(ls):
        import hashlib
        hb = lambda f: int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 2 == 1  # noqa: E731  (split B = held out from CMA fitness)
        only_b = os.environ.get("KRL_LADDER_SPLIT", "all") == "B"
        lsets = {k: [f for f in (os.path.join(P.RL, "data", "ladder_study", "tapes", f"{r['sub']}__{r['episode']}_{r['seat']}.json") for r in v) if not only_b or hb(f)]
                 for k, v in json.load(open(ls)).items()}
        rl_, cl_ = P.open_loop(V635, lsets, a.threads), P.open_loop(cand, lsets, a.threads)
        rep["ladder_real"] = {k: P.compare(cl_[k], rl_[k]) for k in lsets}
        print(f"[fix {a.name}] REAL LADDER: " + " | ".join(f"{k} +{v['better']}/-{v['worse']} (wins {v.get('wins')} vs {v.get('ref_wins')})" for k, v in rep["ladder_real"].items()), flush=True)
    print(f"[fix {a.name}] real tapes: " + " | ".join(f"{k} +{v['better']}/-{v['worse']}" for k, v in rep["real"].items()), flush=True)
    if a.band:
        import test as T
        rep["band"] = {"cand": T.band(a.policy_iter, f"--shell {P.BIG1} --chain-off r127,sm,r95 {a.extra}", f"fix-{a.name}", a.threads),
                       "ref": T.band(a.policy_iter, f"--shell {P.BIG1} --chain-off r127,sm,r95 {a.ref_extra}".strip(), f"fix-ref-i{a.policy_iter}{'-x' if a.ref_extra else ''}", a.threads)}
        print(f"[fix {a.name}] band: cand {rep['band']['cand']['losses_below_2500']} <2500 {rep['band']['cand']['paired_vs_v631']}, "
              f"ref {rep['band']['ref']['losses_below_2500']} {rep['band']['ref']['paired_vs_v631']}", flush=True)
    out = os.path.join(P.RL, "data", "rshell", "fixes")
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, a.name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
