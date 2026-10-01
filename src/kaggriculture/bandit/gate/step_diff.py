"""STEP-BY-STEP economy comparison: our decisions vs a good agent's, day by day.

The forensic proved the gap is the base-tape ECONOMY STRUCTURE (we commit labor
to dairy/wool that gluts; winners commit it elsewhere), and that no rail/NN/cut
fixes it. To rebuild the economy the way the good public agents did -- step by
step, verified -- we first need a turn-by-turn map of WHERE and WHEN their
economy diverges from ours. This plays us vs a good agent and prints, per DAY,
each side's economy actions (plant/animal/coop/pasture/land/hire/buy_seed) so the
divergence is visible decision by decision, plus the running bank gap.

Run: python -m kaggriculture.bandit.gate.step_diff --killer tschinkel
"""
import os, argparse
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
from kaggriculture.bandit.gate import loss_forensics as LF
import kaggriculture.measure.eval_harness as EH


def day_actions(recs, who, day):
    ak = f"{who}_act"
    plant = defaultdict(int); animal = defaultdict(int); build = defaultdict(int)
    hire = 0; land = 0; seed = defaultdict(int)
    for r in recs:
        if r["step"] // 24 != day:
            continue
        a = r[ak] or {}
        for h in (a.get("hands") or []):
            if not h:
                continue
            if h[0] == "PLANT" and len(h) >= 2: plant[h[1]] += 1
            elif h[0] == "BUILD_COOP": build["COOP"] += 1
            elif h[0] == "BUILD_PASTURE": build["PASTURE"] += 1
        f = a.get("farmer") or []
        if f and f[0] == "PLANT" and len(f) >= 2: plant[f[1]] += 1
        for o in (a.get("market") or []):
            if not o:
                continue
            if o[0] == "BUY_ANIMAL" and len(o) >= 2:
                try: animal[o[1]] += int(o[2]) if len(o) > 2 else 1
                except (ValueError, TypeError): animal[o[1]] += 1
            elif o[0] == "HIRE": hire += 1
            elif o[0] == "BUY_LAND": land += 1
            elif o[0] == "BUY_SEED" and len(o) >= 2:
                try: seed[o[1]] += int(o[2]) if len(o) > 2 else 1
                except (ValueError, TypeError): seed[o[1]] += 1
    return plant, animal, build, hire, land, seed


def fmt(d):
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])) if d else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--killer", default="tschinkel")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--days", type=int, default=14, help="how many early days to print in detail")
    a = ap.parse_args()
    ws = H.world_seeds(2)
    seed = a.seed or ws[0][1]
    kp = dict((k, p) for k, p in LF.KILLERS)[a.killer]
    cfg, _ = VL.COMBOS["sweep15"]
    H.build_agent("sd_opt", cfg)
    LF.stage_g = os.path.join(H.SCRATCH, "sd_opt_wingate")
    ours = EH.bandit_binary_agent(LF.stage_g)
    recs, ob, us, them = LF.capture_game(ours, kp, seed, 0)
    print(f"STEP-BY-STEP ECONOMY  us vs {a.killer}  seed={seed}   final us=${us:,.0f} vs ${them:,.0f} gap={us-them:+,.0f}\n", flush=True)

    # running bank gap by day
    gap_at = {}
    for r in recs:
        gap_at[r["step"] // 24] = r["our_money"] - r["opp_money"]

    print("Legend: each cell = that day's NEW actions. [O]=ours [K]=killer\n", flush=True)
    for day in range(min(a.days, 30)):
        op, oa, ob_, oh, ol, os_ = day_actions(recs, "our", day)
        kp_, ka, kb, kh, kl, ks = day_actions(recs, "opp", day)
        if not any([op, oa, ob_, oh, ol, kp_, ka, kb, kh, kl]):
            continue
        g = gap_at.get(day, 0)
        print(f"--- DAY {day:2d}   bank gap={g:+.0f} ---", flush=True)
        print(f"  [O] plant[{fmt(op)}] animal[{fmt(oa)}] build[{fmt(ob_)}] hire={oh} land={ol} seed[{fmt(os_)}]", flush=True)
        print(f"  [K] plant[{fmt(kp_)}] animal[{fmt(ka)}] build[{fmt(kb)}] hire={kh} land={kl} seed[{fmt(ks)}]", flush=True)

    # cumulative animal + coop/pasture buildout (the labor-allocation signal)
    print("\n=== cumulative animals + structures by end ===", flush=True)
    for who, tag in (("our", "OURS"), ("opp", a.killer)):
        A = defaultdict(int); B = defaultdict(int)
        for day in range(30):
            _, an, bd, _, _, _ = day_actions(recs, who, day)
            for k, v in an.items(): A[k] += v
            for k, v in bd.items(): B[k] += v
        print(f"  {tag:12s} animals[{fmt(A)}]  structures[{fmt(B)}]", flush=True)

    try:
        import subprocess as _sp
        _sp.run(["powershell", "-NoProfile", "-Command",
                 "Get-Process kagg -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.Path -like '*sd_opt_wingate*' } | Stop-Process -Force"],
                capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
