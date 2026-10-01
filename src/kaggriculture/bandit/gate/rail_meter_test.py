"""Plug in EVERYTHING we built and sweep configs across multiple seeds vs the
killers -- measuring gap + premium below-base% + premium $/u (production held
fixed; only the reactive layer changes).

Two families of dormant mechanism:
  HOLDING (defer/hold a tape SELL): supply_cap, supply_demand, mlp_sell,
    scarcity_hold -- proven to COLLAPSE the bank (holding our overproduction
    forfeits it; measured -44k..-138k on 2026-09-20). Kept here as controls.
  ADDITIVE / reorder (never reduce total volume): opp_front_run (the opp-dump
    NN -- sell premium BEFORE the opponent crashes the price), price_sell (pull
    premium sells forward to a better price), terminal_rescue (progressive
    endgame liquidation), endgame_liquidate (dead-stock). These respect the
    volume invariant and are the untested candidates.

Parallel across configs (each worker owns one config, runs its games serially,
cleans up its own kagg). Run:
  NN_WORKERS=5 NN_WORLDS=4 python -m kaggriculture.bandit.gate.rail_meter_test
"""
import os, argparse, subprocess
import multiprocessing as mp
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
from kaggriculture.bandit.gate import loss_forensics as LF
import kaggriculture.measure.eval_harness as EH

KILLERS = [k for k in LF.KILLERS if k[0] in ("tschinkel", "tetsutani", "k0006")]


def sw15():
    return VL.cfg(tables={"sweep": 15, "esc_sweep": 10})


def add(c, rail):
    c["guardrails"].append(rail); return c


def herd(**caps):
    r = {"name": "herd_gate", "on": True}
    r.update(caps)
    return add(sw15(), r)


def variants():
    """HERD_GATE (operator design): keep the blind early herd (also yields
    fertilizer), but from D6 -- once shops are known -- STOP GROWING a premium the
    world can't absorb: suppress NEW sheep when no YARN, NEW cow when no milk shop.
    Only fires in incompatible worlds; pre-D6 animals stay. Egg (dump-proof) is the
    intended substitute (a branch, next step)."""
    v = {}
    v["baseline"] = sw15()
    v["stop_sheep_noyarn"] = herd(from_step=146, stop_sheep_noyarn=1)
    v["stop_cow_nomilk"] = herd(from_step=146, stop_cow_nomilk=1)
    v["stop_both"] = herd(from_step=146, stop_sheep_noyarn=1, stop_cow_nomilk=1)
    return v


def _run_config(task):
    name, stage, ws = task
    LF.stage_g = stage
    ours = EH.bandit_binary_agent(stage)
    gaps = []; oacc = LF.newacc()
    try:
        for kn, kp in KILLERS:
            for wn, seed in ws:
                recs, ob, us, them = LF.capture_game(ours, kp, seed, 0)
                gaps.append(us - them)
                LF.acc_add(oacc, LF.econ(recs, "our"))
    finally:
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Get-Process kagg -ErrorAction SilentlyContinue | "
                            f"Where-Object {{ $_.Path -like '*{name}_wingate*' }} | Stop-Process -Force"],
                           capture_output=True)
        except Exception:
            pass
    pf = sum(oacc["filled"][p] for p in LF.PREM)
    pb = sum(oacc["below"][p] for p in LF.PREM)
    pr = sum(oacc["realized"][p] for p in LF.PREM)
    pe = sum(oacc["filled_end"][p] for p in LF.PREM)
    return (name, float(np.mean(gaps)), 100 * pb / pf if pf else 0,
            pr / pf if pf else 0, pf / oacc["n"] if oacc["n"] else 0,
            100 * pe / pf if pf else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=None); a = ap.parse_args()
    n = a.seeds or int(os.environ.get("NN_WORLDS", "4"))
    workers = int(os.environ.get("NN_WORKERS", "5"))
    ws = H.world_seeds(n)
    vs = variants()
    print(f"PLUG-IN SWEEP  configs={len(vs)}  killers={[k for k,_ in KILLERS]}  worlds={len(ws)}  workers={workers}", flush=True)
    tasks = []
    for name, cfg in vs.items():
        H.build_agent(f"rm_{name}", cfg)
        tasks.append((f"rm_{name}", os.path.join(H.SCRATCH, f"rm_{name}_wingate"), ws))
    rows = []
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for name, mg, bp, dpu, pf, ep in pool.imap_unordered(_run_config, tasks):
            nm = name[3:]
            rows.append((nm, mg, bp, dpu, pf, ep))
            print(f"  {nm:18s} gap={mg:+9.0f}  prem_below%={bp:4.0f}  prem_$/u={dpu:6.1f}  prem/game={pf:5.0f}  endgame%={ep:4.0f}", flush=True)
    print("\n=== RANKED by mean gap (baseline is the bar to beat) ===", flush=True)
    for nm, mg, bp, dpu, pf, ep in sorted(rows, key=lambda r: -r[1]):
        tag = "  <-- baseline" if nm == "baseline" else ""
        print(f"  {nm:18s} gap={mg:+9.0f}  below%={bp:4.0f}  $/u={dpu:6.1f}  end%={ep:4.0f}{tag}", flush=True)


if __name__ == "__main__":
    main()
