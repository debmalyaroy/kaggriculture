"""FINAL VERDICT: the optimal-config bandit vs the ENTIRE public panel, over
multiple world-diverse seeds x both seats, on the Rust serve engine (parallel).

Config is win-neutral (verify_levers 2026-09-20), so this fixes ONE config
(sweep15 = all rails + un-starved sweep, the margin-best combo) and answers the
only remaining question at full breadth: how many of the public field do we beat,
by how much, and exactly which agents form the wall. Enumerates every loadable
agent under .local/crown_panel/refs/** + agents/pub_*.py, banded by rating.

Run: NN_WORLDS=8 NN_WORKERS=5 python -m kaggriculture.bandit.gate.panel_verdict
"""
import os, glob
import multiprocessing as mp
import numpy as np
from collections import defaultdict
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
import kaggriculture.measure.eval_harness as EH

BAND_ORDER = ["2700plus", "2500-2700", "2300-2500", "2100-2300", "lt2100", "agents"]


def full_panel():
    paths = glob.glob(os.path.join(ROOT, ".local", "crown_panel", "refs", "**", "*.py"), recursive=True)
    paths += glob.glob(os.path.join(ROOT, "agents", "pub_*.py"))
    out = []
    for p in sorted(set(paths)):
        band = os.path.basename(os.path.dirname(p))
        band = band if band in BAND_ORDER else "agents"
        nm = os.path.splitext(os.path.basename(p))[0][:30]
        try:
            EH._as_agent(p); out.append((band, nm, p))
        except Exception:
            pass
    return out


def _cell(task):
    stage, band, nm, ap, ws = task
    try:
        res = H.eval_vs(EH.bandit_binary_agent(stage), ap, ws)
        return (band, nm, res["win"], res["margin"], None)
    except Exception as e:
        return (band, nm, None, None, f"{type(e).__name__}: {str(e)[:50]}")


def main():
    n = int(os.environ.get("NN_WORLDS", "8"))
    workers = int(os.environ.get("NN_WORKERS", "5"))
    ws = H.world_seeds(n)
    panel = full_panel()
    # optimal config = sweep15 (win-neutral choice; margin-best)
    cfg, desc = VL.COMBOS["sweep15"]
    H.build_agent("pv_optimal", cfg)
    stage = os.path.join(H.SCRATCH, "pv_optimal_wingate")
    print(f"PANEL VERDICT  config=sweep15  agents={len(panel)}  worlds={len(ws)} x2 seats  workers={workers}", flush=True)
    print(f"({len(panel)*len(ws)*2} games)\n", flush=True)
    tasks = [(stage, band, nm, ap, ws) for band, nm, ap in panel]
    results = []
    done = 0
    with mp.Pool(workers, maxtasksperchild=4) as pool:
        for band, nm, win, margin, err in pool.imap_unordered(_cell, tasks):
            done += 1
            if err:
                print(f"  [{done}/{len(tasks)}] ! {nm}: {err}", flush=True)
                continue
            results.append((band, nm, win, margin))
            print(f"  [{done}/{len(tasks)}] {band:10s} {nm:32s} win={win:.2f} margin={margin:+9.0f}", flush=True)

    print("\n=== PER-AGENT (sorted by win, then margin) ===", flush=True)
    for band, nm, win, margin in sorted(results, key=lambda r: (-r[2], -r[3])):
        flag = "  <-- LOSE" if win < 0.5 else ""
        print(f"  {band:10s} {nm:32s} win={win:.2f} margin={margin:+9.0f}{flag}", flush=True)

    print("\n=== PER-BAND SUMMARY ===", flush=True)
    byband = defaultdict(list)
    for band, nm, win, margin in results:
        byband[band].append((win, margin))
    for band in BAND_ORDER:
        v = byband.get(band)
        if not v:
            continue
        mw = float(np.mean([w for w, _ in v]))
        mm = float(np.mean([m for _, m in v]))
        beat = sum(1 for w, _ in v if w > 0.5)
        print(f"  {band:10s} n={len(v):2d}  beat={beat}/{len(v)}  win={mw:.2f}  margin={mm:+9.0f}", flush=True)

    win_all = float(np.mean([w for _, _, w, _ in results]))
    beat_all = sum(1 for _, _, w, _ in results if w > 0.5)
    losers = [nm for _, nm, w, _ in results if w < 0.5]
    print(f"\n=== FINAL VERDICT (config=sweep15, {len(ws)} worlds x2 seats) ===", flush=True)
    print(f"  overall win={win_all:.3f}   beat {beat_all}/{len(results)} of the public field", flush=True)
    print(f"  LOSE to ({len(losers)}): {losers}", flush=True)


if __name__ == "__main__":
    main()
