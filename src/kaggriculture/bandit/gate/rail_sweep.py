"""General RAIL KNOB SWEEP -- world-gated, margin-aware, config-only (no rebuild).

Define a grid of candidate guardrail stacks (each with its knobs); this evaluates
every one across N distinct worlds vs the 3 killers, ranks by mean margin, flags
collapses. Build the isolated binary ONCE; tune here freely.

Run: NN_WORLDS=10 python -m kaggriculture.bandit.gate.rail_sweep
"""
import os, json
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H

# THE SWEEP: name -> extra guardrails. Edit freely; config-only, no rebuild.
SWEEP = [
    ("baseline",            []),
    ("scap_f0.9",           [H.supply_cap(0.9), H.budget()]),
    ("scap_f0.75",          [H.supply_cap(0.75), H.budget()]),
    ("scap_f0.6",           [H.supply_cap(0.6), H.budget()]),
    ("scap_f0.6_bound",     [H.supply_cap(0.6, bound_future=1), H.budget()]),
    ("scap_f0.5_bound",     [H.supply_cap(0.5, bound_future=1), H.budget()]),
    ("scap_f0.6_cap400",    [H.supply_cap(0.6, carry_max=400, bound_future=1), H.budget()]),
    ("price+scap0.6_bound", [H.price_sell(), H.supply_cap(0.6, bound_future=1), H.budget()]),
    ("price+scap0.6+oppfr", [H.price_sell(), H.supply_cap(0.6, bound_future=1), H.budget(), H.opp_fr()]),
]


def main():
    n_worlds = int(os.environ.get("NN_WORLDS", "10"))
    ws = H.world_seeds(n_worlds)
    KILL = H.killers()
    print(f"RAIL SWEEP  worlds={len(ws)} (x2 seats)  killers={[n for n,_ in KILL]}", flush=True)
    print(f"  worlds: {[w for w,_ in ws]}\n", flush=True)
    rows = []
    for name, extra in SWEEP:
        ag = H.build_agent(f"sw_{name}", H.cfg_with(extra))
        per = {kn: H.eval_vs(ag, kp, ws) for kn, kp in KILL}
        mw = np.mean([per[kn]["win"] for kn, _ in KILL])
        mm = np.mean([per[kn]["margin"] for kn, _ in KILL])
        worst = min(per[kn]["margin"] for kn, _ in KILL)
        rows.append((name, mw, mm, worst))
        print(f"  {name:22s} meanwin={mw:.2f} mean_marg={mm:+8.0f}  worst={worst:+8.0f}  "
              + "  ".join(f"{kn}={per[kn]['margin']:+7.0f}" for kn, _ in KILL)
              + ("  COLLAPSE" if worst < -30000 else ""), flush=True)
    rows.sort(key=lambda r: -r[2])
    print("\n=== RANKED by mean margin ===", flush=True)
    for name, mw, mm, worst in rows:
        print(f"  {name:22s} mean_marg={mm:+8.0f}  meanwin={mw:.2f}  worst={worst:+8.0f}", flush=True)
    json.dump([{"name": n, "meanwin": w, "mean_marg": m, "worst": ws_} for n, w, m, ws_ in rows],
              open(os.path.join(H.SCRATCH, "rail_sweep.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(H.SCRATCH, 'rail_sweep.json')}", flush=True)


if __name__ == "__main__":
    main()
