"""World-gated candidate GATE: compare configs vs the 3 killers across distinct
worlds, margin-aware, with a per-world breakdown for the strongest config.

Edit CANDS (name -> extra guardrails). Run:
    NN_WORLDS=12 python -m kaggriculture.bandit.gate.world_gate
"""
import os, json
import numpy as np
from kaggriculture.bandit.gate import harness as H

# candidate configs to gate (name -> extra guardrails)
CANDS = [
    ("A_baseline",   []),
    ("B_scap_f0.6",  [H.supply_cap(0.6, bound_future=1), H.budget()]),  # collapsing config: WHERE?
]


def main():
    n_worlds = int(os.environ.get("NN_WORLDS", "12"))
    ws = H.world_seeds(n_worlds); KILL = H.killers()
    print(f"WORLD GATE  worlds={len(ws)} (x2 seats)  killers={[n for n,_ in KILL]}\n", flush=True)
    results = {}
    for name, extra in CANDS:
        ag = H.build_agent(f"wg_{name}", H.cfg_with(extra))
        results[name] = {kn: H.eval_vs(ag, kp, ws) for kn, kp in KILL}
        r = results[name]
        print(f"  {name:14s} " + "  ".join(
            f"{kn}:win={r[kn]['win']:.2f} marg={r[kn]['margin']:+7.0f}" for kn, _ in KILL)
            + f"   meanwin={np.mean([r[kn]['win'] for kn,_ in KILL]):.2f}"
            + f"  mean_marg={np.mean([r[kn]['margin'] for kn,_ in KILL]):+7.0f}", flush=True)

    base = results[CANDS[0][0]]
    best = max(CANDS[1:], key=lambda c: np.mean([results[c[0]][kn]['win'] for kn, _ in KILL]))[0] \
        if len(CANDS) > 1 else CANDS[0][0]
    print(f"\n=== PER-WORLD ({best}, mean over killers; baseline in parens) ===", flush=True)
    won = 0
    for wname, _ in ws:
        mnn = np.mean([results[best][kn]['per_world'][wname][1] for kn, _ in KILL])
        wnn = np.mean([results[best][kn]['per_world'][wname][0] for kn, _ in KILL])
        mb = np.mean([base[kn]['per_world'][wname][1] for kn, _ in KILL])
        won += (mnn >= mb)
        print(f"  {wname:28s} win={wnn:.2f} marg={mnn:+7.0f} (base {mb:+7.0f}) d={mnn-mb:+7.0f}", flush=True)
    print(f"\nworlds where {best} >= baseline margin: {won}/{len(ws)}", flush=True)
    json.dump({"worlds": [w for w, _ in ws],
               "results": {n: {kn: {"win": results[n][kn]["win"], "margin": results[n][kn]["margin"]}
                               for kn, _ in KILL} for n in results}},
              open(os.path.join(H.SCRATCH, "world_gate.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(H.SCRATCH, 'world_gate.json')}", flush=True)


if __name__ == "__main__":
    main()
