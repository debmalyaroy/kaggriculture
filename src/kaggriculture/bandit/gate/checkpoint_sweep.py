"""Sweep the DISPATCH checkpoints (D6/D12/D15/D21/D27) -- the economy lever.

Each checkpoint is where the router re-selects the route/economy, so the checkpoint
SET changes what the agent produces (unlike the inert sell rails). Config-only, no
rebuild. D6 is droppable via d6_optional (build_rust_bandit._normalise_cfg honours it).

Run: NN_WORLDS=8 python -m kaggriculture.bandit.gate.checkpoint_sweep
"""
import os
import numpy as np
from kaggriculture.bandit.gate import harness as H

ALL = [[146, 6], [290, 12], [362, 15], [506, 21], [650, 27]]


def cfg_cp(cps, d6_opt=False):
    c = H.base_config(); c["checkpoints"] = [list(x) for x in cps]
    if d6_opt:
        c["d6_optional"] = True
    return c


CANDS = [
    ("all5_baseline", ALL, False),
    ("D6only",        [[146, 6]], False),
    ("D6_12",         [[146, 6], [290, 12]], False),
    ("D6_12_15",      [[146, 6], [290, 12], [362, 15]], False),
    ("D6_12_15_21",   [[146, 6], [290, 12], [362, 15], [506, 21]], False),
    ("noD6_12to27",   [[290, 12], [362, 15], [506, 21], [650, 27]], True),
    ("none_puretape", [], True),
    ("no_D12",        [[146, 6], [362, 15], [506, 21], [650, 27]], False),
    ("no_D27",        [[146, 6], [290, 12], [362, 15], [506, 21]], False),
]


def main():
    n = int(os.environ.get("NN_WORLDS", "8"))
    ws = H.world_seeds(n)
    KILL = H.killers()
    # default to tschinkel-only for speed unless NN_ALLKILL=1
    refs = KILL if os.environ.get("NN_ALLKILL") else [KILL[0]]
    print(f"CHECKPOINT SWEEP  worlds={len(ws)} (x2 seats)  refs={[k for k,_ in refs]}\n", flush=True)
    rows = []
    for name, cps, d6opt in CANDS:
        ag = H.build_agent(f"cp_{name}", cfg_cp(cps, d6opt))
        per = {kn: H.eval_vs(ag, kp, ws) for kn, kp in refs}
        mm = np.mean([per[kn]["margin"] for kn, _ in refs])
        mw = np.mean([per[kn]["win"] for kn, _ in refs])
        rows.append((name, mw, mm))
        print(f"  {name:16s} win={mw:.2f} margin={mm:+8.0f}", flush=True)
    rows.sort(key=lambda r: -r[2])
    print("\n=== RANKED by margin ===", flush=True)
    for name, mw, mm in rows:
        print(f"  {name:16s} margin={mm:+8.0f}  win={mw:.2f}", flush=True)


if __name__ == "__main__":
    main()
