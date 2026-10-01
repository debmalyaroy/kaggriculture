"""Which PPO checkpoint is best AS v63.5_rl PLAYS IT (big1 shell + chain-off r127,sm,r95), paired against i790.

    python python/rshell/ppo_scan.py --cands "B_i910=weights/ppo/<run>/snapshots/i000910.bin,C_best=weights/ppo2/<run>/best.bin" [--threads 12]

Per candidate (all rl3 + shield v1): public-25 loss tapes (open loop), the 64-world lineage panel on held-out seeds
(mirror excluded), the real-player band gate. Writes data/rshell/ppo_scan.json.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402
import test as T  # noqa: E402

TAIL = f"--shield {P.SHIELD} --shell {P.BIG1} --chain-off r127,sm,r95"


EXTRA = ""  # --extra: added to the reference AND every candidate (e.g. v63.7_rl's --rshell/--knob-over/--endg)


def agent(pol, prof=None):
    return f"--profiles {os.path.abspath(prof) if prof else P.RL3} --policy {os.path.abspath(pol)} {TAIL} {EXTRA}".strip()


def band(pol, name, threads, prof=None):
    import re
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(P.RL, "python", "band_gate.py"), "--cand", os.path.abspath(pol), "--cand-args",
                        f"--shell {P.BIG1} --chain-off r127,sm,r95 {EXTRA}".strip(), "--name", name, "--ref-profiles", P.V64, "--ref-profile", "35",
                        "--ref-group", "35,35,36", "--threads", str(threads)] + (["--profiles", os.path.abspath(prof)] if prof else []), cwd=P.RL, capture_output=True, text=True, env={**os.environ, "KRL_BIN": P.BIN})
    line = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", line)
    return {"losses_below_2500": int(m.group(1)) if m else None, "rate_2500plus": float(m.group(2)) if m else None,
            "paired_vs_v631": [int(m.group(3)), int(m.group(4))] if m else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--ref", default=os.path.join(P.RUN, "snapshots", "i000790.bin"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(P.RL, "data", "rshell", "ppo_scan.json"))
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    global EXTRA
    EXTRA = a.extra
    # NAME=PATH or NAME=PATH@PROFILES (PPO3 checkpoints play the 60-row rl5 table)
    cands = [(c.split("=", 1)[0], *(c.split("=", 1)[1].split("@") + [None])[:2]) for c in a.cands.split(",")]
    losses = sorted(glob.glob(os.path.join(P.RL, "data", "rshell", "public25", "rca", "tapes", "*.json")))
    seeds = P.bank_seeds("heldout", 3, 0)
    ref = agent(a.ref)
    ref_l = P.open_loop(ref, {"l": losses}, a.threads)["l"]
    ref_c, _ = P.closed_loop(ref, seeds, a.threads, opponents=P.FIT)
    rep = {"ref": a.ref, "ref_band": band(a.ref, "scan-ref-i790", a.threads), "cands": {}}
    print(f"[scan] ref i790 band {rep['ref_band']}", flush=True)
    for name, pol, prof in cands:
        if not os.path.exists(pol):
            print(f"[scan] {name}: missing {pol}", flush=True)
            continue
        cand = agent(pol, prof)
        l = P.compare(P.open_loop(cand, {"l": losses}, a.threads)["l"], ref_l)
        c, _ = P.closed_loop(cand, seeds, a.threads, opponents=P.FIT)
        cl = P.compare(c, ref_c)
        b = band(pol, f"scan-{name}", a.threads, prof)
        rep["cands"][name] = {"policy": pol, "public_losses": l, "lineage": cl, "band": b}
        print(f"[scan] {name}: public losses +{l['better']}/-{l['worse']} | lineage +{cl['better']}/-{cl['worse']} p {cl['p']} | "
              f"band {b['losses_below_2500']} <2500 {b['paired_vs_v631']} (ref {rep['ref_band']['losses_below_2500']})", flush=True)
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
