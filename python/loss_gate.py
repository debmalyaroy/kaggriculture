"""Loss-conversion gate: how many of our real ladder losses a candidate turns into wins (queue Q62, every 2 h).

    python python/loss_gate.py [--threads 16]

Plays every tape in data/tapes/losses (python/loss_tapes.py: games our submissions lost or drew on the ladder)
against the same recorded opponent (guarded), for:
  v63.1_rl (reference), v63.1_rl + shell r6, the current PPO best (+ shell r6), and the v63.4_rl build (PPO i630 + shell)
and reports wins out of the losses, paired vs v63.1_rl. The target: an RL model should win these.
Writes data/gates/loss_gate.json (history appended to data/gates/loss_gate.jsonl).
"""
import argparse
import datetime as D
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from band_gate import SHIELD  # noqa: E402
from copy_screen import RL, paired, play  # noqa: E402

V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")
SHELL = os.path.join(RL, "weights", "shell", "r6", "shell.json")
BEST = os.path.join(RL, "weights", "ppo", "ppo-gru64-f2-p36-from-init-20260926T0344Z", "best.bin")
I630 = os.path.join(RL, "weights", "release_i630.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(RL, "data", "tapes", "losses", "*.json")))
    ref_args = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
    arms = {
        "v63.1_rl": ref_args,
        "v63.1_rl+shell_r6 (v63.3_rl)": ref_args + ["--shell", SHELL],
        "ppo_best": ["--profiles", RL3, "--guarded", "--policy", BEST, "--shield", SHIELD],
        "ppo_best+shell_r6": ["--profiles", RL3, "--guarded", "--policy", BEST, "--shield", SHIELD, "--shell", SHELL],
    }
    if os.path.exists(I630):
        arms["ppo_i630+shell_r6 (v63.4_rl)"] = ["--profiles", RL3, "--guarded", "--policy", I630, "--shield", SHIELD, "--shell", SHELL]
    res = {k: play(v, files, a.threads) for k, v in arms.items()}
    ref = res["v63.1_rl"]
    rep = {"t": D.datetime.utcnow().isoformat() + "Z", "losses": len(files), "arms": {}}
    for k, r in res.items():
        rep["arms"][k] = {"wins": int(sum(v == 1 for v in r.values())), "draws": int(sum(v == 0.5 for v in r.values())),
                          "vs_v63.1_rl": paired(r, ref)}
        print(f"[lossgate] {k:32s}: wins {rep['arms'][k]['wins']:3d} of {len(files)} real losses (paired vs v63.1_rl "
              f"+{rep['arms'][k]['vs_v63.1_rl']['better']}/-{rep['arms'][k]['vs_v63.1_rl']['worse']})", flush=True)
    out = os.path.join(RL, "data", "gates")
    json.dump(rep, open(os.path.join(out, "loss_gate.json"), "w"), indent=1)
    with open(os.path.join(out, "loss_gate.jsonl"), "a") as fh:
        fh.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()
