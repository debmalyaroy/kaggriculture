"""Pick the PPO checkpoint that matters: real-loss conversion + band gate + held-out ladder, each with the shell (queue Q65).

    python python/top50/ckpt_scan.py --iters 630,670,700,720,740,750,770 [--shell weights/shell/r6/shell.json] [--threads 24]

For each checkpoint (weights/ppo/<run>/snapshots/iNNNNNN.bin) + shield v1 + the shell, vs v63.1_rl:
  losses   how many of our real ladder losses (data/tapes/losses) it wins
  band     the 991-tape band gate (losses below 2500, 2500+ win rate, paired)
  ladder   the held-out ladder half, paired
Ranks by (losses below 2500 on the band, then losses converted). Writes data/gates/ckpt_scan.json.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from band_gate import SHIELD  # noqa: E402
from copy_screen import RL, paired, play  # noqa: E402

RUN = os.path.join(RL, "weights", "ppo", "ppo-gru64-f2-p36-from-init-20260926T0344Z")
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", required=True)
    ap.add_argument("--shell", default=os.path.join(RL, "weights", "shell", "r6", "shell.json"))
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    shell = os.path.abspath(a.shell)
    ref_args = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
    losses = sorted(glob.glob(os.path.join(RL, "data", "tapes", "losses", "*.json")))
    vf = sorted(glob.glob(os.path.join(RL, "data", "tapes", "ladder_val", "**", "*.json"), recursive=True))
    ref_l, ref_v = play(ref_args, losses, a.threads), play(ref_args, vf, a.threads)
    rows = []
    for it in [int(x) for x in a.iters.split(",")]:
        pol = os.path.join(RUN, "snapshots", f"i{it:06d}.bin")
        if not os.path.exists(pol):
            continue
        cand = ["--profiles", RL3, "--guarded", "--policy", pol, "--shield", SHIELD, "--shell", shell]
        cl, cv = play(cand, losses, a.threads), play(cand, vf, a.threads)
        r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand", pol, "--cand-args", f"--shell {shell}",
                            "--name", f"scan-i{it}", "--ref-profiles", V64, "--ref-profile", "35", "--ref-group", "35,35,36",
                            "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", r.stdout)
        row = {"iter": it, "loss_wins": int(sum(v == 1 for v in cl.values())), "losses_paired": paired(cl, ref_l),
               "ladder": paired(cv, ref_v), "band_losses_below_2500": int(m.group(1)) if m else None,
               "band_2500plus": float(m.group(2)) if m else None, "band_paired": [int(m.group(3)), int(m.group(4))] if m else None}
        rows.append(row)
        print(f"[scan] i{it}+shell: real losses won {row['loss_wins']}/{len(losses)} | band {row['band_paired']}, <2500 losses "
              f"{row['band_losses_below_2500']}, 2500+ {row['band_2500plus']} | ladder {row['ladder']['wins']} vs {row['ladder']['ref_wins']}", flush=True)
    rows.sort(key=lambda r: (r["band_losses_below_2500"] or 999, -r["loss_wins"]))
    json.dump({"shell": shell, "ref_loss_wins": int(sum(v == 1 for v in ref_l.values())), "rows": rows},
              open(os.path.join(RL, "data", "gates", "ckpt_scan.json"), "w"), indent=1)
    print("[scan] best:", rows[0] if rows else None)


if __name__ == "__main__":
    main()
