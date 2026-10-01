"""One DAgger round for reactive shell v2: label the states the CANDIDATE visits, retrain, re-tune, re-test.

    python python/rshell/dagger.py --round 1 --prev data/rshell/cma [--threads 24] [--gens 15]

  1. closed-loop labels ON POLICY: branch2 --selfplay with the candidate (shell ON, CMA knobs, chain-off) as seat A,
     the 7 lineage opponents, fresh bank seeds (train split, rotation 80 + 8*round..)
  2. open-loop labels on the games the candidate LOSES or wins by < $3k among the training ladder half and our
     real losses split A (tapeplay picks them; branch2 labels the candidate as the shadow)
  3. retrain on every label file so far -> weights/rshell/v2rN
  4. CMA-ES for --gens generations starting from the previous best settings -> data/rshell/cmaN
  5. held-out test -> data/rshell/test_dagger_rN.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402

PY = sys.executable
B2 = os.path.join(P.BIN, "branch2" + P.EXE)


def sh(cmd):
    print("[dagger] $", " ".join(cmd)[:400], flush=True)
    subprocess.run(cmd, cwd=P.RL, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--prev", required=True, help="folder with best_rshell.json, best_knobs.json, chain_off.txt")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15, help="0 = no CMA: test the retrained net with the previous settings vs the shipped reference (fix_test)")
    ap.add_argument("--it", type=int, default=790, help="run B checkpoint of the candidate (910 = v63.6/v63.7_rl)")
    ap.add_argument("--rs", default=None, help="shell config (default PREV/best_rshell.json)")
    ap.add_argument("--kn", default=None, help="knob overrides (default PREV/best_knobs.json)")
    ap.add_argument("--endg", default=None, help="endgame model json the candidate ships with (v63.7_rl)")
    a = ap.parse_args()
    r = a.round
    lab = os.path.join(P.RL, "data", "rshell", "labels")
    os.makedirs(lab, exist_ok=True)
    rs, kn = a.rs or os.path.join(a.prev, "best_rshell.json"), a.kn or os.path.join(a.prev, "best_knobs.json")
    off = open(os.path.join(a.prev, "chain_off.txt")).read().strip() if os.path.exists(os.path.join(a.prev, "chain_off.txt")) else ""
    eg = f" --endg {a.endg}" if a.endg else ""
    cand = P.ppo_args(a.it, shell=P.BIG1, extra=f"--rshell {rs} --knob-over {kn}" + (f" --chain-off {off}" if off else "") + eg)
    opps = "|".join(f"{n}:{v}" for n, v in P.OPPONENTS.items())
    cl_out = os.path.join(lab, f"cl_dagger{r}.tsv")
    if not (os.path.exists(cl_out) and os.path.getsize(cl_out) > 0):
      sh([B2, "--selfplay", "--a-args", cand, "--opps", opps, "--bank", P.BANK, "--split", "train", "--per-world", "8",
        "--seed-skip", str(80 + 8 * r), "--k", "24", "--threads", str(a.threads), "--out", os.path.join(lab, f"cl_dagger{r}.tsv")])
    # hard open-loop games: the candidate's losses and narrow wins
    hard = os.path.join(P.RL, "data", "rshell", f"hard_r{r}")
    shutil.rmtree(hard, ignore_errors=True)
    os.makedirs(hard)
    sets = P.tape_sets("A")
    files = sets["ladder"] + sets["losses"]
    res = P.tape_play(cand.split() + ["--guarded"], files, a.threads)
    for f in files:
        k = os.path.basename(f)[:-5]
        dst = os.path.join(hard, os.path.basename(f))
        if res.get(k, 1.0) < 1.0 and not os.path.lexists(dst):
            os.symlink(os.path.abspath(f), dst)
    print(f"[dagger] {len(os.listdir(hard))} hard tapes (candidate did not win)", flush=True)
    if os.listdir(hard):
        sh([B2, "--tapes", hard, "--a-args", cand, "--k", "24", "--threads", str(a.threads), "--out", os.path.join(lab, f"ol_dagger{r}.tsv")])
    out = os.path.join(P.RL, "weights", "rshell", f"v2r{r}")
    sh([PY, os.path.join(P.RL, "python", "rshell", "train.py"), "--labels", os.path.join(lab, "*.tsv"), "--out", out])
    # carry the tuned settings over to the new network, then re-tune from there
    base = json.load(open(rs))
    new = json.load(open(os.path.join(out, "rshell.json")))
    for k, v in base.items():
        if k not in ("net_file", "lineage_file", "trained", "cma", "name", "note"):
            new[k] = v
    json.dump(new, open(os.path.join(out, "rshell.json"), "w"), indent=1)
    if a.gens == 0:
        # paired vs the shipped reference: same policy/knobs/chain-off/endg, only the shell network differs
        ref_x = f"--rshell {rs} --knob-over {kn}{eg}"
        new_x = f"--rshell {os.path.join(out, 'rshell.json')} --knob-over {kn}{eg}"
        sh([PY, os.path.join(P.RL, "python", "rshell", "fix_test.py"), "--policy-iter", str(a.it), "--threads", str(a.threads), "--band",
            "--name", f"dagger_r{r}", "--ref-extra", ref_x, "--extra", new_x])
        return
    cma = os.path.join(P.RL, "data", "rshell", f"cma{r}")
    shutil.copy(os.path.join(a.prev, "chain_off.txt"), os.path.join(out, "chain_off.txt")) if off else None
    sh([PY, os.path.join(P.RL, "python", "rshell", "cmaes.py"), "--shell", os.path.join(out, "rshell.json"), "--gens", str(a.gens), "--sigma", "0.1",
        "--threads", str(a.threads), "--out", cma] + (["--chain-off-file", os.path.join(out, "chain_off.txt")] if off else []))
    sh([PY, os.path.join(P.RL, "python", "rshell", "test.py"), "--rshell", os.path.join(cma, "best_rshell.json"), "--knobs", os.path.join(cma, "best_knobs.json"),
        "--chain-off-file", os.path.join(cma, "chain_off.txt"), "--name", f"dagger_r{r}", "--threads", str(a.threads)])


if __name__ == "__main__":
    main()
