"""T2: closed-loop endgame labels vs our real releases (endg-label --branch).

    python python/v6312/t2_endg.py --out F [--split train] [--per-world 2] [--threads 4] [--limit N] [--no-branch] [--skip 0]"""
import argparse, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from args import stage_args, RL
OPPS = ["v63.6_rl", "v63.7_rl", "v63.8_rl", "v63.10_rl", "v63.11_rl"]

def strip_endg(a):
    return re.sub(r"--endg \S+", "", a).strip()

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--per-world", default="2")
ap.add_argument("--threads", default="4"); ap.add_argument("--limit", default=None); ap.add_argument("--no-branch", action="store_true")
ap.add_argument("--skip", default="0"); ap.add_argument("--props", default="configs/endg/proposals.json")
ap.add_argument("--bin", default=os.environ.get("KRL_BIN") and os.path.join(os.environ["KRL_BIN"], "endg-label") or os.path.join(RL, "target-v6312b", "release", "endg-label.exe"))
a = ap.parse_args()
base, ours = stage_args(os.path.join(RL, "data/builds/v63.11_rl/stage"))
opps = "|".join(f"{o}:{strip_endg(stage_args(os.path.join(RL, 'data/builds', o, 'stage'))[1])}" for o in OPPS)
cmd = [a.bin, "--base", base, "--a-args", strip_endg(ours), "--endg", os.path.join(RL, a.props), "--opps", opps,
       "--bank", os.path.join(RL, "data/worlds/w64_bank.json"), "--split", a.split, "--per-world", a.per_world, "--seed-skip", a.skip,
       "--all-opps", "--out", a.out, "--threads", a.threads] + ([] if a.no_branch else ["--branch"]) + (["--limit", a.limit] if a.limit else [])
print(" ".join(cmd[:3]), "...", flush=True)
sys.exit(subprocess.run(cmd, cwd=RL).returncode)
