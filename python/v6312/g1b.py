"""G1b: closed-loop GT payoffs vs our real releases (gt-label).

    python python/v6312/g1b.py --out F [--split train] [--per-world 1] [--skip 0] [--threads 4] [--limit N]"""
import argparse, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from args import stage_args, RL
OPPS = ["v63.6_rl", "v63.7_rl", "v63.8_rl", "v63.10_rl", "v63.11_rl"]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--per-world", default="1")
ap.add_argument("--skip", default="0"); ap.add_argument("--threads", default="4"); ap.add_argument("--limit", default=None)
ap.add_argument("--checkpoints", default="360,552,648,684"); ap.add_argument("--min", default="15")
ap.add_argument("--bin", default=os.path.join(os.environ["KRL_BIN"], "gt-label") if os.environ.get("KRL_BIN") else os.path.join(RL, "target-v6312b", "release", "gt-label.exe"))
a = ap.parse_args()
base, ours = stage_args(os.path.join(RL, "data/builds/v63.11_rl/stage"))
opps = "|".join(f"{o}:{stage_args(os.path.join(RL, 'data/builds', o, 'stage'))[1]}" for o in OPPS)
cmd = [a.bin, "--base", base, "--a-args", ours, "--opps", opps, "--bank", os.path.join(RL, "data/worlds/w64_bank.json"),
       "--split", a.split, "--per-world", a.per_world, "--seed-skip", a.skip, "--checkpoints", a.checkpoints, "--min", a.min,
       "--out", a.out, "--threads", a.threads] + (["--limit", a.limit] if a.limit else [])
sys.exit(subprocess.run(cmd, cwd=RL).returncode)
