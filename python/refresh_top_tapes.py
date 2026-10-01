"""Refresh the late top-player tapes in the PPO training pool after each delta pull (queue job Q22).

    python python/refresh_top_tapes.py [--since 2026-09-18] [--min-rating 2700] [--threads 16]

1. band_tapes --top: every game played at --min-rating+ since --since (gate-set games excluded) -> a staging dir.
2. tapeplay --verify: keep only tapes that reproduce the replay's banks exactly.
3. Score v63 (rl3 profile 35) on them; losses and wins by < $3,000 are hard -> data/tapes/hard_late, linked twice into
   train/hard_x2 (`__l1`/`__l2`), so they weigh 3x like the other hard tapes. data/tapes/hard (the validation set) is
   never touched.
4. Swap the staging dir in as data/tapes/train/top_late. PPO launches a fresh ppo-rollout every iteration, so the
   next iteration trains on the new pool without a restart.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
TAPES = os.path.join(RL, "data", "tapes")
STAGE = os.path.join(TAPES, "stage_top_late")
OUT = os.path.join(TAPES, "train", "top_late")
HARD = os.path.join(TAPES, "hard_late")
HARD_X2 = os.path.join(TAPES, "train", "hard_x2")
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")


def play(args, files, threads):
    """tapeplay over `files` in shards; returns {tape id: [replay_us, replay_them, sim_us, sim_them]}."""
    shards = [files[i::threads] for i in range(threads) if files[i::threads]]

    def one(shard):
        d = tempfile.mkdtemp(prefix="toplate-")
        try:
            for f in shard:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, *args], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"tapeplay rc {r.returncode}: {r.stderr[-500:]}")
            return {ln.split("\t")[0]: [float(x) for x in ln.split("\t")[2:6]] for ln in r.stdout.splitlines() if ln.count("\t") >= 5}
        finally:
            shutil.rmtree(d, ignore_errors=True)

    out = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for res in ex.map(one, shards):
            out.update(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-18")
    ap.add_argument("--min-rating", type=float, default=2700)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    shutil.rmtree(STAGE, ignore_errors=True)
    subprocess.run([sys.executable, os.path.join(RL, "python", "band_tapes.py"), "--top", "--min-rating", str(a.min_rating),
                    "--since", a.since, "--out", STAGE], cwd=RL, check=True)
    files = sorted(glob.glob(os.path.join(STAGE, "**", "*.json"), recursive=True))
    if not files:
        print("[top_late] no new top games; pool unchanged")
        return

    ver = play(["--verify"], files, a.threads)
    exact = {k for k, v in ver.items() if v[0] == v[2] and v[1] == v[3]}
    for f in files:
        if os.path.basename(f)[:-5] not in exact:
            os.remove(f)
    files = [f for f in files if os.path.basename(f)[:-5] in exact]
    print(f"[top_late] {len(exact)} of {len(ver)} tapes replay exactly ({len(ver) - len(exact)} dropped)")

    ref = play(["--profiles", RL3, "--guarded", "--pa", "35"], files, a.threads)
    hard = {k for k, v in ref.items() if v[2] - v[3] < 3000}
    losses = sum(1 for v in ref.values() if v[2] < v[3])
    print(f"[top_late] v63 on them: {len(ref) - losses} wins, {losses} losses; hard (loss or win by < $3,000): {len(hard)}")

    # install: top_late swap, then the hard copies and their 2x links
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.move(STAGE, OUT)
    shutil.rmtree(HARD, ignore_errors=True)
    os.makedirs(HARD)
    for f in glob.glob(os.path.join(OUT, "**", "*.json"), recursive=True):
        if os.path.basename(f)[:-5] in hard:
            shutil.copy(f, HARD)
    os.makedirs(HARD_X2, exist_ok=True)
    for f in glob.glob(os.path.join(HARD_X2, "*__l[12].json")):
        os.remove(f)
    for f in glob.glob(os.path.join(HARD, "*.json")):
        stem = os.path.basename(f)[:-5]
        for c in ("l1", "l2"):
            os.symlink(os.path.abspath(f), os.path.join(HARD_X2, f"{stem}__{c}.json"))
    n_pool = len(glob.glob(os.path.join(TAPES, "train", "**", "*.json"), recursive=True))
    print(f"[top_late] installed {len(files)} tapes -> {OUT}; {len(hard)} hard x2 -> {HARD_X2}; training pool now {n_pool} entries")


if __name__ == "__main__":
    main()
