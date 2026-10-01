"""One DAgger round for the sales shell (queue Q44, Q45).

    python python/top50/dagger.py --shell weights/shell/rN/shell.json --round N+1 [--games 3000] [--threads 32]

1. PLAY: the current candidate (v63.1_rl + shell rN) plays real opponents' recorded streams through the
   guards (tapeplay --guarded --record): PPO's training pool (bands, 2700+ players, top teams since 18 Sep,
   our ladder games) -- the states OUR policy reaches, which the top players' own games never show.
2. LABEL: `branch` on those games with the candidate as the shadow: at its sale decisions, hold / part / all
   are played to the end on the exact engine (the expert answer in states nobody showed us).
3. TRAIN: python/top50/train_shell.py on every label set so far (top players + all DAgger rounds) -> rN+1.
4. EVAL: python/top50/eval_shell.py (held-out ladder + band gate vs v63.1_rl).
Outputs under data/top50/dagger/rN+1/ and weights/shell/rN+1/.
"""
import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
POOL = ["lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus", "top", "top_late"]  # not "ladder": our ladder games are the shell's selection set
MIN_FREE_GB = float(os.environ.get("KRL_MIN_FREE_GB", "15"))


def hard_only(files, threads):
    """keep the recorded opponents v63.1_rl loses to or beats by < $3,000"""
    n = max(1, threads)
    shards = [files[i::n] for i in range(n)]

    def one(sh):
        d = tempfile.mkdtemp(prefix="dhard-")
        try:
            for f in sh:
                os.symlink(f, os.path.join(d, os.path.basename(f)))
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"],
                               cwd=RL, capture_output=True, text=True)
            keep = set()
            for ln in r.stdout.splitlines():
                x = ln.split(chr(9))
                if len(x) >= 6 and float(x[4]) - float(x[5]) < 3000:
                    keep.add(x[0])
            return [f for f in sh if os.path.basename(f)[:-5] in keep]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    with ThreadPoolExecutor(n) as ex:
        return sorted(f for part in ex.map(one, shards) for f in part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--hard", action="store_true", help="play only where v63.1_rl loses or wins by < $3,000 (the states that decide results): "
                    "the training pool + data/tapes/hard, filtered by a v63.1_rl pass")
    a = ap.parse_args()
    shell = os.path.abspath(a.shell)
    rd = os.path.join(RL, "data", "top50", "dagger", f"r{a.round}")
    games, lab = os.path.join(rd, "games"), os.path.join(rd, "branch")
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(games)
    if shutil.disk_usage(rd).free / 2 ** 30 < MIN_FREE_GB:
        raise SystemExit("[dagger] low disk: not starting")
    files = sorted({os.path.realpath(f) for d in POOL for f in glob.glob(os.path.join(RL, "data", "tapes", "train", d, "**", "*.json"), recursive=True)})
    if a.hard:
        files = sorted(set(files) | {os.path.realpath(f) for f in glob.glob(os.path.join(RL, "data", "tapes", "hard", "*.json"))})
        files = list({os.path.basename(f): f for f in files}.values())  # one tape per game id (hard/ repeats train tapes)
        files = hard_only(files, a.threads)
        print(f"[dagger] hard pool: {len(files)} games where v63.1_rl loses or wins by < $3,000", flush=True)
    files = sorted({os.path.basename(f): f for f in files}.values())
    random.Random(a.round).shuffle(files)
    files = files[:a.games]
    n = max(1, a.threads)
    shards = [files[i::n] for i in range(n)]

    def one(sh):
        d = tempfile.mkdtemp(prefix="dagger-")
        try:
            for f in sh:
                os.symlink(f, os.path.join(d, os.path.basename(f)))
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36",
                                "--shell", shell, "--record", games], cwd=RL, capture_output=True, text=True)
            return r.stderr.strip()[-160:]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"[dagger] round {a.round}: candidate {shell} plays {len(files)} recorded opponents", flush=True)
    with ThreadPoolExecutor(n) as ex:
        for line in ex.map(one, shards):
            pass
    print(f"[dagger] {len(os.listdir(games))} games recorded; labelling", flush=True)
    py = sys.executable
    subprocess.run([py, os.path.join(RL, "python", "top50", "branch_all.py"), "--tapes", games, "--out", lab, "--k", str(a.k),
                    "--threads", str(a.threads), "--shell", shell, "--tag", f"d{a.round}"], cwd=RL, check=True)
    dirs = [os.path.join(RL, "data", "top50", "branch", x) for x in ("old", "new")]
    dirs += sorted(glob.glob(os.path.join(RL, "data", "top50", "dagger", "r*", "branch")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    out = os.path.join(RL, "weights", "shell", f"r{a.round}")
    subprocess.run([py, os.path.join(RL, "python", "top50", "train_shell.py"), "--branch", *dirs, "--out", out], cwd=RL, check=True)
    subprocess.run([py, os.path.join(RL, "python", "top50", "eval_shell.py"), "--shell", os.path.join(out, "shell.json"),
                    "--threads", str(min(16, a.threads))], cwd=RL, check=True)
    print(f"[dagger] round {a.round} done -> {out}")


if __name__ == "__main__":
    main()
