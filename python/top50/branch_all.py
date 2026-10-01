"""Run `branch` (crates/runner/src/bin/branch.rs) over tape folders in parallel shards.

    python python/top50/branch_all.py --tapes DIR [DIR ...] --out OUTDIR [--k 40] [--threads 32] [--shell M] [--tag T]

Shadow agent = our live best v63.1_rl (profiles v64rl, profile 35, group 35,35,36), plus --shell M when the
tapes are our own DAgger games (the shadow is then the candidate that played them). Every tape under the
given folders (recursively) is used once; shards run as separate processes. Writes OUTDIR/<tag>_NNN.tsv.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
BRANCH = os.path.join(BIN, "branch" + (".exe" if os.name == "nt" else ""))
V631 = ["--profiles", os.path.join(RL, "configs", "profiles", "v64rl.json"), "--pa", "35", "--group", "35,35,36"]
TAB, NL = chr(9), chr(10)
MIN_FREE_GB = float(os.environ.get("KRL_MIN_FREE_GB", "15"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--shell", default=None)
    ap.add_argument("--tag", default="b")
    ap.add_argument("--other-seat", action="store_true", help="study the tape's opponent (band_tapes.py tapes store our seat)")
    a = ap.parse_args()
    files = sorted({os.path.realpath(f) for d in a.tapes for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)})
    files = [f for f in files if not os.path.basename(f).startswith("index")]
    os.makedirs(a.out, exist_ok=True)
    if shutil.disk_usage(a.out).free / 2 ** 30 < MIN_FREE_GB:
        raise SystemExit(f"[branch_all] under {MIN_FREE_GB} GB free: not starting")
    n = max(1, min(a.threads * 4, len(files) // 20 + 1))
    shards = [files[i::n] for i in range(n)]
    extra = V631 + (["--shell", os.path.abspath(a.shell)] if a.shell else []) + (["--other-seat"] if a.other_seat else [])
    # id -> team of the studied seat (fetch.py tapes: "team"; band_tapes.py tapes: "opp_team"; our DAgger games: "us")
    import json
    with open(os.path.join(a.out, f"index_{a.tag}.tsv"), "w", encoding="utf-8", newline=NL) as fh:
        fh.write(TAB.join(["id", "team", "source"]) + NL)
        for f in files:
            try:
                t = json.load(open(f, encoding="utf-8"))
            except ValueError:
                continue
            team = t.get("opp_team") if a.other_seat else (t.get("team") or t.get("us") or "us")
            fh.write(TAB.join([os.path.basename(f)[:-5], str(team), a.tag]) + NL)

    def one(i):
        d = tempfile.mkdtemp(prefix="branch-")
        try:
            for f in shards[i]:
                os.symlink(f, os.path.join(d, os.path.basename(f)))
            r = subprocess.run([BRANCH, "--tapes", d, "--out", os.path.join(a.out, f"{a.tag}_{i:03d}.tsv"), "--k", str(a.k), *extra],
                               cwd=RL, capture_output=True, text=True)
            return r.stderr.strip()[-200:]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"[branch_all] {len(files)} tapes in {n} shards, k {a.k}, threads {a.threads}", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for i, line in enumerate(ex.map(one, range(n)), 1):
            if i % 10 == 0 or i == n:
                print(f"[branch_all] {i}/{n}: {line}", flush=True)


if __name__ == "__main__":
    main()
