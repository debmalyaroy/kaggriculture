"""Split our ladder games (data/ladder/<sub>/tapes, python/ladder_pull.py) into a PPO training half and a
held-out validation half (queue Q29).

    python python/ladder_split.py [--copy-weight 5] [--threads 16]

By episode id: even -> train, odd -> validation (so the same game never sits in both, whichever of our
agents played it). Each tape is labelled with the shield's day-0 opponent group (tapeplay --verify
--obs-dump: DIFFERENT if we shared under 10% of day-0 squares, else COPY on equal cash at step 1,
else PARTIAL). All 22 of v63's ladder losses were against COPY opponents, so COPY games weigh
--copy-weight x in training.
  data/tapes/train/ladder/<id>.json                 training half (PPO and the oracle read data/tapes/train)
  data/tapes/train/ladder_copy_x/<id>__cK.json      symlinks: COPY games (--copy-weight - 1) more times
  data/tapes/ladder_val/<group>/<id>.json           held-out half (ppo.py validation v4)
  data/tapes/ladder_val/index.tsv, data/tapes/train/ladder/index.tsv   id, sub, group, band, real margin
Re-running rebuilds all of these (new ladder games join after each ladder_pull).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
TAB = chr(9)


def groups(files, threads):
    shards = [files[i::threads] for i in range(threads) if files[i::threads]]

    def one(shard):
        d = tempfile.mkdtemp(prefix="lsplit-")
        try:
            for f in shard:
                shutil.copy(f, d)
            seat = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8"))["seat"] for f in shard}
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--verify", "--obs-dump", d + ".obs"], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-400:])
            g = {}
            for ln in open(d + ".obs", encoding="utf-8"):
                x = ln.rstrip().split(TAB)
                if int(x[2]) == 1 and int(x[1]) == seat.get(x[0], -1):
                    pos0, cash1 = float(x[3 + 90]), float(x[3 + 89])
                    g[x[0]] = "DIFFERENT" if pos0 < 0.1 else ("COPY" if cash1 >= 0.5 else "PARTIAL")
            return g
        finally:
            shutil.rmtree(d, ignore_errors=True)
            for ext in (".obs",):
                try:
                    os.remove(d + ext)
                except OSError:
                    pass

    out = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for r in ex.map(one, shards):
            out.update(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-weight", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RL, "data", "ladder", "*", "tapes", "*.json")))
    g = groups(files, a.threads)
    tr = os.path.join(RL, "data", "tapes", "train", "ladder")
    trx = os.path.join(RL, "data", "tapes", "train", "ladder_copy_x")
    va = os.path.join(RL, "data", "tapes", "ladder_val")
    for d in (tr, trx, va):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    rows = {"train": [], "val": []}
    seen = set()
    for f in files:
        tid = os.path.basename(f)[:-5]
        ep = tid.split("_")[0]
        if ep in seen:  # the same episode from two of our subs (v63 vs v62.1 on the ladder): keep one copy
            continue
        seen.add(ep)
        t = json.load(open(f, encoding="utf-8"))
        grp = g.get(tid, "unknown")
        m = (t["rewards"][t["seat"]] or 0) - (t["rewards"][1 - t["seat"]] or 0)
        sub = f.split(os.sep)[-3]
        row = (tid, sub, grp, t.get("band") or "unknown", m)
        if int(ep) % 2 == 0:
            shutil.copy(f, tr)
            if grp == "COPY":
                for k in range(1, a.copy_weight):
                    os.symlink(os.path.join(tr, tid + ".json"), os.path.join(trx, f"{tid}__c{k}.json"))
            rows["train"].append(row)
        else:
            os.makedirs(os.path.join(va, grp), exist_ok=True)
            shutil.copy(f, os.path.join(va, grp))
            rows["val"].append(row)
    for k, d in (("train", tr), ("val", va)):
        with open(os.path.join(d, "index.tsv"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("id\tsub\tgroup\tband\tmargin\n")
            for r in rows[k]:
                fh.write(TAB.join(str(x) for x in r) + "\n")
    for k in ("train", "val"):
        by = {}
        for r in rows[k]:
            x = by.setdefault(r[2], [0, 0])
            x[0 if r[4] > 0 else 1] += 1 if r[4] != 0 else 0
        print(f"[ladder_split] {k}: {len(rows[k])} games; W/L by group {by}", flush=True)
    print(f"[ladder_split] COPY games weigh {a.copy_weight}x in training ({len(os.listdir(trx))} extra links)")


if __name__ == "__main__":
    main()
