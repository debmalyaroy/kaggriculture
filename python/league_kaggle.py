"""Random-profile league games on Kaggle: N private notebooks in parallel -> data/leagues/rand.

    python python/league_kaggle.py run [--upto 120] [--notebooks 5] [--threads 4]
    python python/league_kaggle.py status

Plans every batch below --upto that is not on disk yet, splits the missing range into
--notebooks contiguous chunks (league-rand --first/--batches, so file names never collide),
pushes one private notebook per chunk (kaggle/kernels/krl-league-<i>), waits, downloads each
output through its manifest, and copies batch-*.{tsv,obs} into data/leagues/rand (never
overwriting a batch that exists). Resumable: data/kaggle_out/league_state.json records each
notebook's phase (pushed -> downloaded -> filed); re-running continues where it stopped.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kjob  # noqa: E402

RL = kjob.RL
DEST = os.path.join(RL, "data", "leagues", "rand")
OUT_ROOT = os.path.join(RL, "data", "kaggle_out")
STATE = os.path.join(OUT_ROOT, "league_state.json")
TEMPLATE = os.path.join(RL, "kaggle", "league", "notebook_template.py")
BIN_DS = f"{kjob.OWNER}/kaggriculture-rl-bin"
WAIT_HOURS = 11


def load():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"notebooks": {}}


def save(st):
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, STATE)


def have():
    return {int(os.path.basename(f)[6:11]) for f in glob.glob(os.path.join(DEST, "batch-*.tsv"))}


def plan(upto, n_nb):
    missing = sorted(set(range(upto)) - have())
    if not missing:
        return []
    chunks, size = [], -(-len(missing) // n_nb)
    i = 0
    while i < len(missing):
        run = [missing[i]]
        while i + 1 < len(missing) and missing[i + 1] == run[-1] + 1 and len(run) < size:
            i += 1
            run.append(missing[i])
        chunks.append((run[0], len(run)))
        i += 1
    return chunks


def file_out(name):
    d = os.path.join(OUT_ROOT, name)
    n = 0
    os.makedirs(DEST, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(d, "league", "batch-*.obs"))):
        tsv = f[:-4] + ".tsv"
        if not os.path.exists(tsv):
            continue
        dst = os.path.join(DEST, os.path.basename(tsv))
        if os.path.exists(dst):
            continue
        shutil.copy(f, os.path.join(DEST, os.path.basename(f)))   # obs first, tsv marks complete
        shutil.copy(tsv, dst + ".part")
        os.replace(dst + ".part", dst)
        n += 1
    shutil.rmtree(d, ignore_errors=True)
    return n


def cmd_run(a):
    st = load()
    active = {k: v for k, v in st["notebooks"].items() if v["phase"] != "filed"}
    if not active:
        chunks = plan(a.upto, a.notebooks)
        if not chunks:
            kjob.log(f"league: all {a.upto} batches present; nothing to do")
            return
        code0 = open(TEMPLATE, encoding="utf-8").read()
        for i, (first, n) in enumerate(chunks):
            name = f"krl-league-{i}"
            code = code0.replace("__FIRST__", str(first)).replace("__BATCHES__", str(n)).replace("__THREADS__", str(a.threads))
            kjob.make_kernel(name, [BIN_DS], code)
            st["notebooks"][name] = {"first": first, "batches": n, "phase": "planned"}
        save(st)
        kjob.log(f"league: planned {len(chunks)} notebooks: " + ", ".join(f"{k} batches {v['first']}..{v['first'] + v['batches'] - 1}" for k, v in st["notebooks"].items() if v["phase"] == "planned"))
    for name, nb in st["notebooks"].items():
        if nb["phase"] == "planned":
            nb["pushed_utc"] = kjob.push(name)
            nb["phase"] = "pushed"
            shutil.rmtree(os.path.join(OUT_ROOT, name), ignore_errors=True)
            save(st)
    t0 = time.time()
    while True:
        pending = [k for k, v in st["notebooks"].items() if v["phase"] == "pushed"]
        for name in pending:
            s = kjob.status(name)
            if s == "COMPLETE":
                nb = st["notebooks"][name]
                run = kjob.download(name, os.path.join(OUT_ROOT, name), "league/manifest.txt", nb["pushed_utc"])
                nb.update(phase="downloaded", secs=run.get("secs"))
                save(st)
                n = file_out(name)
                nb.update(phase="filed", filed=n)
                save(st)
                kjob.log(f"FILED {name}: {n} batches ({nb['batches']} planned) in {run.get('secs')}s on Kaggle")
            elif s in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                raise RuntimeError(f"{name} ended {s}; see its log on Kaggle")
        if all(v["phase"] == "filed" for v in st["notebooks"].values()):
            kjob.log(f"league: done; {len(have())} batches on disk")
            st["notebooks"] = {}
            save(st)
            return
        if time.time() - t0 > WAIT_HOURS * 3600:
            raise RuntimeError("league: notebooks still running after the wait limit")
        time.sleep(kjob.POLL_SEC)


def cmd_status(a):
    st = load()
    print(f"local batches: {len(have())}")
    for name, nb in st["notebooks"].items():
        print(name, nb, kjob.status(name) if nb["phase"] == "pushed" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--upto", type=int, default=120)
    ap.add_argument("--notebooks", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
