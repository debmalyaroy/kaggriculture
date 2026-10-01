"""Copy-race profiles (queue Q30): variants of v63 (rl3 profile 35) screened against the COPY opponents we
actually met on the ladder.

    python python/copy_screen.py [--threads 16] [--top 3]

Every one of v63's 22 ladder losses was against a COPY opponent (same opening as ours). This builds
data/gates/copy_cands.json = rl3 (ids 0-35) + one- and two-knob variants of 35 on the race knobs
(lead horizons, V9 race horizon, RSA look-ahead, clone / mirror / escalated race depth, glut margins, AFR,
counter-D rival model, market pressure), then:
  1. screen: each variant, fixed all game, on the TRAINING half's COPY games (data/tapes/train/ladder),
     paired vs profile 35 on the same recorded opponents;
  2. confirm: the --top variants on the HELD-OUT half (data/tapes/ladder_val), COPY and all groups;
  3. the band gate (988 real players' tapes, the objective) for the best confirmed variant.
Caveat: the recorded opponent is open-loop. Our sim reproduces our own agents' real margins exactly, but
a copy would react to a different us, so a gain here is an estimate to be confirmed on the ladder.
Writes data/gates/copy_screen.json.
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")
OUT_TABLE = os.path.join(RL, "data", "gates", "copy_cands.json")  # generated on the box: data, not a config

VARIANTS = [
    ("lead32", {"ev_h": 32, "dp_h": 32, "mp_h": 32}), ("lead40", {"ev_h": 40, "dp_h": 40, "mp_h": 40}), ("lead16", {"ev_h": 16, "dp_h": 16, "mp_h": 16}),
    ("v9r72", {"v9_race_default": 72, "v9_race_max": 84}), ("v9r48", {"v9_race_default": 48, "v9_race_max": 60}),
    ("rsa16", {"rsa_look": 16}), ("rsa20", {"rsa_look": 20}), ("rsa8", {"rsa_look": 8}),
    ("clone12", {"race_clone": 12}), ("clone16", {"race_clone": 16}), ("clone24", {"race_clone": 24}),
    ("mirror12", {"race_mirror": 12}), ("mirror36", {"race_mirror": 36}), ("mirror48", {"race_mirror": 48}),
    ("esc36", {"race_escalated": 36}), ("esc48", {"race_escalated": 48}),
    ("pxm-2", {"racepx_margin": -2}), ("pxm+2", {"racepx_margin": 2}), ("gate-2", {"racegate_margin": -2}), ("gate+2", {"racegate_margin": 2}),
    ("afr", {"afr_on": True}), ("cxd1", {"cxd_model": 1}), ("cxd2", {"cxd_model": 2}), ("press", {"press_on": True}),
    ("v92h24", {"v92_h": 24}), ("v92h72", {"v92_h": 72}),
    ("lead32_rsa16", {"ev_h": 32, "dp_h": 32, "mp_h": 32, "rsa_look": 16}), ("clone16_mirror36", {"race_clone": 16, "race_mirror": 36}),
]


def play(args, files, threads):
    n = max(1, min(threads, len(files)))
    shards = [files[i::n] for i in range(n)]

    def one(sh):
        d = tempfile.mkdtemp(prefix="cscr-")
        try:
            for f in sh:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, *args], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"tapeplay rc {r.returncode}: {r.stderr[-400:]}")
            out = {}
            for ln in r.stdout.splitlines():
                x = ln.split("\t")
                if len(x) >= 6:
                    m = float(x[4]) - float(x[5])
                    out[x[0]] = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)

    res = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for r in ex.map(one, shards):
            res.update(r)
    return res


def paired(c, r):
    ks = [k for k in c if k in r]
    b = sum(c[k] > r[k] for k in ks)
    w = sum(c[k] < r[k] for k in ks)
    n = b + w
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0
    return {"games": len(ks), "wins": int(sum(c[k] == 1 for k in ks)), "ref_wins": int(sum(r[k] == 1 for k in ks)), "better": b, "worse": w, "p": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    t = json.load(open(RL3, encoding="utf-8"))
    base = dict(t["profiles"][35])
    ids = {}
    for name, kn in VARIANTS:
        p = dict(base)
        p.update(kn)
        p["name"] = f"c35_{name}"
        ids[name] = len(t["profiles"])
        t["profiles"].append(p)
    t["note"] = "rl3 + copy-race variants of profile 35 (python/copy_screen.py)"
    json.dump(t, open(OUT_TABLE, "w", encoding="utf-8"), indent=1)
    idx = lambda d: {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(d, "index.tsv"), encoding="utf-8"), delimiter="\t")}  # noqa: E731
    tr_dir, va_dir = os.path.join(RL, "data", "tapes", "train", "ladder"), os.path.join(RL, "data", "tapes", "ladder_val")
    tr_g, va_g = idx(tr_dir), idx(va_dir)
    tr_copy = [os.path.join(tr_dir, k + ".json") for k, g in tr_g.items() if g == "COPY"]
    va_all = sorted(glob.glob(os.path.join(va_dir, "**", "*.json"), recursive=True))
    pa = lambda k: ["--profiles", OUT_TABLE, "--guarded", "--pa", str(k)]  # noqa: E731
    rep = {"table": OUT_TABLE, "screen": {}, "confirm": {}}
    ref = play(pa(35), tr_copy, a.threads)
    print(f"[copy] screen on {len(tr_copy)} training-half COPY games; v63 wins {int(sum(v == 1 for v in ref.values()))}", flush=True)
    for name, k in ids.items():
        rep["screen"][name] = paired(play(pa(k), tr_copy, a.threads), ref)
        s = rep["screen"][name]
        print(f"[copy] {name:18s} (id {k}): wins {s['wins']} vs {s['ref_wins']}, paired +{s['better']}/-{s['worse']} p {s['p']}", flush=True)
    top = sorted(ids, key=lambda n: (rep["screen"][n]["better"] - rep["screen"][n]["worse"]), reverse=True)[:a.top]
    top = [n for n in top if rep["screen"][n]["better"] > rep["screen"][n]["worse"]]
    ref_va = play(pa(35), va_all, a.threads)
    for name in top:
        c = play(pa(ids[name]), va_all, a.threads)
        rep["confirm"][name] = {"all": paired(c, ref_va), "COPY": paired({k: v for k, v in c.items() if va_g.get(k) == "COPY"}, ref_va)}
        x = rep["confirm"][name]
        print(f"[copy] held-out {name}: all +{x['all']['better']}/-{x['all']['worse']} p {x['all']['p']}; COPY +{x['COPY']['better']}/-{x['COPY']['worse']} p {x['COPY']['p']}", flush=True)
    good = [n for n in rep["confirm"] if rep["confirm"][n]["all"]["better"] > rep["confirm"][n]["all"]["worse"]]
    if good:
        best = max(good, key=lambda n: rep["confirm"][n]["all"]["better"] - rep["confirm"][n]["all"]["worse"])
        r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", str(ids[best]), "--name", f"copy-{best}",
                            "--profiles", OUT_TABLE, "--ref-profile", "35", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        rep["band_gate"] = {"variant": best, "line": (r.stdout.strip().splitlines() or [""])[-1]}
        print(r.stdout[-1200:], flush=True)
    else:
        rep["band_gate"] = "skipped: no variant beats v63 on the held-out ladder half"
        print(f"[copy] {rep['band_gate']}", flush=True)
    json.dump(rep, open(os.path.join(RL, "data", "gates", "copy_screen.json"), "w"), indent=1)
    print("[copy] -> data/gates/copy_screen.json")


if __name__ == "__main__":
    main()
