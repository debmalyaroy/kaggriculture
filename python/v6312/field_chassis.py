"""Chassis-only field test: a chassis (layers cut) vs the 80 public agents across the 64 worlds, paired games.

    python python/v6312/field_chassis.py make NAME CHASSIS_DIR        -> .local/fieldc/NAME (candidate dir)
    python python/v6312/field_chassis.py run NAME [--workers 8] [--per-opp 16]
    python python/v6312/field_chassis.py cmp NAME_A NAME_B            (paired: better / worse, sign test, per world)

Candidate = the c4 submission folder with base/ replaced by the chassis, the current agent-stdio (cluster router) and
`--cut chassis`, so only the chassis plays. Games = per opponent `--per-opp` (world, seed) pairs from the 80-agent gate
set (.local/v6312/a0_v6311.jsonl, fixed by hash so every candidate plays the same games), both seats. Results:
data/chassis/field/NAME.jsonl (opp, world_label, seed, seat, us, them, gap).
"""
import argparse, hashlib, json, math, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python", "rshell"))
import public25 as P  # noqa: E402

SRC = os.path.join(RL, ".local", "v6312", "cand_c4")
EXE = os.path.join(RL, "target-v6312n", "release", "agent-stdio.exe")
OUT = os.path.join(RL, "data", "chassis", "field")


def make(name, chassis):
    d = os.path.join(RL, ".local", "fieldc", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(os.path.join(d, "base"))
    shutil.copytree(chassis, os.path.join(d, "base"))
    shutil.copy(EXE, os.path.join(d, "agent-stdio.exe"))
    m = open(os.path.join(d, "main.py"), encoding="utf-8").read()
    a = 'cmd = [path, "--base", os.path.join(root, "base")]'
    assert a in m
    m = m.replace(a, a + '\n        cmd += ["--cut", "chassis"]', 1)
    open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(m)
    print("candidate", d)


def tasks(per_opp, panel=None):
    rows = [json.loads(l) for l in open(os.path.join(RL, ".local", "v6312", "a0_v6311.jsonl"))]
    if panel:
        # a fixed panel of opponents x every world of the gate set x both seats
        ps = set(json.load(open(panel)))
        return sorted({(r["opp"], r["world"], r["seed"], r["seat"]) for r in rows if r["opp"] in ps})
    by = {}
    for r in rows:
        by.setdefault(r["opp"], {}).setdefault((r["world"], r["seed"]), []).append(r["seat"])
    out = []
    for opp, ws in sorted(by.items()):
        keys = sorted(ws, key=lambda k: hashlib.md5(f"{opp}{k}".encode()).hexdigest())[:per_opp]
        for (w, s) in keys:
            for seat in (0, 1):
                out.append((opp, w, s, seat))
    return out


def run(name, workers, per_opp, panel=None, tag="", vs=None, check=32, p_stop=0.01, same_stop=160):
    """Play the candidate's games. FAST TRACK (30 Sep): games go in a hash-shuffled order (any prefix is a balanced
    sample of opponents / worlds / seats) and, with `vs` (a results name), the run STOPS EARLY once it is decided:
    every `check` games the paired sign test vs `vs`; stop when p < p_stop, or when >= same_stop paired games have
    no discordant result (identical). Stopped runs print FAST_STOP with the reason."""
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name + tag + ".jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["opp"], r["seed"], r["seat"]) for r in map(json.loads, open(out)) if "us" in r}
    cand = os.path.join(RL, ".local", "fieldc", name, "main.py")
    ts = [(cand, o, os.path.join(RL, ".local/field80/field", o + ".py"), w, s, seat) for (o, w, s, seat) in tasks(per_opp, panel) if (o, s, seat) not in done]
    ts.sort(key=lambda t: hashlib.md5(f"{t[1]}{t[4]}{t[5]}".encode()).hexdigest())
    base = {}
    if vs and os.path.exists(os.path.join(OUT, vs + ".jsonl")):
        base = {(r["opp"], r["seed"], r["seat"]): r for r in map(json.loads, open(os.path.join(OUT, vs + ".jsonl"))) if "us" in r}
    mine = {}
    if os.path.exists(out):
        mine = {(r["opp"], r["seed"], r["seat"]): r for r in map(json.loads, open(out)) if "us" in r}
    print(name, len(ts), "games", flush=True)

    def decided():
        ks = [k for k in mine if k in base]
        b = sum(1 for k in ks if pts(mine[k]) > pts(base[k]))
        w = sum(1 for k in ks if pts(mine[k]) < pts(base[k]))
        m, kk = b + w, min(b, w)
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m) if m else 1.0
        if m and p < p_stop:
            return f"decided p {p:.1e} (+{b}/-{w}, n {len(ks)})"
        if len(ks) >= same_stop and m == 0:
            return f"identical over {len(ks)} games"
        return None

    ex = ProcessPoolExecutor(workers)
    stop = None
    with open(out, "a") as f:
        futs = [ex.submit(P.play, t) for t in ts]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            f.write(json.dumps(r) + chr(10))
            f.flush()
            if "us" in r:
                mine[(r["opp"], r["seed"], r["seat"])] = r
            if i % 200 == 0:
                print(name, i, flush=True)
            if base and (i + 1) % check == 0:
                stop = decided()
                if stop:
                    for x in futs:
                        x.cancel()
                    break
    ex.shutdown(wait=False, cancel_futures=True)
    if stop:
        print("FAST_STOP", name, stop, flush=True)
        # the running workers' agent-stdio children die with their games; kill stragglers of this candidate
        try:
            import subprocess
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-Process agent-stdio -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*fieldc*" + name + "*' } | Stop-Process -Force"],
                           capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)


def pts(r):
    return 1.0 if r["us"] > r["them"] else (0.5 if r["us"] == r["them"] else 0.0)


def cmp(a, b):
    """a / b = results file names without .jsonl (e.g. ch_X_panel12)."""
    A = {(r["opp"], r["seed"], r["seat"]): r for r in map(json.loads, open(os.path.join(OUT, a + ".jsonl"))) if "us" in r}
    B = {(r["opp"], r["seed"], r["seat"]): r for r in map(json.loads, open(os.path.join(OUT, b + ".jsonl"))) if "us" in r}
    ks = sorted(set(A) & set(B))
    sa = sum(pts(A[k]) for k in ks) / max(1, len(ks))
    sb = sum(pts(B[k]) for k in ks) / max(1, len(ks))
    bt = sum(1 for k in ks if pts(A[k]) > pts(B[k]))
    wo = sum(1 for k in ks if pts(A[k]) < pts(B[k]))
    m, kk = bt + wo, min(bt, wo)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m) if m else 1.0
    print(f"{a} vs {b}: n {len(ks)}  score {sa:.4f} vs {sb:.4f}  better {bt} worse {wo}  p {p:.2e}")
    per = {}
    for k in ks:
        w = A[k]["world"]
        e = per.setdefault(w, [0, 0.0, 0.0])
        e[0] += 1; e[1] += pts(A[k]); e[2] += pts(B[k])
    bad = [(w, e) for w, e in per.items() if e[1] < e[2]]
    print(f"worlds where {a} scores below {b}: {len(bad)} of {len(per)}")
    for w, e in sorted(bad, key=lambda x: x[1][1] - x[1][2])[:15]:
        print(f"  {w:34} n {e[0]:3}  {e[1] / e[0]:.3f} vs {e[2] / e[0]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-opp", type=int, default=16)
    ap.add_argument("--panel", default=None, help="JSON list of opponents: every world x both seats")
    ap.add_argument("--tag", default="", help="suffix of the results file (e.g. _panel12)")
    ap.add_argument("--vs", default=None, help="FAST TRACK: stop early vs this results name (paired sign test)")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.args[0], a.args[1])
    elif a.cmd == "run":
        run(a.args[0], a.workers, a.per_opp, a.panel, a.tag, a.vs)
    elif a.cmd == "cmp":
        cmp(a.args[0], a.args[1])
