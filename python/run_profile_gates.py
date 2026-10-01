"""Field gates for Rust-agent profiles: each profile plays the winplan roster (24 worlds, both seats
on 2) through the Rust bridge; results pair against the Python v61.1 gate (same keys).

    python python/run_profile_gates.py --profiles 0,13,12,2,10 --workers 6 [--ref v611_full__167700b1eb28]
Shims: data/winplan/rust/v611_p<k>.py (content carries the binary sha, so a rebuilt binary is a
new gate hash). Resumable (winplan.gate skips recorded games).
"""
import argparse, hashlib, json, os, sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = RL  # one repo since the 2026-10-01 merge
sys.path.insert(0, os.path.join(REPO, "src"))
EXE_DEFAULT = os.path.join(RL, "target-dev", "release", "agent-stdio.exe")
PROFILES = os.path.join(RL, "configs", "profiles", "v1.json")
SHIMS = os.path.join(REPO, "data", "winplan", "rust")

SHIM = '''# Rust v61.1 port, profile {k} ({name}), clone profile {clone}; binary sha {sha}
import sys
sys.path.insert(0, r"{py}")
from rust_agent import RustAgent as _RustAgent
_A = _RustAgent(exe=r"{exe}", profiles=r"{profiles}", profile={k}, clone_profile={clone}, clone_strict={strict})


def agent(observation, configuration=None):
    return _A(observation, configuration)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="0,13,12,2,10")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ref", default="v611_full__167700b1eb28")
    ap.add_argument("--exe", default=EXE_DEFAULT)
    ap.add_argument("--table", default=PROFILES, help="profile table json")
    ap.add_argument("--strict", action="store_true", help="strict clone gate (positions + similarity, no cash mirror)")
    ap.add_argument("--clone", type=int, default=None, help="clone-gated controller: switch to this profile on clone days")
    a = ap.parse_args()
    table = os.path.abspath(a.table)
    from kaggriculture.winplan import gate as G, paths as P
    os.makedirs(SHIMS, exist_ok=True)
    EXE = a.exe
    sha = hashlib.sha256(open(EXE, "rb").read()).hexdigest()[:12]
    names = [p["name"] for p in json.load(open(table))["profiles"]]
    ref = G.load(os.path.join(P.GATES, a.ref + ".jsonl"))
    for k in [int(x) for x in a.profiles.split(",")]:
        tag = f"p{k}" + (f"_ctrl{a.clone}" if a.clone is not None else "") + ("s" if a.strict and a.clone is not None else "")
        shim = os.path.join(SHIMS, f"v611_{tag}.py")
        open(shim, "w").write(SHIM.format(k=k, name=names[k], sha=sha, py=os.path.join(RL, "python"), exe=EXE, profiles=table, clone=a.clone, strict=a.strict))
        label = f"rust_v611_{tag}_{names[k]}"
        print(f"GATE-START {label}", flush=True)
        G.run_gate(shim, label, G.roster(), worlds=24, symmetry_worlds=2, workers=a.workers)
        path = os.path.join(P.GATES, f"{label}__{G._file_hash(shim)}.jsonl")
        rows = G.load(path)
        w = sum(r["gap"] > 0 for r in rows.values())
        l = sum(r["gap"] < 0 for r in rows.values())
        better = worse = same_bank = 0
        for key, r in rows.items():
            q = ref.get(key)
            if not q:
                continue
            same_bank += r["us"] == q["us"] and r["them"] == q["them"]
            o = (r["gap"] > 0) - (r["gap"] < 0)
            oq = (q["gap"] > 0) - (q["gap"] < 0)
            better += o > oq
            worse += o < oq
        print(f"GATE-DONE {label} W {w} L {l} n {len(rows)} | vs {a.ref}: better {better} worse {worse} identical-banks {same_bank}", flush=True)


if __name__ == "__main__":
    main()
