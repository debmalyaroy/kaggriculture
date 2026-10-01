"""Assemble the MULTI-CHECKPOINT dispatcher agent from generated branches.

Emits a single self-contained agent (imports only math) that:
  * replays the base opening;
  * at EACH operator checkpoint (D6/D12/D15/D21/D27) reads the CUMULATIVE
    unlocked-shop set, forms the sorted key, and splices in the branch the
    generator evolved for that shop set (nested: each builds on the prior);
  * applies the reactive anti-dump SELL overlay (scarcity metering: never sell a
    crashable premium into a glutted market; dump-proof egg/wheat sell freely).

No legacy tape is embedded -- only the generated branches + the base scaffold's
opening prefix. Usage:
  python src/kaggriculture/bandit/build/build_multi_ckpt.py --out .local/candidates/bandit_mck.py
"""
from kaggriculture.paths import ROOT
import argparse, base64, json, os, sys, zlib

from kaggriculture.bandit.scaffold import our_route   # bandit-owned base scaffold (opening only is used)

CKPTS = [(146, 6), (290, 12), (362, 15), (506, 21), (650, 27)]


def pk(o):
    return base64.b85encode(zlib.compress(
        json.dumps(o, separators=(",", ":")).encode(), 9)).decode()


TEMPLATE = '''\
"""Multi-checkpoint adaptive bandit -- generated tape + world-dispatch at
D6/D12/D15/D21/D27 (cumulative shop set) + anti-dump reactive sell overlay."""
import base64, json, math, zlib

_BASE = json.loads(zlib.decompress(base64.b85decode("{base}")))
_BR = json.loads(zlib.decompress(base64.b85decode("{branches}")))
_CKPTS = {ckpts}
_MP = {{"WHEAT": (25.0, "log"), "CARROT": (35.0, "hinge"), "TOMATO": (60.0, "hinge"),
       "STRAWBERRY": (120.0, "sq"), "MELON": (250.0, "sq"), "EGG": (50.0, "log"),
       "MILK": (160.0, "linear"), "WOOL": (200.0, "sq"), "FERTILIZER": (100.0, "linear")}}
_DUMPPROOF = ("WHEAT", "EGG")
_S = {{}}


def _scarce(prod, minv):
    # True if the market is scarce (inv < I0) -> price above base -> worth selling
    return minv < 10000


def agent(obs):
    try:
        return _run(obs)
    except Exception:
        return {{"farmer": ["PASS"], "hands": [], "market": []}}


def _run(obs):
    step = int(obs.get("step", 0) or 0)
    if step == 0 or _S.get("route") is None or step < _S.get("last", -1):
        _S["route"] = list(_BASE)
        _S["done"] = []
    _S["last"] = step
    town = obs.get("town") or {{}}
    shops = town.get("unlocked_shops") or []
    for (cstep, day) in _CKPTS:
        if cstep <= step <= cstep + 2 and day not in _S["done"] and len(shops) >= 2:
            _S["done"].append(day)
            key = "|".join(sorted(shops[:8]))
            br = _BR.get(str(day), {{}}).get(key)
            if br and br.get("cont"):
                _S["route"] = _S["route"][:cstep] + br["cont"]
    r = _S["route"]
    act = r[step] if step < len(r) else {{"farmer": ["PASS"], "hands": [], "market": []}}
    act = {{"farmer": act.get("farmer", ["PASS"]),
           "hands": list(act.get("hands") or []),
           "market": [m for m in (act.get("market") or []) if m]}}
    # --- anti-dump reactive overlay: drop premium SELLs into a glutted market ---
    minv = ((obs.get("market") or {{}}).get("inventory") or {{}})
    kept = []
    for m in act["market"]:
        if isinstance(m, list) and len(m) >= 2 and m[0] == "SELL":
            p = m[1]
            if p not in _DUMPPROOF and int(minv.get(p, 10000)) >= 10000 \\
                    and step < 690:
                continue   # market glutted (>= I0) -> hold; shops re-scarce it
        kept.append(m)
    act["market"] = kept[:10]
    return act
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default=os.path.join(
        ROOT, "models", "bandit", "branches.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = our_route()[:719]
    br = json.load(open(a.branches, encoding="utf-8"))
    n = sum(len(v) for v in br.values())
    src = TEMPLATE.format(base=pk(base), branches=pk(br), ckpts=repr(CKPTS))
    open(a.out, "w", encoding="utf-8", newline="\n").write(src)
    compile(src, a.out, "exec")
    print(f"built {a.out} ({len(src):,} bytes) with {n} multi-checkpoint branches "
          f"across {len(br)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
