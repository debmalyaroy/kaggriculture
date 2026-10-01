"""Learned-dispatcher training data (2026-09-07).

Robustness principle (why this beats sim_search): label from REAL outcomes
vs ADAPTIVE opponents, not fixed-opponent rollouts. For each (adaptive
opponent, seed):
  1. play the common base prefix to day 6 (step 144), capture the rich
     feature vector at that boundary (includes the rival's board so far);
  2. for each candidate day-6 continuation, play [base[:144] + cont] to
     terminal vs the SAME opponent, record final bank;
  3. label = index of the best-banking continuation.
Train a tree: features@144 -> best continuation. The tree learns "given
this observed state (incl. opponent board), continuation X wins" from
actual adaptive play.

Output: models/trackp/dispatcher/train_144.jsonl  {f:[...], y:int, opp, seed, banks:[...]}

    python src/trackp/harness/dispatcher_train_data.py --seeds 3,4,5,6
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import glob
import json
import os
import re
import sys
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
os.environ.setdefault("KAGG_BIN", os.path.join(ROOT, "rustengine", "kagg.exe"))
import kaggriculture.engine.serve_match as SM  # noqa: E402
import dispatcher_features as DF                               # noqa: E402

SPLIT = 144


def base_route():
    src = open(os.path.join(ROOT, "agents", "v45.0_bandit.py"),
               encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    return json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))[:719]


def candidates(base):
    """Day-6 continuations [144:719] the dispatcher chooses among."""
    cands = [("base", base[SPLIT:719])]
    d3 = json.load(open(os.path.join(ROOT, "models", "trackp",
                                     "day3_branches.json"), encoding="utf-8"))
    for k, v in list(d3.items())[:3]:
        cont = (base[:74] + v["cont"])[SPLIT:719]
        cands.append(("d3_" + k[:8], cont))
    d6 = json.load(open(os.path.join(ROOT, "models", "trackp",
                                     "day6_branches.json"), encoding="utf-8"))
    for k, v in list(d6.items())[:4]:
        cands.append(("d6_" + k[:12], v["cont"]))
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3,4,5,6")
    ap.add_argument("--base-route", default=None,
                    help="json route to use as the prefix (aggressive economy)")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "dispatcher", "train_144.jsonl"))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    base = base_route()
    if a.base_route:
        import json as _j
        br = _j.load(open(a.base_route, encoding="utf-8"))
        base = (br["route"] if isinstance(br, dict) else br)[:719]
    cands = candidates(base)
    print(f"candidates: {[c[0] for c in cands]}", flush=True)
    opps = []
    for d in ("data/gauntlet", "data/gauntlet_adaptive"):
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.py"))):
            if "our_" not in os.path.basename(f):
                opps.append(f)
    print(f"opponents: {len(opps)}", flush=True)
    srv = SM.Serve()
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for op in opps:
            on = os.path.basename(op)[:-3]
            try:
                OA = SM.load_agent(op)
            except Exception:                                  # noqa: BLE001
                continue
            for seed in seeds:
                for seat in (0, 1):
                    # play base to SPLIT, capture features@SPLIT
                    try:
                        js = srv.cmd(f"RESET {seed}")
                        feat = None
                        for st in range(719):
                            me = SM.action_to_line(
                                base[st] if st < len(base) else None)
                            ot = SM.action_to_line(OA(SM.obs_for(1 - seat,
                                                                 js)))
                            la, lb = (me, ot) if seat == 0 else (ot, me)
                            js = srv.cmd(f"STEP2 {la}\x1e{lb}")
                            if "error" in js:
                                break
                            if st + 1 == SPLIT:
                                feat = DF.features(SM.obs_for(seat, js))
                    except Exception:                          # noqa: BLE001
                        srv.close()
                        srv = SM.Serve()
                        continue
                    if feat is None:
                        continue
                    # play each candidate [base[:144]+cont] fully
                    banks = []
                    for _, cont in cands:
                        route = base[:SPLIT] + cont
                        try:
                            js = srv.cmd(f"RESET {seed}")
                            ok = True
                            for st in range(719):
                                me = SM.action_to_line(
                                    route[st] if st < len(route) else None)
                                ot = SM.action_to_line(
                                    OA(SM.obs_for(1 - seat, js)))
                                la, lb = (me, ot) if seat == 0 else (ot, me)
                                js = srv.cmd(f"STEP2 {la}\x1e{lb}")
                                if "error" in js:
                                    ok = False
                                    break
                            if not ok:
                                banks.append(-1e9)
                                continue
                            b = [float(f.get("money") or 0)
                                 for f in js["farms"]]
                            banks.append(b[seat] - b[1 - seat])
                        except Exception:                      # noqa: BLE001
                            srv.close()
                            srv = SM.Serve()
                            banks.append(-1e9)
                    if len(banks) != len(cands):
                        continue
                    y = max(range(len(banks)), key=lambda i: banks[i])
                    out.write(json.dumps({"f": feat, "y": y, "opp": on,
                                          "seed": seed, "seat": seat,
                                          "banks": banks}) + "\n")
                    out.flush()
                    n += 1
            print(f"  {on:<38} rows so far {n}", flush=True)
    srv.close()
    print(f"\n{n} training rows -> {a.out}")
    print(f"candidate names: {[c[0] for c in cands]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
