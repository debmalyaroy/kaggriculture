"""Ablation harness (2026-09-07): measure each feature flag's contribution.

Builds the agent with all flags ON (baseline), then one variant per flag
with that flag OFF, and plays each CLOSED-LOOP vs the real sparring
gauntlet (both seats). Reports the per-flag delta in win-rate and mean
margin: a flag whose OFF variant does BETTER is "taking us down"; a flag
whose OFF variant does WORSE is "making us strong". The operator's
which-helps/which-hurts ledger.

    python src/trackp/harness/ablate.py --gauntlet data/gauntlet \
        --seeds 3,4,5,6 --out models/trackp/ablation.json
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import glob
import json
import os
import subprocess
import sys

HARN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARN)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import feature_flags as FF                                    # noqa: E402
import kaggriculture.engine.serve_match as SM  # noqa: E402

WORK = os.path.join(ROOT, ".local", "ablate")


def build(chassis, out, flags):
    fp = os.path.join(WORK, "flags.json")
    json.dump(flags, open(fp, "w"))
    r = subprocess.run(
        [sys.executable, os.path.join(HARN, "build_v461_full.py"),
         "--chassis", chassis, "--out", out, "--flags", fp],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build failed: {r.stderr[-400:]}")
    return out


class _Srv:
    """Crash-resilient serve wrapper: recreates the subprocess on any
    failure (memory contention can kill it mid-game)."""
    def __init__(self):
        self.s = SM.Serve()

    def reset(self):
        try:
            self.s.close()
        except Exception:                                      # noqa: BLE001
            pass
        self.s = SM.Serve()

    def cmd(self, line):
        return self.s.cmd(line)

    def close(self):
        try:
            self.s.close()
        except Exception:                                      # noqa: BLE001
            pass


def gate(cand, opps, seeds, srv):
    A = SM.load_agent(cand)
    W = L = 0
    msum = 0.0
    for op in opps:
        try:
            OA = SM.load_agent(op)
        except Exception:                                      # noqa: BLE001
            continue
        for s in seeds:
            for seat in (0, 1):
                try:
                    js = srv.cmd(f"RESET {s}")
                    ok = True
                    for _ in range(719):
                        me = SM.action_to_line(A(SM.obs_for(seat, js)))
                        ot = SM.action_to_line(
                            OA(SM.obs_for(1 - seat, js)))
                        la, lb = (me, ot) if seat == 0 else (ot, me)
                        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
                        if "error" in js:
                            ok = False
                            break
                except Exception:                              # noqa: BLE001
                    srv.reset()                                # serve crashed
                    continue
                if not ok:
                    continue
                b = [float(f.get("money") or 0) for f in js["farms"]]
                mine, opp = b[seat], b[1 - seat]
                W += mine > opp
                L += mine < opp
                msum += mine - opp
    n = max(W + L, 1)
    return {"w": W, "l": L, "score": W / n, "margin": msum / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chassis", default=os.path.join(
        ROOT, ".local", "candidates", "v461_race.py"))
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "data",
                                                       "gauntlet"))
    ap.add_argument("--seeds", default="3,4,5,6")
    ap.add_argument("--flags", nargs="*", default=None,
                    help="subset of flags to ablate (default: all pipeline)")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "ablation.json"))
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    opps = sorted(g for g in glob.glob(os.path.join(a.gauntlet, "*.py"))
                  if "our_" not in os.path.basename(g))
    base_flags = FF.default_flags()
    to_ablate = a.flags or list(FF.PIPELINE_LAYERS.values())
    srv = _Srv()
    try:
        base = build(a.chassis, os.path.join(WORK, "base.py"), base_flags)
        b = gate(base, opps, seeds, srv)
        print(f"BASELINE (all ON): {b['w']}-{b['l']} "
              f"score {b['score']:.3f} margin {b['margin']:,.0f}", flush=True)
        rows = []
        for flag in to_ablate:
            f2 = copy.deepcopy(base_flags)
            f2[flag] = False
            v = build(a.chassis, os.path.join(WORK, f"off_{flag}.py"), f2)
            r = gate(v, opps, seeds, srv)
            dscore = r["score"] - b["score"]
            dmargin = r["margin"] - b["margin"]
            # OFF better than baseline => flag HURTS (negative contribution)
            verdict = ("HURTS" if dmargin > 250 else
                       ("HELPS" if dmargin < -250 else "neutral"))
            rows.append({"flag": flag, "off_score": r["score"],
                         "off_margin": r["margin"], "d_score": dscore,
                         "d_margin": dmargin, "verdict": verdict})
            print(f"  OFF {flag:<20} {r['w']}-{r['l']} "
                  f"margin {r['margin']:>+9,.0f}  d {dmargin:>+8,.0f}  "
                  f"{verdict}", flush=True)
    finally:
        srv.close()
    rows.sort(key=lambda x: x["d_margin"])   # most-helpful first
    json.dump({"baseline": b, "flags": rows}, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    print("HELPS (removing hurts us):",
          [r["flag"] for r in rows if r["verdict"] == "HELPS"])
    print("HURTS (removing helps us):",
          [r["flag"] for r in rows if r["verdict"] == "HURTS"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
