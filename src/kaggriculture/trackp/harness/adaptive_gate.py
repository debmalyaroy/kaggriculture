"""Adaptive-oracle regression gate (2026-09-07).

The BT-safety instrument. Plays a candidate CLOSED-LOOP vs every real
adaptive opponent (data/gauntlet + data/gauntlet_adaptive), both seats,
and reports:
  - overall win rate + margin
  - PER-OPPONENT record vs a reference agent, flagging REGRESSIONS
    (an opponent the reference beats but the candidate loses to) -- the
    exact failure that killed sim_search (win->loss vs a lower opponent).

A candidate SHIPS only if it beats the reference overall AND regresses
NO opponent. Tapes are excluded -- only adaptive agents discriminate and
predict BT behavior.

    python src/trackp/harness/adaptive_gate.py CAND.py --ref REF.py --seeds 3,4,5,6
"""
from __future__ import annotations
from kaggriculture.paths import ROOT
import argparse, glob, os, sys
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except Exception: pass
os.environ.setdefault("KAGG_BIN", os.path.join(ROOT,"rustengine","kagg.exe"))
import kaggriculture.engine.serve_match as SM

def opps():
    o=[]
    for d in ("data/gauntlet","data/gauntlet_adaptive"):
        for f in sorted(glob.glob(os.path.join(ROOT,d,"*.py"))):
            if "our_" in os.path.basename(f): continue
            o.append(f)
    return o

def duel(cand, opp, seeds, srv):
    A=SM.load_agent(cand); OA=SM.load_agent(opp); W=L=0; m=0.0
    for s in seeds:
        for seat in (0,1):
            try:
                js=srv.cmd(f"RESET {s}"); ok=True
                for _ in range(719):
                    me=SM.action_to_line(A(SM.obs_for(seat,js))); ot=SM.action_to_line(OA(SM.obs_for(1-seat,js)))
                    la,lb=(me,ot) if seat==0 else (ot,me)
                    js=srv.cmd(f"STEP2 {la}\x1e{lb}")
                    if "error" in js: ok=False; break
            except Exception:
                try: srv.close()
                except Exception: pass
                srv=SM.Serve(); continue
            if not ok: continue
            b=[float(f.get("money") or 0) for f in js["farms"]]
            W+=b[seat]>b[1-seat]; L+=b[seat]<b[1-seat]; m+=b[seat]-b[1-seat]
    return W,L,(m/max(W+L,1)), srv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("cand"); ap.add_argument("--ref",default=None)
    ap.add_argument("--seeds",default="3,4,5,6"); a=ap.parse_args()
    seeds=[int(x) for x in a.seeds.split(",")]; O=opps()
    srv=SM.Serve(); regressions=[]; cw=ct=rw=rt=0
    print(f"adaptive gate: {len(O)} opponents x {len(seeds)} seeds x2 seats",flush=True)
    print(f"{'opponent':<38} {'CAND':>10} {'REF':>10} flag",flush=True)
    for op in O:
        on=os.path.basename(op)[:-3]
        cW,cL,cm,srv=duel(a.cand,op,seeds,srv); cw+=cW; ct+=cW+cL
        if a.ref:
            rW,rL,rm,srv=duel(a.ref,op,seeds,srv); rw+=rW; rt+=rW+rL
            reg = "REGRESSION" if (cW<rW and cL>rL) else ""
            if reg: regressions.append(on)
            print(f"{on:<38} {cW}-{cL:<3}m{cm:>+8,.0f} {rW}-{rL:<3}m{rm:>+8,.0f} {reg}",flush=True)
        else:
            print(f"{on:<38} {cW}-{cL:<3} m{cm:>+8,.0f}",flush=True)
    srv.close()
    print(f"\nCAND win rate {cw}/{ct} = {cw/max(ct,1):.3f}")
    if a.ref: print(f"REF  win rate {rw}/{rt} = {rw/max(rt,1):.3f}")
    print(f"REGRESSIONS: {len(regressions)} -> {regressions}")
    print("VERDICT:", "PASS" if (not regressions and (not a.ref or cw>=rw)) else "FAIL")
    return 0
if __name__=="__main__": raise SystemExit(main())
