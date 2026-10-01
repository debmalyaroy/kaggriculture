"""Receding-horizon DAY-GRAPH optimizer: exhaustive branches per day,
engine-true evaluation (operator design 2026-09-02: per-step full
enumeration is ~10^10 branches/step — so the graph is built at day
granularity where FULL enumeration is affordable, and the real engine
evaluates every branch so all constraints are exact).

For each day d = 0..29:
  branches = every combination of quantized knobs for that day
             (sell aggression x hold shift x hires delta x feed delta)
  each branch -> write day_over[d] -> build agent -> roll FULL game on
  serve vs a small elite panel (later days at incumbent settings)
  keep argmax(mean own bank), advance to d+1.

    python src/trackp/day_opt.py [--days 30] [--opps 3]
Output: models/trackp/day_opt/best_genome.json (+ per-day log).
"""
from kaggriculture.paths import ROOT
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(ROOT, "models", "trackp", "day_opt")
WORK = os.path.join(ROOT, ".local", "day_opt")
SEED = 901

# quantized per-day knob grid (exhaustive product = 3*3*3*2 = 54 branches)
SELL_AGGR = (0.5, 1.0, 2.0)        # multiplies every sell_cap that day
HOLD_SHIFT = (-4, 0, 4)            # shifts every hold_until that day
HIRE_DELTA = (-3, 0, 3)            # adds to that day's hire count
FEED_DELTA = (0, 6)                # adds to feed_buy that day


def build(genome, out):
    gp = out + ".genome.json"
    json.dump(genome, open(gp, "w"))
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "build_econ_agent.py"),
                        "--genome", gp, "--out", out],
                       capture_output=True, text=True, timeout=120)
    return os.path.exists(out) and "wrote" in (r.stdout or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--opps", type=int, default=3)
    ap.add_argument("--base-genome", default=os.path.join(
        ROOT, "models", "trackp", "ga_panel", "best_genome.json"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    import kaggriculture.engine.serve_match as SM
    srv = SM.Serve()
    man = json.load(open(os.path.join(ROOT, ".local", "elite_panel",
                    "manifest.json"), encoding="utf-8"))
    tapes = [t["tape"] for t in man["teams"]][:: max(1, len(man["teams"])
             // args.opps)][:args.opps]

    def score(agent_path):
        tot = 0.0
        for tape in tapes:
            for seat in (0, 1):
                a, b = SM.load_agent(agent_path), SM.load_agent(tape)
                ab, tb = (SM.run_match(a, b, SEED, srv) if seat == 0
                          else SM.run_match(b, a, SEED, srv)[::-1])
                tot += ab
        return tot / (2 * len(tapes))

    g = json.load(open(args.base_genome))
    g.setdefault("day_over", {})
    cp = os.path.join(WORK, "cand.py")
    build(g, cp)
    inc = score(cp)
    print(f"incumbent mean own-bank: {inc:,.0f}", flush=True)
    log = []
    for d in range(args.days):
        t0 = time.time()
        best = (inc, None)
        for sa, hs, hd, fd in itertools.product(
                SELL_AGGR, HOLD_SHIFT, HIRE_DELTA, FEED_DELTA):
            if (sa, hs, hd, fd) == (1.0, 0, 0, 0):
                continue                      # incumbent already scored
            ov = {}
            if sa != 1.0:
                ov["sell_cap"] = {k: max(1, int(v * sa))
                                  for k, v in g["sell_cap"].items()}
            if hs:
                ov["hold_until"] = {k: max(0, min(29, v + hs))
                                    for k, v in g["hold_until"].items()}
            if hd:
                h = list(g["hires"])
                h[min(d, len(h) - 1)] = max(0, h[min(d, len(h) - 1)] + hd)
                ov["hires"] = h
            if fd:
                ov["feed_buy"] = int(g.get("feed_buy", 4)) + fd
            g["day_over"][str(d)] = ov
            if not build(g, cp):
                continue
            v = score(cp)
            if v > best[0]:
                best = (v, dict(ov))
        if best[1] is not None:
            g["day_over"][str(d)] = best[1]
            inc = best[0]
        else:
            g["day_over"].pop(str(d), None)
        json.dump(g, open(os.path.join(OUT, "best_genome.json"), "w"),
                  indent=1)
        log.append({"day": d, "bank": inc, "chose": best[1]})
        json.dump(log, open(os.path.join(OUT, "log.json"), "w"), indent=1)
        print(f"day {d}: bank {inc:,.0f} chose {best[1]} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
