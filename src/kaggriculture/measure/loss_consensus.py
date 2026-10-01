"""Decision-point consensus: at the EXACT states where our bandit loses, what do
the 120 top agents do that we don't?

Instead of copying a whole economy, this replays a loss game and at each day in
the decisive window feeds OUR seat's exact observation to all 120 panel agents,
tallying what they DO (buy land? sell what? plant what? hire?). Comparing our
action to the consensus of the 120 reveals the specific decision we get wrong --
which may be a cheap fix (a sell/buy/plant at the right moment), not a whole
density economy. (Caveat: agents fed a mid-game obs act on it statelessly; the
consensus across 120 smooths individual state noise -- read fractions, not any
single agent.)

Usage: python src/loss_consensus.py --opp OPP.py --seed 1001 [--world W]
       [--days 14-26]
"""
from kaggriculture.paths import ROOT
import argparse, glob, importlib.util, io, json, os, subprocess, sys
from collections import defaultdict

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")
PANEL = os.path.join(ROOT, ".local/panel_full")


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("p" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    except Exception:
        return None
    finally:
        sys.stdout = old


def categorize(action):
    """Reduce an action dict to counted decision signals."""
    c = defaultdict(int)
    for o in (action.get("market") or []):
        if not o: continue
        h = o[0]
        if h == "BUY_LAND": c["BUY_LAND"] += 1
        elif h == "HIRE": c["HIRE"] += 1
        elif h == "SELL" and len(o) >= 3: c[f"SELL_{o[1]}"] += int(o[2]) if str(o[2]).lstrip('-').isdigit() else 0
        elif h == "BUY_SEED" and len(o) >= 2: c[f"BUYSEED_{o[1]}"] += 1
        elif h == "BUY_ANIMAL" and len(o) >= 2: c[f"BUYANIMAL_{o[1]}"] += 1
    # farmer + hands: production ops
    ops = [action.get("farmer") or ["PASS"]] + (action.get("hands") or [])
    for o in ops:
        if not o: continue
        h = o[0] if isinstance(o, list) else o
        if h in ("PLANT",) and len(o) >= 2: c[f"PLANT_{o[1]}"] += 1
        elif h in ("WATER", "HARVEST", "DIG", "FEED", "CARE", "BUILD_PASTURE", "BUILD_COOP",
                   "FERTILIZE", "COLLECT_FERTILIZER"): c[h] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default=os.path.join(PANEL, "lynnsakurai_farming_score_v5_timing_optimized.py"))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--days", default="14-27")
    a = ap.parse_args()
    d0, d1 = (int(x) for x in a.days.split("-"))
    opp = load(a.opp)
    print("loading 120 panel agents...", flush=True)
    agents = []
    for p in sorted(glob.glob(os.path.join(PANEL, "*.py"))):
        fn = load(p)
        if fn: agents.append((os.path.basename(p), fn))
    print(f"loaded {len(agents)} agents", flush=True)

    proc = subprocess.Popen([KAGG, "bandit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env={**os.environ, "KAGG_COMPAT": "1"})
    proc.stdin.write("RESETP\n"); proc.stdin.flush(); proc.stdout.readline()
    E = ServeEnv(); obs = E.reset(a.seed); world = None
    per_day = {}   # day -> (our_cat, {signal: n_agents_doing_it}, {signal: total})
    while not obs.get("done"):
        step = obs.get("step", 0); day = step // 24
        cv = seat_view(obs, a.seat); ov = seat_view(obs, 1 - a.seat)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = f"{sh[0]}|{sh[1]}"
        proc.stdin.write(json.dumps(cv) + "\n"); proc.stdin.flush()
        our = json.loads(proc.stdout.readline())
        # sample once per day at the day-start turn, in the decisive window
        if step % 24 == 0 and d0 <= day <= d1:
            frac = defaultdict(int); tot = defaultdict(int)
            for _, fn in agents:
                try: act = fn(dict(cv)) or {}
                except Exception: act = {}
                cc = categorize(act)
                for k, v in cc.items():
                    frac[k] += 1; tot[k] += v
            per_day[day] = (categorize(our), frac, tot, len(agents))
        obs = E.step_both(our if a.seat == 0 else opp(dict(ov)),
                          opp(dict(ov)) if a.seat == 0 else our)
    E.close(); proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.terminate()
    fm = obs["farms"]; us = fm[a.seat]["money"]; them = fm[1 - a.seat]["money"]
    print(f"\nworld={world}  result {'WIN' if us>them else 'LOSS'}  {us:.0f} vs {them:.0f}\n")
    print("For each decisive day: OUR action signals, then the DIVERGENCES -- signals")
    print("many top agents emit that WE don't (or vice-versa). Read '% agents (avg qty)'.")
    for day in sorted(per_day):
        ours, frac, tot, n = per_day[day]
        print(f"\n== day {day} ==")
        print("  OURS:", {k: v for k, v in sorted(ours.items())})
        rows = []
        keys = set(frac) | set(ours)
        for k in keys:
            pct = 100 * frac.get(k, 0) / n
            avg = tot.get(k, 0) / max(1, frac.get(k, 0))
            ourv = ours.get(k, 0)
            # divergence: top agents do it a lot but we do 0 (or far less)
            if pct >= 40 and ourv == 0:
                rows.append((pct, f"  MISS  {k:20} {pct:3.0f}% agents (avg {avg:.0f})  <- WE DO 0"))
            elif ourv > 0 and pct < 20:
                rows.append((-pct, f"  ONLYUS {k:19} we={ourv}  only {pct:.0f}% agents"))
        for _, line in sorted(rows, reverse=True):
            print(line)


if __name__ == "__main__":
    main()
