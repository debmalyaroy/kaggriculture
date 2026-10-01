"""Densify the bandit's economy tape -- OFFLINE, simulation-validated.

The frontier winners beat us in crop-demand worlds by growing ~10 MORE crops on 1
more land parcel (game_insight census: 68 plants vs our 58). Reactive/recorded
approaches failed (labor-logistics / desync). Offline we can simulate each step
and keep only legal, bank-improving additions.

STAGE 1 (this file, --probe): drive `kagg bandit` through a crop world and report
the headroom per day -- idle (PASS) hand-ops, empty & locked tiles, spare cash --
so we know WHERE and HOW MUCH we can add before constructing anything.

Usage: python src/densify_tape.py --probe [--opp OPP.py] [--seed 1001] [--world W]
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, json, os, subprocess, sys

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    finally:
        sys.stdout = old


def census(farm):
    a = p = empty = lock = 0
    for row in (farm.get("tiles") or []):
        for t in row:
            if t is None: empty += 1
            elif t == "LOCKED": lock += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k in ("COOP", "PASTURE") and t.get("animal"): a += 1
                elif k == "PLANT": p += 1
    return a, p, empty, lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--opp", default=os.path.join(ROOT, ".local/panel_full/lynnsakurai_farming_score_v5_timing_optimized.py"))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--world", default="FARMERS_MARKET|SMOOTHIE_SHOP")
    a = ap.parse_args()
    opp = load(a.opp)
    proc = subprocess.Popen([KAGG, "bandit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env={**os.environ, "KAGG_COMPAT": "1"})
    proc.stdin.write("RESETP\n"); proc.stdin.flush(); proc.stdout.readline()
    E = ServeEnv(); obs = E.reset(a.seed); world = None
    # accumulate idle hand-ops per day
    day_idle = {}; day_hands = {}
    print(" day | hands idle% | empty lock | cash   | anim/plant | land-affordable?")
    while not obs.get("done"):
        step = obs.get("step", 0); day = step // 24
        cv = seat_view(obs, 0)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = f"{sh[0]}|{sh[1]}"
        proc.stdin.write(json.dumps(cv) + "\n"); proc.stdin.flush()
        our = json.loads(proc.stdout.readline())
        # count idle hand-ops this turn (PASS)
        hs = our.get("hands") or []
        idle = sum(1 for h in hs if h and h[0] == "PASS")
        day_idle[day] = day_idle.get(day, 0) + idle
        day_hands[day] = day_hands.get(day, 0) + len(hs)
        if step % 24 == 0:
            fm = obs["farms"]; cash = fm[0]["money"]
            an, pl, em, lk = census(fm[0])
            aff = "YES" if (cash > 3000 and lk > 0) else ("(cash)" if lk > 0 else "-")
            di = day_idle.get(day - 1, 0); dh = day_hands.get(day - 1, 1)
            pct = 100 * di / max(1, dh)
            print(f"  d{day:2} | {pct:4.0f}% ({di:3}/{dh:3}) | {em:3} {lk:3} | {cash:6.0f} | {an:2}/{pl:2}     | {aff}")
        obs = E.step_both(our, opp(dict(seat_view(obs, 1))))
    E.close(); proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.terminate()
    fm = obs["farms"]
    print(f"\nworld={world}  final {fm[0]['money']:.0f} vs {fm[1]['money']:.0f}")
    tot_idle = sum(day_idle.values()); tot = sum(day_hands.values())
    print(f"TOTAL idle hand-ops: {tot_idle}/{tot} ({100*tot_idle/max(1,tot):.0f}%)")
    print("Mid-game (d10-24) idle:", sum(day_idle.get(d,0) for d in range(10,25)),
          "of", sum(day_hands.get(d,0) for d in range(10,25)))


if __name__ == "__main__":
    main()
