"""game_insight.py -- step-by-step insight for one or a set of games.

Plays a candidate (the Rust `kagg bandit` bridge, with optional env flags, or a
.py agent) against an opponent over a seed range, and for each realized game
emits a per-day trace of BOTH seats + auto-detected leaks. Aggregates recurring
leaks over the set. This is the reusable diagnostic behind the v58 endgame work
(generalises the ad-hoc world_trace.py).

Per-day metrics (both seats): money, tile census (animals / plants / empty /
locked / weed), shed total + premium. Auto-insight flags the largest divergences
(land not unlocked, plant-count gap, endgame money gap, weeds/starvation).

Usage:
  python .local/livepool/game_insight.py \
      --cand bandit --env KAGG_COMPAT=1 \
      --opp .local/panel_full/lynnsakurai_farming_score_v5_timing_optimized.py \
      --seeds 1000-1032 [--world "FARMERS_MARKET|SMOOTHIE_SHOP"] [--turns D0-D1]
  # --cand may be "bandit"/"trackp" (kagg family) or a path to a .py agent.
"""
import argparse, glob, importlib.util, io, json, os, subprocess, sys
from collections import defaultdict

import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")
PREM = ("STRAWBERRY", "MILK", "WOOL", "MELON", "TOMATO")


def load_py(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    finally:
        sys.stdout = old


def census(farm):
    a = p = e = L = w = 0
    for row in (farm.get("tiles") or []):
        for t in row:
            if t is None: e += 1
            elif t == "LOCKED": L += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k in ("COOP", "PASTURE") and t.get("animal"): a += 1
                elif k == "PLANT": p += 1
                elif k == "WEED": w += 1
    sh = farm.get("shed") or {}
    return dict(anim=a, plant=p, empty=e, lock=L, weed=w,
                shed=sum(sh.values()), prem=sum(v for k, v in sh.items() if k in PREM))


class Cand:
    """Candidate wrapper: either a persistent kagg bridge or a .py agent."""
    def __init__(self, spec, env_over):
        self.proc = None; self.pyfn = None
        if spec in ("bandit", "trackp"):
            env = dict(os.environ); env.update(env_over)
            self.proc = subprocess.Popen([KAGG, spec], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        else:
            self.pyfn = load_py(spec)

    def reset(self):
        if self.proc:
            self.proc.stdin.write("RESETP\n"); self.proc.stdin.flush(); self.proc.stdout.readline()

    def act(self, cv):
        if self.proc:
            self.proc.stdin.write(json.dumps(cv) + "\n"); self.proc.stdin.flush()
            return json.loads(self.proc.stdout.readline())
        try:
            return self.pyfn(dict(cv)) or {"farmer": ["PASS"], "hands": [], "market": []}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

    def close(self):
        if self.proc:
            try: self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush(); self.proc.terminate()
            except Exception: pass


def play(cand, opp, seed, seat=0):
    cand.reset(); E = ServeEnv(); obs = E.reset(seed)
    world = None; days = []
    while not obs.get("done"):
        cv = seat_view(obs, seat); ov = seat_view(obs, 1 - seat)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = f"{sh[0]}|{sh[1]}"
        ca = cand.act(cv); oa = opp(dict(ov))
        step = obs.get("step", 0)
        if step % 24 == 0:
            fm = obs["farms"]
            days.append((step // 24, round(fm[seat]["money"]), round(fm[1 - seat]["money"]),
                         census(fm[seat]), census(fm[1 - seat])))
        obs = E.step_both(ca if seat == 0 else oa, oa if seat == 0 else ca)
    E.close()
    fm = obs["farms"]
    return world, round(fm[seat]["money"]), round(fm[1 - seat]["money"]), days


def insights(us, them, days):
    """Auto-detect the biggest leaks; return a list of one-line strings."""
    out = []
    if not days: return out
    # decisive day (last lead flip)
    lead = None; dec = 0
    for d, u, t, _, _ in days:
        cur = u >= t
        if lead is not None and cur != lead: dec = d
        lead = cur
    out.append(f"decisive day (last lead flip): {dec}")
    last = days[-1]
    uc, tc = last[3], last[4]
    # land: locked tiles we never unlocked vs them
    if uc["lock"] - tc["lock"] >= 6:
        out.append(f"LAND: we leave {uc['lock']} tiles LOCKED vs their {tc['lock']} "
                   f"(unlock + plant ~{uc['lock']} more tiles)")
    # sustained plant gap
    mid = [d for d in days if 18 <= d[0] <= 26]
    if mid:
        pg = sum(d[4]["plant"] - d[3]["plant"] for d in mid) / len(mid)
        if pg >= 4:
            out.append(f"CROP DENSITY: they grow {pg:+.0f} more plants (d18-26 avg)")
    # endgame money gap (last 4 days)
    if len(days) >= 5:
        d0 = days[-5]; eu = last[1] - d0[1]; et = last[2] - d0[2]
        if et - eu >= 2000:
            out.append(f"ENDGAME: they out-earn us by {et - eu:+.0f} in the last "
                       f"{last[0]-d0[0]} days (us +{eu}, them +{et})")
    # weeds / starvation
    mw = max(d[3]["weed"] for d in days); ma = min(d[3]["anim"] for d in days if d[0] >= 12) if any(d[0]>=12 for d in days) else 0
    a0 = max(d[3]["anim"] for d in days)
    if mw >= 3: out.append(f"WEEDS: up to {mw} weeds on our farm (unwatered plants)")
    if a0 - ma >= 2: out.append(f"STARVATION: our animals fell {a0}->{ma} (unfed)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="bandit")
    ap.add_argument("--opp", default=os.path.join(ROOT, ".local/panel_full/lynnsakurai_farming_score_v5_timing_optimized.py"))
    ap.add_argument("--env", action="append", default=[], help="ENV=VAL for the kagg bridge")
    ap.add_argument("--seeds", default="1000-1024")
    ap.add_argument("--world", default=None, help="only report games realizing this shop pair")
    ap.add_argument("--full", action="store_true", help="print the full per-day table (else insights only)")
    a = ap.parse_args()
    env_over = dict(kv.split("=", 1) for kv in a.env)
    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]; lo, hi = int(lo), int(hi)
    opp = load_py(a.opp)
    cand = Cand(a.cand, env_over)
    print(f"cand={a.cand} env={env_over} opp={os.path.basename(a.opp)} seeds {lo}-{hi}"
          + (f" world={a.world}" if a.world else ""), flush=True)
    agg = defaultdict(int); n_games = 0; wins = 0
    for seed in range(lo, hi + 1):
        world, us, them, days = play(cand, opp, seed)
        if a.world and world != a.world:
            continue
        n_games += 1; wins += (us > them)
        res = "WIN " if us > them else "LOSS"
        ins = insights(us, them, days)
        print(f"\nseed {seed} {world} {res} {us} vs {them} ({us-them:+})")
        for line in ins:
            print("   - " + line)
            tag = line.split(":")[0]
            if tag in ("LAND", "CROP DENSITY", "ENDGAME", "WEEDS", "STARVATION"):
                agg[tag] += 1
        if a.full:
            print("   day |  us$   them$ | a/p/e/L/w ours | a/p/e/L/w theirs")
            for d, u, t, uc, tc in days:
                if d >= 16:
                    print(f"    d{d:2} |{u:6} {t:6} | {uc['anim']:2}/{uc['plant']:2}/{uc['empty']:2}/{uc['lock']:2}/{uc['weed']:1}"
                          f"      | {tc['anim']:2}/{tc['plant']:2}/{tc['empty']:2}/{tc['lock']:2}/{tc['weed']:1}")
    cand.close()
    print(f"\n=== {n_games} games, {wins} wins ({wins/max(1,n_games):.3f}) ===")
    if agg:
        print("recurring leaks:", dict(agg))


if __name__ == "__main__":
    main()
