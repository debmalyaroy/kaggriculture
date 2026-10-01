"""DETAILED failure forensics: WHY the bandit loses to the 5-agent wall
(tschinkel / alperen_rhythm / k0006 / nathanjacob / tetsutani).

For each (killer, seed) it drives the game on the Rust serve engine (ServeEnv,
step-aligned -- no env.steps off-by-one), capturing per step for BOTH seats:
banks, market prices+inventory, sheds, and every action. From that it computes,
without assuming anything:
  * the bank-gap trajectory day by day (WHEN the gap opens),
  * production for both seats (plant/animal/coop/pasture/land/hire/seed),
  * SELLs per product with REALIZED $/unit (price-at-step x qty) and revenue,
  * derived spend (= 3000 + revenue - final bank) = capital efficiency,
  * premium market SHARE + realized-price gap (do we glut / sell cheaper?),
  * mid-game vs endgame revenue split.
It also replays our obs stream through the isolated binary with KAGG_LOG to report
what our rails/dispatch/sells actually did in the loss.

Run: python -m kaggriculture.bandit.gate.loss_forensics --seeds 2
"""
import os, io, json, argparse, subprocess, importlib.util, sys
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
import kaggriculture.measure.eval_harness as EH
from kaggriculture.trackp.serve_env import ServeEnv, seat_view

PREMIUM = ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG"]
ALLPROD = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BASEPX = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
          "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
KILLERS = [
    ("tschinkel", os.path.join(ROOT, "agents", "pub_tschinkel_2945.py")),
    ("alperen_rhythm", os.path.join(ROOT, "agents", "pub_alperen_rhythm.py")),
    ("k0006", os.path.join(ROOT, ".local", "crown_panel", "refs", "2300-2500", "ref_k0006_2494.py")),
    ("nathanjacob", os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_nathanjacob_anticlone.py")),
    ("tetsutani", os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_tetsutani_mirror.py")),
]


def load_pyagent(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("k" + os.path.basename(p), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        # Mirror kaggle_environments.agent.get_last_callable: the ladder runs the
        # module's LAST callable by dict insertion order, which is not `agent`
        # when `agent` was re-bound after helpers were defined (2026-09-24).
        return [v for v in vars(m).values() if callable(v)][-1]
    finally:
        sys.stdout = old


def capture_game(ours, killer_path, seed, our_seat):
    """Drive our bandit vs killer on ServeEnv; return per-step records + obs stream.
    `ours` is a REUSED bandit_binary_agent (it kills+respawns its own process on
    each game's step 0, so one process is recycled -- no per-game leak)."""
    # Official-runner fidelity, same fixes as serve_match.run_match:
    #   S3 -- 2-arg agents get a structified configuration (via SM._call);
    #   S4 -- each seat gets a fresh deep-copied obs (SM.obs_for), no aliasing;
    #   S5 -- solicit actions for steps 0..718 only; the bank is read from the
    #         step-719 state (driving to `done` adds an extra step-719 action).
    from kaggriculture.engine import serve_match as SM
    opp = load_pyagent(killer_path)
    E = ServeEnv(kagg_path=H.FRESH_KAGG); obs = E.reset(seed)
    recs = []; our_obs_stream = []
    while not obs.get("done") and obs["step"] < SM.EPISODE_STEPS - 1:
        cv = SM.obs_for(our_seat, obs); ov = SM.obs_for(1 - our_seat, obs)
        our_obs_stream.append(seat_view(obs, our_seat))
        a_ours = SM._call(ours, cv) or {"farmer": ["PASS"], "hands": [], "market": []}
        a_opp = SM._call(opp, ov) or {"farmer": ["PASS"], "hands": [], "market": []}
        farms = obs["farms"]; mk = obs["market"]; priv = obs["private"]
        recs.append({
            "step": obs["step"],
            "our_money": float(farms[our_seat].get("money", 0)),
            "opp_money": float(farms[1 - our_seat].get("money", 0)),
            "prices": dict(mk.get("prices") or {}),
            "inv": dict(mk.get("inventory") or {}),
            "our_shed": dict((priv[our_seat].get("shed") or {})),
            "opp_shed": dict((priv[1 - our_seat].get("shed") or {})),
            "our_act": a_ours, "opp_act": a_opp,
        })
        a0, a1 = (a_ours, a_opp) if our_seat == 0 else (a_opp, a_ours)
        obs = E.step_both(a0, a1)
    E.close()
    fin = obs["farms"]
    return recs, our_obs_stream, float(fin[our_seat]["money"]), float(fin[1 - our_seat]["money"])


def econ(recs, who):
    """Production + EST-FILLED sells for seat 'who'. Fixes the requested-vs-filled
    inflation: an oversized SELL (e.g. the endgame qty-1000 dump into an empty
    shed) requests 1000 but fills only what the shed holds. We cap each SELL to
    the shed at that step (decrementing within-step for repeats), value it at the
    obs price at the sale step, and count how much filled volume sold BELOW base
    (selling into a declining/glutted market -- the scarcity-timing test)."""
    ak = f"{who}_act"; sk = f"{who}_shed"
    plant = defaultdict(int); animal = defaultdict(int); build = defaultdict(int)
    hire = 0; land = 0
    filled = defaultdict(float); realized = defaultdict(float)
    below = defaultdict(float); filled_end = defaultdict(float)
    for r in recs:
        a = r[ak] or {}
        shed = r.get(sk) or {}
        for h in (a.get("hands") or []):
            if not h: continue
            if h[0] == "PLANT" and len(h) >= 2: plant[h[1]] += 1
            elif h[0] == "BUILD_COOP": build["COOP"] += 1
            elif h[0] == "BUILD_PASTURE": build["PASTURE"] += 1
        f = a.get("farmer") or []
        if f and f[0] == "PLANT" and len(f) >= 2: plant[f[1]] += 1
        sold_this = defaultdict(int)
        for o in (a.get("market") or []):
            if not o: continue
            op = o[0]
            if op == "SELL" and len(o) >= 3:
                try: req = int(o[2])
                except (ValueError, TypeError): req = 0
                avail = int(shed.get(o[1], 0) or 0) - sold_this[o[1]]
                q = max(0, min(req, avail))
                if q <= 0: continue
                sold_this[o[1]] += q
                px = float(r["prices"].get(o[1], 0) or 0) or BASEPX.get(o[1], 1)
                filled[o[1]] += q; realized[o[1]] += q * px
                if px < BASEPX.get(o[1], 1): below[o[1]] += q
                if r["step"] >= 648: filled_end[o[1]] += q
            elif op == "HIRE": hire += 1
            elif op == "BUY_LAND": land += 1
            elif op == "BUY_ANIMAL" and len(o) >= 2:
                try: animal[o[1]] += int(o[2]) if len(o) > 2 and str(o[2]).lstrip('-').isdigit() else 1
                except (ValueError, TypeError): animal[o[1]] += 1
    return dict(plant=plant, animal=animal, build=build, hire=hire, land=land,
                filled=filled, realized=realized, below=below, filled_end=filled_end)


def kagg_log(stage, obs_stream):
    """Replay our obs stream through the binary with KAGG_LOG -> (dispatch summary, rail firing, sells).

    KAGG_LOG=dispatch,rails,sells emits thousands of stderr lines (>64 KB). A
    stderr=PIPE read only AT THE END DEADLOCKS: the OS pipe buffer fills, the
    binary blocks writing stderr, and our stdout.readline() blocks waiting for a
    reply that can never come. Redirect stderr to a FILE (no buffer limit)."""
    env2 = dict(os.environ); env2["KAGG_LOG"] = "dispatch,rails,sells"
    exe = os.path.join(stage, "kagg.exe")
    errpath = os.path.join(stage, "_kagglog.txt")
    with open(errpath, "w", encoding="utf-8") as errfh:
        p = subprocess.Popen([exe, "mbandit", "config.json", "branches.json", "base.tape"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfh,
                             text=True, encoding="utf-8", bufsize=1, cwd=stage, env=env2)
        for o in obs_stream:
            p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush(); p.stdout.readline()
        p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait()
    err = open(errpath, encoding="utf-8").read()
    disp_hits = sum(1 for l in err.splitlines() if l.startswith("LOG dispatch ") and "hit=1" in l)
    disp_fires = sum(1 for l in err.splitlines() if l.startswith("LOG dispatch step="))
    rail_chg = defaultdict(int); rail_noop = defaultdict(int)
    for l in err.splitlines():
        if l.startswith("LOG rail "):
            parts = dict(kv.split("=", 1) for kv in l.split() if "=" in kv)
            nm = parts.get("name", "?")
            if "no-op" in l: rail_noop[nm] += 1
            elif "CHANGED" in l: rail_chg[nm] += 1
    sells_steps = sum(1 for l in err.splitlines() if l.startswith("LOG sells "))
    return disp_hits, disp_fires, dict(rail_chg), sells_steps


PREM = ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG"]


def newacc():
    keys = ("filled", "realized", "below", "filled_end", "plant", "animal", "build")
    d = {k: defaultdict(float) for k in keys}
    d["hire"] = 0.0; d["land"] = 0.0; d["n"] = 0
    return d


def acc_add(dst, e):
    for k in ("filled", "realized", "below", "filled_end", "plant", "animal", "build"):
        for p, v in e[k].items():
            dst[k][p] += v
    dst["hire"] += e["hire"]; dst["land"] += e["land"]; dst["n"] += 1


def print_killer(kn, gaps, ours, them):
    n = len(gaps)
    print(f"\n{'='*88}", flush=True)
    print(f"{kn}   mean gap={np.mean(gaps):+.0f}  over {n} seed(s)   per-seed={[round(g) for g in gaps]}", flush=True)
    print(f"{'='*88}", flush=True)
    print(f"  {'product':11s}|  OUR fill  $/u  bel% |  THEIR fill  $/u  bel% |  $/u gap", flush=True)
    for p in ALLPROD:
        of = ours["filled"].get(p, 0); orv = ours["realized"].get(p, 0); ob = ours["below"].get(p, 0)
        kf = them["filled"].get(p, 0); krv = them["realized"].get(p, 0); kb = them["below"].get(p, 0)
        if of == 0 and kf == 0:
            continue
        oa = orv / of if of else 0; ka = krv / kf if kf else 0
        obp = 100 * ob / of if of else 0; kbp = 100 * kb / kf if kf else 0
        star = " *" if p in PREM else "  "
        print(f"  {p:9s}{star}| {of/n:8.0f} {oa:5.1f} {obp:4.0f}% | {kf/n:9.0f} {ka:5.1f} {kbp:4.0f}% | {oa-ka:+7.1f}", flush=True)
    opf = sum(ours["filled"][p] for p in PREM); opb = sum(ours["below"][p] for p in PREM)
    kpf = sum(them["filled"][p] for p in PREM); kpb = sum(them["below"][p] for p in PREM)
    print(f"  PREMIUM(*) sold BELOW base:  OURS {100*opb/opf if opf else 0:.0f}% of {opf/n:.0f}/game"
          f"   |   THEIRS {100*kpb/kpf if kpf else 0:.0f}% of {kpf/n:.0f}/game", flush=True)
    oend = sum(ours["filled_end"].values()); oall = sum(ours["filled"].values())
    kend = sum(them["filled_end"].values()); kall = sum(them["filled"].values())
    print(f"  ENDGAME(>=d27) filled share: OURS {100*oend/oall if oall else 0:.0f}%   |   THEIRS {100*kend/kall if kall else 0:.0f}%", flush=True)
    print(f"  PRODUCTION/game: ours plant={dict((k,round(v/n)) for k,v in ours['plant'].items())} "
          f"animals={dict((k,round(v/n)) for k,v in ours['animal'].items())} hire={ours['hire']/n:.0f}", flush=True)


stage_g = None
def main():
    global stage_g
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=4); a = ap.parse_args()
    ws = H.world_seeds(a.seeds)
    cfg, _ = VL.COMBOS["sweep15"]
    H.build_agent("lf_optimal", cfg)
    stage_g = os.path.join(H.SCRATCH, "lf_optimal_wingate")
    ours = EH.bandit_binary_agent(stage_g)          # built once, recycled per game
    print(f"LOSS FORENSICS  config=sweep15  killers={[k for k,_ in KILLERS]}  worlds={len(ws)} (seat 0; seats bit-identical)", flush=True)
    results = []
    try:
        for kn, kp in KILLERS:
            if not os.path.exists(kp):
                continue
            gaps = []; oacc = newacc(); kacc = newacc()
            for wn, seed in ws:
                recs, ob, us, them = capture_game(ours, kp, seed, 0)
                gaps.append(us - them)
                acc_add(oacc, econ(recs, "our")); acc_add(kacc, econ(recs, "opp"))
                print(f"  [{kn} seed={seed}] gap={us-them:+.0f}", flush=True)
            print_killer(kn, gaps, oacc, kacc)
            results.append((kn, float(np.mean(gaps))))
    finally:
        try:
            import subprocess as _sp
            _sp.run(["powershell", "-NoProfile", "-Command",
                     "Get-Process kagg -ErrorAction SilentlyContinue | "
                     "Where-Object { $_.Path -like '*lf_optimal_wingate*' } | "
                     "Stop-Process -Force"], capture_output=True)
        except Exception:
            pass
    print(f"\n{'#'*60}\nMEAN GAP by killer ({len(ws)} seeds)\n{'#'*60}", flush=True)
    for kn, g in sorted(results, key=lambda r: r[1]):
        print(f"  {kn:16s} {g:+.0f}", flush=True)


if __name__ == "__main__":
    main()
