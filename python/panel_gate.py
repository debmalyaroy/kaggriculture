"""Public-panel gate across worlds (release-candidate check; runs on the AWS box).

    python python/panel_gate.py --cand weights/ppo/<run>/best.bin --name <candidate> [--k 12] [--per-world 2]

A RANDOM PANEL of opponents is drawn from the roster = every public agent shim in
data/winplan/field (the winplan field, ~49) + ours (v61, v61.1 Python; v62 = Rust v61.1 port on
profile 13, v62.1 = profile 19). v62, v62.1 and v63 (rl3 profile 35) are always in the panel (every RC is measured head to
head against the live pair); the rest are drawn stratified by how hard each agent has been for
v61.1 in recorded gate games (hard = beat v61.1 at least once, medium = close games, easy), with a
date-seeded RNG so a panel is reproducible.
WORLDS: real ladder seeds from the corpus index, labelled by their shop-world the same way the
gate harness does (serve GENGAME, day-0 shops), up to --per-world seeds for each of up to --worlds
worlds; every (opponent, seed) is played in BOTH seats.
The candidate (Rust agent + --policy) and the reference (Rust v61.1, profile 0) play the identical
games on the faithful harness (Python agents on the Rust serve engine; exact banks vs the official
engine). Rows stream to data/panel/<label>.jsonl (resumable).
Output data/gates/panel__<name>__<UTC>.json with per-opponent / per-world / per-stratum records and:
  checks.beats_v611        paired McNemar better, p < .05
  checks.no_worse_opponent no opponent where the candidate is significantly worse than v61.1
  checks.beats_v62         candidate scores > .5 vs v62 AND >= v61.1's score vs v62 on the same games
  checks.beats_v621        same vs v62.1
  checks.band_hard         score vs the hard stratum >= v61.1's
  checks.band_easy         losses vs the easy stratum <= v61.1's (the "never lose downward" rule)
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = RL  # one repo since the 2026-10-01 merge
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import krlenv  # noqa: E402

FIELD = os.path.join(REPO, "data", "winplan", "field")
REF_GATES = os.path.join(REPO, "data", "winplan", "gates")
OUT = os.path.join(RL, "data", "panel")
GATES = os.path.join(RL, "data", "gates")
PROFILES = os.path.join(RL, "configs", "profiles", "rl3.json")
BASE = os.path.join(RL, "configs", "bases", "v61.1")
SHIM = '''# {label}: Rust v61.1 port via the stdio bridge ({desc}); binary {sha}
import sys
sys.path.insert(0, r"{py}")
from rust_agent import RustAgent as _RustAgent
_A = _RustAgent(exe=r"{exe}", base=r"{base}", profiles=r"{profiles}", profile={profile}, policy={policy})


def agent(observation, configuration=None):
    return _A(observation, configuration)
'''


def shim(label, desc, profile=None, policy=None):
    os.makedirs(os.path.join(OUT, "shims"), exist_ok=True)
    exe = krlenv.bin("agent-stdio")
    sha = hashlib.sha256(open(exe, "rb").read()).hexdigest()[:12]
    txt = SHIM.format(label=label, desc=desc, sha=sha, py=os.path.join(RL, "python"), exe=exe, base=BASE, profiles=PROFILES,
                      profile=repr(profile), policy=repr(policy))
    p = os.path.join(OUT, "shims", f"{label}.py")
    open(p, "w").write(txt)
    return p


def roster():
    r = {f[:-3]: os.path.join(FIELD, f) for f in sorted(os.listdir(FIELD)) if f.endswith(".py")}
    for name, f in (("v61", "v61_bandit.py"), ("v61.1", "v61.1_bandit.py")):
        p = os.path.join(REPO, "agents", f)
        if os.path.exists(p):
            r[name] = p
    r["v62"] = shim("v62_p13", "profile 13 aggr_deep, as submitted 56538836", profile=13)
    r["v62.1"] = shim("v621_p19", "profile 19 ad_rsa12_l24, as submitted 56538921", profile=19)
    r["v63"] = shim("v63_p35", "rl3 profile 35 = v3 profile 71 c16_v92_term_tsell, as submitted 56567216", profile=35)
    return r


def strata(names):
    """hard / medium / easy per opponent from recorded v61.1 gate games (unknown -> medium)."""
    rec = {}
    for f in glob.glob(os.path.join(REF_GATES, "*v611*.jsonl")) + glob.glob(os.path.join(REF_GATES, "rust_v611_p0*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "gap" in r:
                rec.setdefault(r["opp"], []).append(r["gap"])
    out = {}
    for n in names:
        g = rec.get(n)
        if not g:
            out[n] = "medium"
        elif any(x < 0 for x in g):
            out[n] = "hard"
        elif sorted(g)[len(g) // 2] < 3000:
            out[n] = "medium"
        else:
            out[n] = "easy"
    return out


def draw_panel(names, strat, k, seed):
    rng = random.Random(seed)
    fixed = [n for n in ("v62", "v62.1", "v63") if n in names]
    want = {"hard": max(1, round((k - len(fixed)) * .5)), "medium": max(1, round((k - len(fixed)) * .3))}
    want["easy"] = max(1, k - len(fixed) - want["hard"] - want["medium"])
    pick = list(fixed)
    for s in ("hard", "medium", "easy"):
        pool = sorted(n for n in names if strat[n] == s and n not in pick)
        pick += rng.sample(pool, min(want[s], len(pool)))
    return pick


def world_seeds(n_worlds, per_world, seed):
    """[(world, seed)] from real ladder seeds (corpus index), labelled like harness.world_seeds."""
    import pandas as pd
    import kaggriculture.engine.serve_match as SM
    seeds = pd.read_parquet(os.path.join(RL, "data", "slim", "s1", "index", "episodes.parquet"), columns=["seed"])["seed"].dropna().astype("int64").unique().tolist()
    random.Random(seed).shuffle(seeds)
    P = chr(31).join(SM.action_to_line(None) for _ in range(719))
    srv = SM.Serve()
    by = {}
    try:
        for s in seeds[:4000]:
            js = srv.cmd("GENGAME " + str(int(s)) + chr(30) + P + chr(30) + P)
            days = js.get("days") or []
            sh = ((days[0].get("town") or {}).get("unlocked_shops") or []) if days else []
            if len(sh) < 2:
                continue
            w = f"{sh[0]}|{sh[1]}"
            if w in by or len(by) < n_worlds:
                if len(by.setdefault(w, [])) < per_world:
                    by[w].append(int(s))
            if len(by) >= n_worlds and all(len(v) >= per_world for v in by.values()):
                break
    finally:
        srv.close()
    return [(w, s) for w in sorted(by) for s in by[w]]


def play(task):
    from kaggriculture.winplan.gate import _play
    who, ours_path, opp, opp_path, world, seed, seat = task
    row = _play((ours_path, opp, opp_path, world, seed, seat))
    row["who"] = who
    return row


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="policy weights.bin")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--per-world", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--date", default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), help="panel RNG seed")
    a = ap.parse_args()
    rseed = int(hashlib.sha256(a.date.encode()).hexdigest()[:8], 16)
    R = roster()
    strat = strata(list(R))
    panel = draw_panel(list(R), strat, a.k, rseed)
    ws = world_seeds(a.worlds, a.per_world, rseed)
    cand = shim(f"cand_{hashlib.sha256(open(a.cand, 'rb').read()).hexdigest()[:10]}", f"policy {a.name}", policy=os.path.abspath(a.cand))
    ref = shim("ref_p0", "profile 0 = v61.1", profile=0)
    os.makedirs(OUT, exist_ok=True)
    label = f"{a.name}__{a.date}"
    out = os.path.join(OUT, label + ".jsonl")
    done = {}
    if os.path.exists(out):
        for ln in open(out, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                done[(r["who"], r["opp"], r["seed"], r["seat"])] = r
    tasks = [(who, p, o, R[o], w, s, seat) for who, p in (("cand", cand), ("ref", ref)) for o in panel for w, s in ws
             for seat in (0, 1) if (who, o, s, seat) not in done]
    print(f"[panel] {a.name}: panel {panel}; {len(ws)} world-seeds over {len({w for w, _ in ws})} worlds; "
          f"{len(tasks)} games to play ({len(done)} recorded); workers {a.workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as fh:
        futs = [ex.submit(play, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" not in r:
                done[(r["who"], r["opp"], r["seed"], r["seat"])] = r
            if i % 100 == 0:
                print(f"[panel] {i}/{len(tasks)} games, {time.time() - t0:.0f}s", flush=True)
    # ---- paired verdicts
    sc = lambda r: 1.0 if r["gap"] > 0 else 0.0 if r["gap"] < 0 else 0.5  # noqa: E731
    keys = sorted({(o, s, seat) for (who, o, s, seat) in done if who == "cand" and ("ref", o, s, seat) in done})
    per, bt, ct = {}, 0, 0
    for o in panel:
        ks = [k for k in keys if k[0] == o]
        c = [sc(done[("cand",) + k]) for k in ks]
        r = [sc(done[("ref",) + k]) for k in ks]
        b = sum(x > y for x, y in zip(c, r))
        w = sum(x < y for x, y in zip(c, r))
        per[o] = {"stratum": strat[o], "n": len(ks), "cand_score": sum(c) / max(1, len(c)), "ref_score": sum(r) / max(1, len(r)),
                  "cand_losses": sum(x == 0 for x in c), "ref_losses": sum(x == 0 for x in r), "better": b, "worse": w, "p": mcnemar(b, w)}
        bt, ct = bt + b, ct + w
    worlds = {}
    for (o, s, seat) in keys:
        w = done[("cand", o, s, seat)]["world"]
        d = worlds.setdefault(w, {"n": 0, "cand": 0.0, "ref": 0.0})
        d["n"] += 1
        d["cand"] += sc(done[("cand", o, s, seat)])
        d["ref"] += sc(done[("ref", o, s, seat)])
    band = {}
    for s in ("hard", "medium", "easy"):
        os_ = [o for o in panel if strat[o] == s and o not in ("v62", "v62.1", "v63")]
        n = sum(per[o]["n"] for o in os_)
        band[s] = {"opponents": os_, "n": n, "cand_score": sum(per[o]["cand_score"] * per[o]["n"] for o in os_) / max(1, n),
                   "ref_score": sum(per[o]["ref_score"] * per[o]["n"] for o in os_) / max(1, n),
                   "cand_losses": sum(per[o]["cand_losses"] for o in os_), "ref_losses": sum(per[o]["ref_losses"] for o in os_)}
    p = mcnemar(bt, ct)
    checks = {
        "beats_v611": bt > ct and p < .05,
        "no_worse_opponent": not any(v["worse"] > v["better"] and v["p"] < .05 for v in per.values()),
        "beats_v62": "v62" in per and per["v62"]["cand_score"] > .5 and per["v62"]["cand_score"] >= per["v62"]["ref_score"],
        "beats_v621": "v62.1" in per and per["v62.1"]["cand_score"] > .5 and per["v62.1"]["cand_score"] >= per["v62.1"]["ref_score"],
        "beats_v63": "v63" in per and per["v63"]["cand_score"] > .5 and per["v63"]["cand_score"] >= per["v63"]["ref_score"],
        "band_hard": band["hard"]["cand_score"] >= band["hard"]["ref_score"],
        "band_easy": band["easy"]["cand_losses"] <= band["easy"]["ref_losses"],
    }
    res = {"name": a.name, "weights": a.cand, "date": a.date, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "panel": panel, "world_seeds": ws, "games_paired": len(keys), "better": bt, "worse": ct, "p": p,
           "per_opponent": per, "per_world": worlds, "per_stratum": band, "checks": checks,
           "verdict": "PASS" if all(checks.values()) else "FAIL"}
    os.makedirs(GATES, exist_ok=True)
    json.dump(res, open(os.path.join(GATES, f"panel__{a.name}__{res['t'].replace(':', '')}.json"), "w"), indent=1)
    print(f"[panel] {a.name}: {res['verdict']} paired +{bt}/-{ct} p {p:.3g}; checks {checks}", flush=True)
    for o, v in per.items():
        print(f"   {o:<40} [{v['stratum']:<6}] cand {v['cand_score']:.3f} ref {v['ref_score']:.3f} +{v['better']}/-{v['worse']} p {v['p']:.3g}")


if __name__ == "__main__":
    main()
