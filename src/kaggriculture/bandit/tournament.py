"""Bandit tournament: knob candidates of the v62 Rust bandit vs ALL public agents + our live agents,
across the 24 worlds, both seats, paired against the reference (v62.1 = --profile 19).

    python -m kaggriculture.bandit.tournament --name v622 [--cands C2_end13,C3_end13_jit1] [--workers N]

Opponents: models/bandit/roster.json (every public agent that plays a full game) + v61, v61.1 (Python),
v62 (Rust bandit profile 13), v62.1 (profile 19). Worlds: one real ladder seed per shop-world, labelled
like the gate harness (serve GENGAME, day-0 shops). Games are played on the faithful harness
(kaggriculture.winplan.gate._play: Python agents on the Rust serve engine, exact vs the official engine).
Rows stream to models/bandit/tournament/<name>.jsonl (resumable). Report: models/bandit/tournament/<name>.json
with, per candidate: score, paired better/worse vs the reference + exact McNemar p, per opponent,
per world, head-to-head vs v62 / v62.1, and the release-candidate verdict:
  RC = paired significantly better (p < .05) AND no opponent significantly worse AND score vs v62 and
       vs v62.1 >= the reference's.
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from kaggriculture.paths import ROOT

OUT = os.path.join(ROOT, "models", "bandit", "tournament")
SHIMS = os.path.join(OUT, "shims")
CANDS = {
    "C0_p19": ["--profile", "19"],
    "C1_jit_copy1": ["--profile", "19", "--group", "19,19,19", "--jitter", "0,0,1"],
    "C2_end13": ["--profile", "19", "--group", "19,19,19", "--endgame", "13,19,13"],
    "C3_end13_jit1": ["--profile", "19", "--group", "19,19,19", "--endgame", "13,19,13", "--jitter", "0,0,1"],
    "C6_end13_jitend2": ["--profile", "19", "--group", "19,19,19", "--endgame", "13,19,13", "--jitter", "0,0,0", "--jitter-end", "0,0,2"],
    # D24 switch against DIFFERENT opponents only; COPY/PARTIAL stay on p19 (v622 run 1: every C2 loss was the v62.1 mirror)
    "C7_diff13": ["--profile", "19", "--group", "19,19,19", "--endgame", "13,19,19"],
    "C8_diff13_jit1": ["--profile", "19", "--group", "19,19,19", "--endgame", "13,19,19", "--jitter", "0,0,1"],
    "C9_diff12": ["--profile", "19", "--group", "19,19,19", "--endgame", "12,19,19"],
    "C10_diff15": ["--profile", "19", "--group", "19,19,19", "--endgame", "15,19,19"],
    # D24 sweep (.local/v622_sweep.log): only p31 (= p19 + anti-front-run) is not worse vs reactive copies
    # (+4/-0, all vs front-runners); exact D23 mirrors keep p19 (--mirror-tol 0)
    "C11_end31_mirror": ["--profile", "19", "--group", "19,19,19", "--endgame", "31,31,31", "--mirror-tol", "0"],
    # herd_safe ca25's CARROT2 margin (-25 vs v61.1's -20): self-play vs reactive lineage 0.902 vs 0.880,
    # paired +106/-48 p 3e-6; real ladder games neutral (+6/-5). profiles v3 id 32 = p19 + ca_margin -25.
    "C12_ca25": ["--profiles", os.path.join(ROOT, "configs", "bandit", "profiles", "v3.json"), "--profile", "32"],
    # v92 rival-sales predictor (herd_safe ca25, shepherds 4-turn confidence-gated window), ported to Rust
    # bank-exact. Ablation: the piece of ca25 that beats v62.1 (p .002). Screens: real games +18/-7 (p .04),
    # self-play vs reactive lineage 0.928 vs 0.880 (+178/-57). profiles v3 id 48 = p19 + v92_on, ext_window 4.
    "C13_v92ext": ["--profiles", os.path.join(ROOT, "configs", "bandit", "profiles", "v3.json"), "--profile", "48"],
    # C13 + end-game sale-timing search (rival = worst of copy and v92 forecast, act only for >= $50):
    # self-play vs C13 +10/-0 (p .002), real games identical. profiles v3 id 70.
    "C14_v92ext_tsell": ["--profiles", os.path.join(ROOT, "configs", "bandit", "profiles", "v3.json"), "--profile", "70"],
    # C13 + terminal planner 512 sims / 2 passes / 8 proposals + C14's sale search: self-play vs C13 +26/-0
    # (p 3e-8), 0.942 vs 0.928; real games +1/-0. profiles v3 id 71.
    "C16_v92_term_tsell": ["--profiles", os.path.join(ROOT, "configs", "bandit", "profiles", "v3.json"), "--profile", "71"],
}
V4 = os.path.join(ROOT, "configs", "bandit", "profiles", "v4.json")
# v63 (live since 2026-09-26) = v3 id 71 = v4 id 71; "P<id>" names any v4 profile (v63.1 / v64 levers, dispatch sets)
CANDS["V63"] = ["--profiles", V4, "--profile", "71"]
REF = "C0_p19"
SHIM = '''# v62 Rust bandit, knobs: {flags}
from kaggriculture.bandit.rust_bridge import RustBandit as _R
_A = _R({flags!r})


def agent(observation, configuration=None):
    return _A(observation, configuration)
'''


def cand_flags(c):
    if c in CANDS:
        return CANDS[c]
    if c.startswith("P") and c[1:].isdigit():
        return ["--profiles", V4, "--profile", c[1:]]
    raise KeyError(c)


def shim(name, flags):
    os.makedirs(SHIMS, exist_ok=True)
    p = os.path.join(SHIMS, f"{name}.py")
    open(p, "w").write(SHIM.format(flags=list(flags)))
    return p


def opponents():
    ro = json.load(open(os.path.join(ROOT, "models", "bandit", "roster.json")))["runnable"]
    opp = {r["name"]: os.path.join(ROOT, r["path"]) for r in ro}
    opp["v61"] = os.path.join(ROOT, "agents", "v61_bandit.py")
    opp["v61.1"] = os.path.join(ROOT, "agents", "v61.1_bandit.py")
    opp["v62"] = shim("live_v62_p13", ["--profile", "13"])
    opp["v62.1"] = shim("live_v621_p19", ["--profile", "19"])
    opp["v63"] = shim("live_v63_p71", ["--profiles", V4, "--profile", "71"])
    return opp


def world_seeds(n_worlds=24, seed=0):
    import pandas as pd
    import kaggriculture.engine.serve_match as SM
    ix = os.path.join(ROOT, "data", "slim", "s1", "index", "episodes.parquet")
    seeds = pd.read_parquet(ix, columns=["seed"])["seed"].dropna().astype("int64").unique().tolist()
    random.Random(seed).shuffle(seeds)
    P = chr(31).join(SM.action_to_line(None) for _ in range(719))
    srv, by = SM.Serve(), {}
    try:
        for s in seeds[:3000]:
            js = srv.cmd("GENGAME " + str(int(s)) + chr(30) + P + chr(30) + P)
            days = js.get("days") or []
            sh = ((days[0].get("town") or {}).get("unlocked_shops") or []) if days else []
            if len(sh) >= 2 and f"{sh[0]}|{sh[1]}" not in by:
                by[f"{sh[0]}|{sh[1]}"] = int(s)
            if len(by) >= n_worlds:
                break
    finally:
        srv.close()
    return sorted(by.items())


def play(task):
    from kaggriculture.winplan.gate import _play
    cand, cpath, opp, opath, world, seed, seat = task
    # the candidate's D6 decision (pair, opponent cluster, dispatch) via KAGG_DISPATCH_LOG
    log = os.path.join(OUT, "dlog", f"{os.getpid()}.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    if os.path.exists(log):
        os.remove(log)
    os.environ["KAGG_DISPATCH_LOG"] = log
    r = _play((cpath, opp, opath, world, seed, seat))
    r["cand"] = cand
    try:
        d = [json.loads(l) for l in open(log, encoding="utf-8")]
        mine = [x for x in d if x.get("player") == seat] or d
        if mine:
            r.update(cluster=mine[0]["cluster"], pair=mine[0]["pair"], d_route=mine[0].get("route"), d_rule=mine[0].get("rule"))
    except OSError:
        pass
    return r


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def report(rows, cands, name):
    sc = lambda r: 1.0 if r["gap"] > 0 else 0.0 if r["gap"] < 0 else 0.5  # noqa: E731
    by = {(r["cand"], r["opp"], r["seed"], r["seat"]): sc(r) for r in rows if "gap" in r}
    ref = {k[1:]: v for k, v in by.items() if k[0] == REF}
    out = {"name": name, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reference": REF, "candidates": {}}
    for c in cands:
        cr = {k[1:]: v for k, v in by.items() if k[0] == c}
        keys = sorted(set(cr) & set(ref), key=str)
        if not keys:
            continue
        b = sum(cr[k] > ref[k] for k in keys)
        w = sum(cr[k] < ref[k] for k in keys)
        per = {}
        for o in sorted({k[0] for k in keys}):
            ko = [k for k in keys if k[0] == o]
            bo = sum(cr[k] > ref[k] for k in ko)
            wo = sum(cr[k] < ref[k] for k in ko)
            per[o] = {"n": len(ko), "cand": sum(cr[k] for k in ko) / len(ko), "ref": sum(ref[k] for k in ko) / len(ko),
                      "better": bo, "worse": wo, "p": mcnemar(bo, wo)}
        worse = [o for o, v in per.items() if v["worse"] > v["better"] and v["p"] < .05]
        h2h = {o: per[o] for o in ("v62", "v62.1", "v63") if o in per}
        rc = (c != REF and b > w and mcnemar(b, w) < .05 and not worse
              and all(v["cand"] >= v["ref"] for v in h2h.values()))
        out["candidates"][c] = {"games": len(keys), "score": sum(cr[k] for k in keys) / len(keys),
                                "ref_score": sum(ref[k] for k in keys) / len(keys), "better": b, "worse": w,
                                "p": mcnemar(b, w), "worse_opponents": worse, "vs_live": h2h,
                                "release_candidate": rc, "per_opponent": per}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v622")
    ap.add_argument("--cands", default="C2_end13,C3_end13_jit1")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 4))
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--ref", default="C0_p19", help="reference candidate (V63 = the live v63)")
    ap.add_argument("--opps", default=None, help="JSON list of opponent names to keep (roster names and live v61..v63)")
    ap.add_argument("--only-worlds", default=None, help="comma list of shop pairs A|B to play (default: every picked world)")
    ap.add_argument("--world-seed", type=int, default=0, help="shuffle seed of the world-seed pick (1+ = a fresh seed per world)")
    a = ap.parse_args()
    global REF
    REF = a.ref
    cands = [REF] + [c for c in a.cands.split(",") if c and c != REF]
    paths = {c: shim(c, cand_flags(c)) for c in cands}
    opp = opponents()
    if a.opps:
        keep = set(json.load(open(a.opps)))
        opp = {k: v for k, v in opp.items() if k in keep}
    ws = world_seeds(a.worlds, a.world_seed)
    if a.only_worlds:
        ws = [w for w in ws if w[0] in set(a.only_worlds.split(","))]
    os.makedirs(OUT, exist_ok=True)
    jl = os.path.join(OUT, f"{a.name}.jsonl")
    rows = [json.loads(l) for l in open(jl, encoding="utf-8")] if os.path.exists(jl) else []
    done = {(r["cand"], r["opp"], r["seed"], r["seat"]) for r in rows if "error" not in r}
    # cell-major (every candidate plays a cell back to back) so paired results accrue from the start
    tasks = [(c, paths[c], o, op, w, s, seat) for w, s in ws for o, op in opp.items() for seat in (0, 1) for c in cands
             if (c, o, s, seat) not in done]
    print(f"[tournament] {a.name}: candidates {cands}; {len(opp)} opponents x {len(ws)} worlds x 2 seats; "
          f"{len(tasks)} games to play ({len(done)} recorded); workers {a.workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(jl, "a", encoding="utf-8") as fh:
        futs = [ex.submit(play, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            if i % 200 == 0:
                rep = report(rows, cands, a.name)
                json.dump(rep, open(os.path.join(OUT, f"{a.name}.json"), "w"), indent=1)
                el = time.time() - t0
                print(f"[tournament] {i}/{len(tasks)} games, {el / 60:.0f} min, ETA {el / i * (len(tasks) - i) / 60:.0f} min; "
                      + "; ".join(f"{c} {v['score']:.3f} vs ref {v['ref_score']:.3f} (+{v['better']}/-{v['worse']})"
                                  for c, v in rep["candidates"].items() if c != REF), flush=True)
    rep = report(rows, cands, a.name)
    json.dump(rep, open(os.path.join(OUT, f"{a.name}.json"), "w"), indent=1)
    for c, v in rep["candidates"].items():
        print(f"[tournament] {c}: score {v['score']:.3f} vs ref {v['ref_score']:.3f}; paired +{v['better']}/-{v['worse']} "
              f"p {v['p']:.3g}; worse opponents {v['worse_opponents'] or 'none'}; vs live "
              f"{ {o: round(x['cand'], 2) for o, x in v['vs_live'].items()} }; RELEASE CANDIDATE: {v['release_candidate']}", flush=True)


if __name__ == "__main__":
    main()
