"""Port public agents of our chassis family to Rust: find WHERE each one differs from a Rust configuration.

19 of the top-25 public agents carry exactly v61.1's routes, router tables, settings and opening (they differ only in
their wrapper layers). SHADOW mode plays the Python agent in a real game (serve engine) against a fixed opponent and,
every step, feeds the SAME observation to a Rust agent (base + profile) whose actions are discarded; it reports where
the two actions differ (units vs market, by step), which points at the layer to port.

    python -m kaggriculture.bandit.port_public shadow --agent herd_safe_v3_experimental_risk_aware_feed [--seeds 2]
        [--rust "--profiles configs/bandit/profiles/v4.json --profile 0"] [--opp P100] [--out .local/port]
    python -m kaggriculture.bandit.port_public shadow --all [--workers 6]
    python -m kaggriculture.bandit.port_public fidelity --standins .local/port/prof.json --ids 1,2,3 [--tourn cma2t]

FIDELITY replays a Python tournament's exact games (seed, seat, candidate profile) with a Rust STAND-IN (base v61.1 +
a stand-in profile) in place of each public agent, on Rust selfplay (bank-identical to the serve harness), and reports per
agent and stand-in: outcome agreement (win/loss equal to the Python game) and paired-verdict agreement (candidate vs the
reference decided the same way). A stand-in may enter a CMA fitness only where both are high.

Writes <out>/<agent>.json: per seed, agreement rates and every differing step (python action, rust action).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
from concurrent.futures import ProcessPoolExecutor

from kaggriculture.paths import ROOT

FIELD = os.path.join(ROOT, "data", "winplan", "field")
V4 = os.path.join(ROOT, "configs", "bandit", "profiles", "v4.json")
PANEL = os.path.join(ROOT, "models", "bandit", "top25_public.json")


def agent_main(name):
    s = open(os.path.join(FIELD, f"{name}.py"), encoding="utf-8").read()
    return os.path.join(re.search(r"_d = '([^']+)'", s).group(1), "main.py")


def _norm(a):
    a = a or {}
    j = lambda x: json.loads(json.dumps(x if x is not None else [], default=list))
    return j(a.get("farmer")), j(a.get("hands")), j(a.get("market"))


def shadow_game(name, seed, seat, rust_flags, opp_flags):
    from kaggriculture.bandit.gate import loss_forensics as LF
    from kaggriculture.bandit.rust_bridge import RustBandit
    from kaggriculture.engine import serve_match as SM
    from kaggriculture.bandit.gate import harness as H
    from kaggriculture.trackp.serve_env import ServeEnv
    with contextlib.redirect_stdout(io.StringIO()):
        py = LF.load_pyagent(os.path.join(FIELD, f"{name}.py"))
    rs = RustBandit(rust_flags)
    opp = RustBandit(opp_flags)
    E = ServeEnv(kagg_path=H.FRESH_KAGG)
    obs = E.reset(seed)
    diffs, n, units_eq, mkt_eq = [], 0, 0, 0
    try:
        while not obs.get("done") and obs["step"] < SM.EPISODE_STEPS - 1:
            cv, ov = SM.obs_for(seat, obs), SM.obs_for(1 - seat, obs)
            with contextlib.redirect_stdout(io.StringIO()):
                a_py = SM._call(py, cv) or {"farmer": ["PASS"], "hands": [], "market": []}
            a_rs = rs(SM.obs_for(seat, obs))
            a_op = opp(ov) or {"farmer": ["PASS"], "hands": [], "market": []}
            p, r = _norm(a_py), _norm(a_rs)
            n += 1
            ue, me = p[:2] == r[:2], p[2] == r[2]
            units_eq += ue
            mkt_eq += me
            if not (ue and me):
                diffs.append({"step": obs["step"], "units": not ue, "market": not me, "py": a_py, "rs": a_rs})
            a0, a1 = (a_py, a_op) if seat == 0 else (a_op, a_py)
            obs = E.step_both(a0, a1)
        fin = obs["farms"]
        bank = (float(fin[seat]["money"]), float(fin[1 - seat]["money"]))
    finally:
        E.close()
        for x in (rs, opp):
            try:
                x.proc.kill()
            except Exception:  # noqa: BLE001
                pass
    return {"agent": name, "seed": seed, "seat": seat, "steps": n, "units_agree": units_eq / max(1, n), "market_agree": mkt_eq / max(1, n),
            "first_diff": diffs[0]["step"] if diffs else None, "n_diff": len(diffs), "bank": bank, "diffs": diffs}


def _job(args):
    try:
        return shadow_game(*args)
    except Exception as exc:  # noqa: BLE001
        return {"agent": args[0], "seed": args[1], "seat": args[2], "error": f"{type(exc).__name__}: {exc}"[:300]}


BASE_DIR = os.path.join(ROOT, "configs", "bandit", "bases", "v61.1")
SELFPLAY = os.path.join(ROOT, "rustengine", "v62", "target-x", "release", "selfplay.exe")


def _fid_job(args):
    import subprocess
    prof, cand_id, sid, seeds_seat = args
    out = {}
    for seat, seeds in seeds_seat.items():
        if not seeds:
            continue
        pa, pb = (cand_id, sid) if seat == 0 else (sid, cand_id)
        o = subprocess.run([SELFPLAY, "--a", BASE_DIR, "--b", BASE_DIR, "--profiles", prof, "--pa", str(pa), "--pb", str(pb),
                            "--seeds", ",".join(map(str, seeds)), "--threads", "1"], capture_output=True, text=True).stdout
        for ln in o.splitlines():
            q = ln.split("	")
            if len(q) >= 3 and q[0].lstrip("-").isdigit():
                us, them = (float(q[1]), float(q[2])) if seat == 0 else (float(q[2]), float(q[1]))
                out[(int(q[0]), seat)] = us - them
    return args[1:3], out


def fidelity(a):
    import copy
    v4 = json.load(open(V4, encoding="utf-8"))
    extra = json.load(open(a.standins, encoding="utf-8"))["profiles"]
    ids = [int(x) for x in a.ids.split(",")] if a.ids else []
    own = {}
    if a.fitted:
        # each agent's own fitted stand-in (fit_<agent>.json) as one more stand-in id, used only for that agent
        import glob
        for fp in sorted(glob.glob(os.path.join(a.out, "fit_*.json"))):
            d = json.load(open(fp, encoding="utf-8"))
            own[d["agent"]] = len(extra)
            extra.append({**d["profile"], "name": f"fit_{d['agent'][:40]}"})
    table = copy.deepcopy(v4)
    n0 = len(table["profiles"])
    table["profiles"] += extra
    prof = os.path.join(a.out, "fidelity_profiles.json")
    json.dump(table, open(prof, "w", encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "models", "bandit", "tournament", f"{a.tourn}.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if "gap" in r and (not a.agent or r["opp"] == a.agent)]
    cands = sorted({r["cand"] for r in rows})
    panel = json.load(open(PANEL))
    py = {}
    for r in rows:
        if r["opp"] in panel:
            py[(r["opp"], r["cand"], r["seed"], r["seat"])] = r["gap"]
    jobs = []
    for opp in sorted({k[0] for k in py}):
        for c in cands:
            ss = {0: sorted({k[2] for k in py if k[0] == opp and k[1] == c and k[3] == 0}), 1: sorted({k[2] for k in py if k[0] == opp and k[1] == c and k[3] == 1})}
            for i in ids + ([own[opp]] if opp in own else []):
                jobs.append((prof, int(c[1:]), n0 + i, ss, opp, c, i))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for j, (_, out) in zip(jobs, ex.map(_fid_job, [j[:4] for j in jobs])):
            opp, c, i = j[4:]
            for (seed, seat), gap in out.items():
                res[(opp, c, i, seed, seat)] = gap
    with open(os.path.join(a.out, f"fidelity_{a.tourn}_raw.jsonl"), "w", encoding="utf-8") as f:
        for (opp, c, i, seed, seat), gap in res.items():
            f.write(json.dumps({"opp": opp, "cand": c, "standin": i, "seed": seed, "seat": seat, "rust_gap": gap,
                                "py_gap": py.get((opp, c, seed, seat))}) + "\n")
    w = lambda g: 1 if g > 0 else 0 if g < 0 else 0.5
    ref = cands[0] if a.ref not in cands else a.ref
    report = {}
    for opp in sorted({k[0] for k in py}):
        for i in ids + ([own[opp]] if opp in own else []):
            agree = n = vagree = vn = 0
            for (o, c, seed, seat), g in py.items():
                if o != opp or (opp, c, i, seed, seat) not in res:
                    continue
                n += 1
                agree += w(g) == w(res[(opp, c, i, seed, seat)])
            for (o, c, seed, seat), g in py.items():
                if o != opp or c == ref or (o, ref, seed, seat) not in py:
                    continue
                k1, k0 = (opp, c, i, seed, seat), (opp, ref, i, seed, seat)
                if k1 in res and k0 in res:
                    vn += 1
                    vagree += (w(g) - w(py[(o, ref, seed, seat)])) == (w(res[k1]) - w(res[k0]))
            report[f"{opp}|{i}"] = {"outcome_agree": agree / max(1, n), "n": n, "verdict_agree": vagree / max(1, vn), "vn": vn}
            print(f"[fid] {opp[:44]:44} standin {i} ({extra[i]['name']}): outcome {agree / max(1, n):.3f} (n {n})  paired verdict {vagree / max(1, vn):.3f} (n {vn})", flush=True)
    json.dump(report, open(os.path.join(a.out, f"fidelity_{a.tourn}.json"), "w"), indent=1)


REC = os.path.join(ROOT, ".local", "port", "rec")
STAGES = ["room", "v219", "order", "v231", "r36", "r37", "v233", "r51_input", "r51_warehouse", "r85", "r95", "r97", "courier", "carrot",
          "fert", "opening", "ctrtable", "overflow", "ca", "or2", "ch", "sr", "hd2", "cs", 37, "r127", "pg", "v44y", "y", "e335",
          "e343_wl", "adv", "t62a", "pipe", "wb3", "fx", "dp", "mp", "bd", "mpx", "sm", "cxd", "e410", "e402", "mg", "ig", "rsa"]
KNOB_MOVES = [("v92_on", [True, False]), ("v92_ext_window", [0, 4, 8]), ("v92_top", [1, 3]), ("v92_k", [3, 4]), ("hfeed_on", [True, False]),
              ("ca_margin", [-15.0, -20.0, -25.0]), ("or2_slot_margin", [8.0, 12.0]), ("cxd_model", [0, 1]), ("term_on", [True, False]),
              ("tsell_on", [True, False]), ("r85_feed_on", [True, False]), ("e410_on", [True, False])]


def record(name, seed, seat, opp_flags):
    """Play the Python agent once and save (seat observation, its action) per step."""
    from kaggriculture.bandit.gate import loss_forensics as LF
    from kaggriculture.bandit.rust_bridge import RustBandit, _plain
    from kaggriculture.engine import serve_match as SM
    from kaggriculture.bandit.gate import harness as H
    from kaggriculture.trackp.serve_env import ServeEnv
    os.makedirs(REC, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        py = LF.load_pyagent(os.path.join(FIELD, f"{name}.py"))
    opp = RustBandit(opp_flags)
    E = ServeEnv(kagg_path=H.FRESH_KAGG)
    obs = E.reset(seed)
    path = os.path.join(REC, f"{name}__{seed}_{seat}.jsonl")
    try:
        with open(path, "w", encoding="utf-8") as f:
            while not obs.get("done") and obs["step"] < SM.EPISODE_STEPS - 1:
                cv, ov = SM.obs_for(seat, obs), SM.obs_for(1 - seat, obs)
                plain = _plain(SM.obs_for(seat, obs))
                with contextlib.redirect_stdout(io.StringIO()):
                    a_py = SM._call(py, cv) or {"farmer": ["PASS"], "hands": [], "market": []}
                a_op = opp(ov) or {"farmer": ["PASS"], "hands": [], "market": []}
                f.write(json.dumps({"obs": plain, "py": json.loads(json.dumps(a_py, default=list))}, separators=(",", ":")) + "\n")
                a0, a1 = (a_py, a_op) if seat == 0 else (a_op, a_py)
                obs = E.step_both(a0, a1)
    finally:
        E.close()
        opp.proc.kill()
    return path


def _rec_job(args):
    try:
        return record(*args)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {args[0]} {type(exc).__name__}: {exc}"[:200]


def _canon(m):
    import collections
    c = collections.Counter()
    for o in m or []:
        if not o:
            continue
        if o[0] in ("HIRE", "BUY_LAND"):
            c[(o[0],)] += 1
        elif len(o) >= 3:
            if int(o[2]) > 0:
                c[(o[0], o[1])] += int(o[2])
        else:
            c[tuple(o)] += 1
    return c


def replay_cost(recs, profile, tmpdir):
    """Differing steps (units, or market beyond order/slot cosmetics) of a Rust profile on recorded games."""
    import subprocess
    from kaggriculture.bandit.rust_bridge import EXE, BASE
    prof = os.path.join(tmpdir, f"p{os.getpid()}_{abs(hash(json.dumps(profile, sort_keys=True)))}.json")
    json.dump({"profiles": [{"name": "v61.1"}, profile]}, open(prof, "w"))
    total = 0
    for lines, pys in recs:
        out = subprocess.run([EXE, "--base", BASE, "--profiles", prof, "--profile", "1"], input="\n".join(lines) + "\n",
                             capture_output=True, text=True).stdout.splitlines()
        if len(out) < len(pys):
            return 10 ** 6
        for o, p in zip(out, pys):
            r = json.loads(o)
            ue = _norm(p)[:2] == _norm(r)[:2]
            me = _canon(p.get("market")) == _canon(r.get("market"))
            total += not (ue and me)
    os.remove(prof)
    return total


def _cost_job(args):
    recs, profile, tmpdir = args
    return replay_cost(recs, profile, tmpdir)


def fit(a):
    """Greedy search over stage skips and knob values minimising differing steps on the recorded games."""
    import glob
    files = sorted(glob.glob(os.path.join(REC, f"{a.agent}__*.jsonl")))
    recs = []
    for fp in files:
        lines, pys = [], []
        for l in open(fp, encoding="utf-8"):
            d = json.loads(l)
            lines.append(json.dumps(d["obs"], separators=(",", ":")))
            pys.append(d["py"])
        recs.append((lines, pys))
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    cur = json.loads(a.start) if a.start else {}
    cur.setdefault("name", "fit")
    best = replay_cost(recs, cur, tmp)
    print(f"[fit] {a.agent}: {len(recs)} games, start cost {best} {cur}", flush=True)
    for rnd in range(a.rounds):
        moves = []
        skip = list(cur.get("skip", []))
        for st in STAGES:
            ns = [x for x in skip if x != st] if st in skip else skip + [st]
            moves.append((f"skip {'-' if st in skip else '+'}{st}", {**cur, "skip": ns}))
        for k, vals in KNOB_MOVES:
            for v in vals:
                if cur.get(k) != v:
                    moves.append((f"{k}={v}", {**cur, k: v}))
        with ProcessPoolExecutor(a.workers) as ex:
            costs = list(ex.map(_cost_job, [(recs, m[1], tmp) for m in moves]))
        i = min(range(len(moves)), key=lambda j: costs[j])
        if costs[i] >= best:
            print(f"[fit] round {rnd}: no improving move (best {best})", flush=True)
            break
        best, cur = costs[i], moves[i][1]
        print(f"[fit] round {rnd}: {moves[i][0]} -> cost {best}", flush=True)
    json.dump({"agent": a.agent, "cost": best, "profile": cur, "games": len(recs), "steps": sum(len(p) for _, p in recs)},
              open(os.path.join(a.out, f"fit_{a.agent}.json"), "w"), indent=1)
    print(f"[fit] {a.agent} FINAL cost {best}: {cur}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shadow", "fidelity", "record", "fit"])
    ap.add_argument("--start", default=None, help="fit: starting profile JSON")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--standins", default=os.path.join(ROOT, ".local", "port", "prof.json"))
    ap.add_argument("--ids", default="1,2,3")
    ap.add_argument("--tourn", default="cma2t")
    ap.add_argument("--ref", default="P100")
    ap.add_argument("--fitted", action="store_true", help="fidelity: also test each agent's own fit_<agent>.json stand-in")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=820000)
    ap.add_argument("--rust", default=f"--profiles {V4} --profile 0")
    ap.add_argument("--opp", default=f"--profiles {V4} --profile 100")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "port"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "fidelity":
        return fidelity(a)
    if a.cmd == "fit":
        return fit(a)
    if a.cmd == "record":
        names = json.load(open(PANEL)) if a.all else [a.agent]
        jobs = [(n, a.seed0 + i, i % 2, shlex.split(a.opp, posix=False)) for n in names for i in range(a.seeds)]
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(_rec_job, jobs):
                print(f"[record] {r}", flush=True)
        return
    names = json.load(open(PANEL)) if a.all else [a.agent]
    jobs = [(n, a.seed0 + i, i % 2, shlex.split(a.rust, posix=False), shlex.split(a.opp, posix=False)) for n in names for i in range(a.seeds)]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, jobs):
            res.setdefault(r["agent"], []).append(r)
            if "error" in r:
                print(f"[port] {r['agent'][:44]:44} seed {r['seed']} ERROR {r['error'][:120]}", flush=True)
            else:
                print(f"[port] {r['agent'][:44]:44} seed {r['seed']} seat {r['seat']}: units {r['units_agree']:.3f} market {r['market_agree']:.3f} "
                      f"first diff step {r['first_diff']} ({r['n_diff']} steps differ) bank {r['bank'][0]:.0f}-{r['bank'][1]:.0f}", flush=True)
    for n, rs in res.items():
        json.dump(rs, open(os.path.join(a.out, f"{n}.json"), "w"), default=str)


if __name__ == "__main__":
    main()
