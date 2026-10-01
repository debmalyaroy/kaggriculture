"""Route library v2, step A3 (lineage stage): every candidate route forced in ITS world, closed loop, paired vs the
current route, v63.5_rl as the agent.

    python python/routes/screen_routes.py [--seeds 6] [--split train] [--skip 20] [--threads 16] [--out data/routes/screen_lineage.json]

For each world with candidates: the world's bank seeds (w64_bank <split>, --seeds per world starting at --skip),
both seats, vs the 6 lineage opponents (v63, v62.1, v62, v61.1, v63.1_rl, random-knob clones; the mirror is left out:
it rewards any deviation). Reference = v63.5_rl on the same base without a route table (the current route; extra
routes are never followed, so the reference plays exactly as v61.1's library). Candidate = + --route-table {world: id}.
A game scores win 1 / draw 0.5 / loss 0 for our seat; paired better / worse and the exact sign p per candidate.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_ROUTES_BIN") or os.path.join(RL, "target-routes", "release")
SELFPLAY = os.path.join(BIN, "selfplay")
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
RUN = os.path.join(RL, "weights", "ppo", "ppo-gru64-f2-p36-from-init-20260926T0344Z")
AGENT = (f"--profiles {RL3} --policy {RUN}/snapshots/i000790.bin --shield {RL}/configs/shield/v1.json "
         f"--shell {RL}/weights/shell/big1/shell.json --chain-off r127,sm,r95")
OPPS = {"v63": f"--profiles {RL3} --pa 35", "v62.1": f"--profiles {RL3} --pa 19", "v62": f"--profiles {RL3} --pa 13",
        "v61.1": f"--profiles {RL3} --pa 0", "v63.1_rl": f"--profiles {V64} --pa 35 --group 35,35,36", "rand": "RAND"}
PLAIN = f"--profiles {RL3} --pa 0"
V611 = os.path.join(RL, "configs", "bases", "v61.1")


def sp(a_dir, a_args, b_dir, b_args, seeds, rand_seat=None):
    cmd = [SELFPLAY, "--a", a_dir, "--b", b_dir, "--seeds", ",".join(map(str, seeds)), "--threads", "1", "--a-args", a_args, "--b-args", b_args]
    if rand_seat is not None:
        cmd += ["--rand-b", "7", "--rand-seat", str(rand_seat)]
    r = subprocess.run(cmd, cwd=RL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    out = {}
    for ln in r.stdout.splitlines():
        x = ln.split("\t")
        if len(x) >= 6:
            out[int(x[0])] = (float(x[1]), float(x[2]), x[5])
    return out


def play(base, extra, seeds):
    """{(opp, seed, seat): score, world} for our agent on `base` (+extra args) from both seats."""
    agent = f"{AGENT} {extra}".strip()
    res = {}
    for name, opp in OPPS.items():
        for seat in (0, 1):
            if opp == "RAND":
                o = sp(base, agent, V611, PLAIN, seeds, 1) if seat == 0 else sp(V611, PLAIN, base, agent, seeds, 0)
            else:
                o = sp(base, agent, V611, opp, seeds) if seat == 0 else sp(V611, opp, base, agent, seeds)
            for s, (b0, b1, w) in o.items():
                m = (b0 - b1) if seat == 0 else (b1 - b0)
                res[(name, s, seat)] = (1.0 if m > 0 else 0.0 if m < 0 else 0.5, w)
    return res


def signp(b, w):
    n = b + w
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(RL, "data", "routes", "cands_stage2.json"))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--split", default="train")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma list of candidate keys")
    ap.add_argument("--out", default=os.path.join(RL, "data", "routes", "screen_lineage.json"))
    a = ap.parse_args()
    bank = json.load(open(os.path.join(RL, "data", "worlds", "w64_bank.json")))[a.split]
    cands = json.load(open(a.cands))
    if a.only:
        keep = set(a.only.split(","))
        cands = [c for c in cands if c["key"] in keep]
    worlds = sorted({c["world"] for c in cands if c.get("world") in bank})
    tmp = tempfile.mkdtemp(prefix="rtab-")
    jobs = []
    for w in worlds:
        seeds = bank[w][a.skip: a.skip + a.seeds]
        for b in sorted({c["base"] for c in cands if c["world"] == w}):
            jobs.append(("ref", w, b, None, seeds))
        for c in cands:
            if c["world"] == w:
                f = os.path.join(tmp, f"{c['route']}.json")
                json.dump({w: c["route"]}, open(f, "w"))
                jobs.append(("cand", w, c["base"], (c, f), seeds))
    print(f"[routes] {len(cands)} candidates in {len(worlds)} worlds; {len(jobs)} runs x {len(OPPS) * 2 * a.seeds} games", flush=True)

    def run(j):
        kind, w, b, cf, seeds = j
        extra = f"--route-table {cf[1]}" if cf else ""
        return j, play(os.path.join(RL, "configs", "bases", b), extra, seeds)

    refs, rows = {}, []
    with ThreadPoolExecutor(a.threads) as ex:
        results = list(ex.map(run, jobs))
    for (kind, w, b, cf, seeds), res in results:
        if kind == "ref":
            refs[(w, b)] = res
    for (kind, w, b, cf, seeds), res in results:
        if kind != "cand":
            continue
        c = cf[0]
        ref = refs[(w, b)]
        keys = [k for k in res if k in ref]
        bt = sum(res[k][0] > ref[k][0] for k in keys)
        wo = sum(res[k][0] < ref[k][0] for k in keys)
        mism = sum(res[k][1] != w for k in keys)
        rows.append({"key": c["key"], "team": c.get("team"), "world": w, "route": c["route"], "base": b, "games": len(keys),
                     "score": sum(res[k][0] for k in keys) / max(1, len(keys)), "ref_score": sum(ref[k][0] for k in keys) / max(1, len(keys)),
                     "better": bt, "worse": wo, "net": bt - wo, "p": signp(bt, wo), "world_mismatch": mism})
    rows.sort(key=lambda r: (r["world"], -r["net"]))
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"[routes] {r['world']:34s} {r['key']:16s} {r['team'] or '?':24.24s} +{r['better']}/-{r['worse']} net {r['net']:+d} "
              f"score {r['score']:.3f} vs {r['ref_score']:.3f} mism {r['world_mismatch']}", flush=True)


if __name__ == "__main__":
    main()
