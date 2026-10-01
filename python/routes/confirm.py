"""Route library v2, step A4: confirm screened routes on HELD-OUT bank seeds + the real-player band gate.

    python python/routes/confirm.py --keys K1,K2 --cands data/routes/cands_stage2.json[,data/routes/cands_b_stage2.json]
        [--threads 16] [--out data/routes/confirm.json]

  held-out lineage  each candidate's world: the bank's held-out seeds (3) + train seeds 200.. (fresh, 6) from both
                    seats vs the 6 lineage opponents, paired vs the current route (screen_routes.play)
  band              the 991-tape real-player gate with ALL candidates' world overrides together (one route table),
                    for the candidate table and for v63.5_rl, both vs v63.1_rl
Writes data/routes/confirm.json and data/routes/lib_v2_candidate.json (the combined table of the keys given).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_routes as S  # noqa: E402


def band(extra, name, threads):
    pol = os.path.join(S.RUN, "snapshots", "i000790.bin")
    r = subprocess.run([sys.executable, os.path.join(S.RL, "python", "band_gate.py"), "--cand", pol,
                        "--cand-args", f"--shell {S.RL}/weights/shell/big1/shell.json --chain-off r127,sm,r95 {extra}".strip(), "--name", name,
                        "--ref-profiles", S.V64, "--ref-profile", "35", "--ref-group", "35,35,36", "--threads", str(threads)],
                       cwd=S.RL, capture_output=True, text=True, env={**os.environ, "KRL_BIN": S.BIN})
    line = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", line)
    return {"line": line, "losses_below_2500": int(m.group(1)) if m else None, "rate_2500plus": float(m.group(2)) if m else None,
            "paired_vs_v631": [int(m.group(3)), int(m.group(4))] if m else None, "stderr": r.stderr[-300:] if not m else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--fresh", type=int, default=6, help="fresh train seeds per world (from index --fresh-from)")
    ap.add_argument("--fresh-from", type=int, default=20, help="first train-seed index (the screens used 100..105; worlds have 131+ seeds)")
    ap.add_argument("--no-band", action="store_true")
    ap.add_argument("--out", default=os.path.join(S.RL, "data", "routes", "confirm.json"))
    a = ap.parse_args()
    keys = a.keys.split(",")
    allc = [c for f in a.cands.split(",") for c in json.load(open(f))]
    cands = [next(c for c in allc if c["key"] == k) for k in keys]
    bases = {c["base"] for c in cands}
    assert len(bases) == 1, f"one base per table: {bases}"
    base = os.path.join(S.RL, "configs", "bases", bases.pop())
    bank = json.load(open(os.path.join(S.RL, "data", "worlds", "w64_bank.json")))
    tmp = tempfile.mkdtemp(prefix="rconf-")
    jobs = []
    for c in cands:
        seeds = bank["heldout"][c["world"]][:3] + bank["train"][c["world"]][a.fresh_from:a.fresh_from + a.fresh]
        f = os.path.join(tmp, f"{c['route']}.json")
        json.dump({c["world"]: c["route"]}, open(f, "w"))
        jobs.append((c, "", seeds))
        jobs.append((c, f"--route-table {f}", seeds))
    with ThreadPoolExecutor(min(a.threads, len(jobs))) as ex:
        res = list(ex.map(lambda j: (j, S.play(base, j[1], j[2])), jobs))
    rep = {"routes": {}}
    for c in cands:
        ref = next(r for (cc, e, _), r in res if cc is c and not e)
        cnd = next(r for (cc, e, _), r in res if cc is c and e)
        ks = [k for k in cnd if k in ref]
        b = sum(cnd[k][0] > ref[k][0] for k in ks)
        w = sum(cnd[k][0] < ref[k][0] for k in ks)
        rep["routes"][c["key"]] = {"world": c["world"], "route": c["route"], "games": len(ks), "better": b, "worse": w,
                                   "score": sum(cnd[k][0] for k in ks) / len(ks), "ref_score": sum(ref[k][0] for k in ks) / len(ks), "p": S.signp(b, w),
                                   "world_mismatch": sum(cnd[k][1] != c["world"] for k in ks)}
        print(f"[confirm] {c['world']:34s} {c['key']:16s} held-out lineage +{b}/-{w} score {rep['routes'][c['key']]['score']:.3f} vs {rep['routes'][c['key']]['ref_score']:.3f}", flush=True)
    table = {c["world"]: c["route"] for c in cands}
    tf = os.path.join(S.RL, "data", "routes", "lib_v2_candidate.json")
    json.dump(table, open(tf, "w"), indent=1)
    rep["table"] = table
    if a.no_band:
        json.dump(rep, open(a.out, "w"), indent=1)
        return
    rep["band"] = {"cand": band(f"--base {base} --route-table {tf}", "routes-lib-v2", a.threads), "ref": band(f"--base {base}", "routes-ref-v635", a.threads)}
    print(f"[confirm] band: table {rep['band']['cand']['losses_below_2500']} <2500 {rep['band']['cand']['paired_vs_v631']} rate {rep['band']['cand']['rate_2500plus']}; "
          f"v63.5_rl {rep['band']['ref']['losses_below_2500']} {rep['band']['ref']['paired_vs_v631']} rate {rep['band']['ref']['rate_2500plus']}", flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
