"""Profile tournament on the all-Rust runner: every profile k vs profile 0 (v61.1 mirror = the
clone war), both seats, same seeds. Reports per profile: wins/losses/draws from k's side, the
paired sign test (McNemar on per-seed-seat outcomes vs the 0-vs-0 baseline), and mean margin.

    python python/profile_tournament.py --profiles configs/profiles/v1.json --n 100 --threads 6
Output: data/tournaments/profiles_<version>_<RUNID>.json (+ a printed table).
"""
import argparse, json, math, os, subprocess, time

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(RL, "target-dev", "release", "selfplay.exe")
BASE = os.path.join(RL, "configs", "bases", "v61.1")


def run(profiles, pa, pb, seeds, threads):
    p = subprocess.run([EXE, "--a", BASE, "--profiles", profiles, "--pa", str(pa), "--pb", str(pb),
                        "--seeds", ",".join(map(str, seeds)), "--threads", str(threads)],
                       capture_output=True, text=True, cwd=RL)
    out = {}
    for line in p.stdout.splitlines():
        seed, b0, b1, *_ = line.split("\t")
        out[int(seed)] = (float(b0), float(b1))
    return out


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=os.path.join(RL, "configs", "profiles", "v1.json"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    table = json.load(open(a.profiles))
    names = [p["name"] for p in table["profiles"]]
    seeds = list(range(a.seed0, a.seed0 + a.n))
    t0 = time.time()
    base = run(a.profiles, 0, 0, seeds, a.threads)            # 0 vs 0 reference
    rows = []
    for k in range(1, len(names)):
        as0 = run(a.profiles, k, 0, seeds, a.threads)          # k in seat 0
        as1 = run(a.profiles, 0, k, seeds, a.threads)          # k in seat 1
        w = l = d = 0
        margins = []
        better = worse = 0                                     # paired vs the 0-vs-0 outcome
        for s in seeds:
            for seat, res in ((0, as0), (1, as1)):
                mine, theirs = res[s][seat], res[s][1 - seat]
                ref_mine, ref_theirs = base[s][seat], base[s][1 - seat]
                outcome = (mine > theirs) - (mine < theirs)
                ref = (ref_mine > ref_theirs) - (ref_mine < ref_theirs)
                w += outcome > 0
                l += outcome < 0
                d += outcome == 0
                margins.append(mine - theirs)
                better += outcome > ref
                worse += outcome < ref
        p = binom_two_sided(better, better + worse)
        rows.append(dict(profile=k, name=names[k], wins=w, losses=l, draws=d, better=better, worse=worse, p=p,
                         mean_margin=sum(margins) / len(margins)))
        print(f"{k:2d} {names[k]:16s} W {w:4d} L {l:4d} D {d:4d} | vs ref better {better:3d} worse {worse:3d} p={p:.4f} | margin {rows[-1]['mean_margin']:+.0f}", flush=True)
    runid = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    os.makedirs(os.path.join(RL, "data", "tournaments"), exist_ok=True)
    path = os.path.join(RL, "data", "tournaments", f"profiles_{table.get('version', 'x')}_{runid}.json")
    ref_w = sum((b[0] > b[1]) + (b[1] > b[0]) for b in base.values())
    json.dump(dict(runid=runid, profiles=a.profiles, seeds=[seeds[0], seeds[-1]], ref_decisive=ref_w, rows=rows,
                   secs=time.time() - t0), open(path, "w"), indent=1)
    print(f"TOURNAMENT done in {time.time() - t0:.0f}s -> {path}; 0-vs-0 decisive games {ref_w}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()
