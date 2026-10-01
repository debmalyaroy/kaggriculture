"""Live-fitness genome GA (Track-P, T1 of the 2026-08-31 plan).

The tape-fitness GA optimized RELATIVE strength vs mutually-degraded
open-loop opponents; its champion, perfectly executed by the live
planner, still banks 35-75k while the real field banks 90-150k (0/36,
margins -70k to -101k). This GA optimizes the honest objective:

    fitness = OWN BANK, live, vs a strong reactive opponent,
    with the LIVE PLANNER as the phenotype (what ships is what's scored).

Protocol carries every burned lesson: rotating seeds, paired per-cell
own-bank deltas vs the incumbent on identical (seed, seat) cells, and a
two-stage fresh-seed confirm before any accept.

    python src/trackp/genome_ga_live.py --gens 2000 --pop 8 --workers 8
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

STATE_DIR = os.path.join(ROOT, "models", "trackp", "genome_ga")
WORK = os.path.join(ROOT, ".local", "trackp", "ga_live")
OPPONENT = os.path.join(ROOT, "data", "gauntlet", "kaito_v48.py")


_SRV = None


def _play(job):
    """One live game on the RUST SERVE substrate (2026-09-01: fresh
    equivalence spot-check exact-bank MATCH; ~4-5x per game, persistent
    server per worker). Falls back to the python engine on any serve
    error so a substrate hiccup can never poison a generation."""
    agent_path, opp, seed, seat = job
    import sys as _s
    global _SRV
    try:
        import kaggriculture.engine.serve_match as SM
        if _SRV is None:
            _SRV = SM.Serve()
        a = SM.load_agent(agent_path if seat == 0 else opp)
        b = SM.load_agent(opp if seat == 0 else agent_path)
        banks = SM.run_match(a, b, seed, srv=_SRV)
        return (banks[seat], banks[1 - seat])
    except Exception:
        try:
            if _SRV is not None:
                _SRV.close()
        except Exception:
            pass
        _SRV = None
        import kaggriculture.engine._vendor as _vendor  # noqa: F401
        from kaggle_environments import make
        env = make("kaggriculture", configuration={
            "episodeSteps": 720, "seed": seed, "actTimeout": 60,
            "runTimeout": 100000})
        pair = [agent_path, opp] if seat == 0 else [opp, agent_path]
        env.run(pair)
        last = env.steps[-1]
        bk = [float(last[i]["reward"] or 0) for i in range(2)]
        return (bk[seat], bk[1 - seat])


def _render(genome, path):
    from kaggriculture.trackp import build_econ_agent as B
    import datetime as dt
    src = B.TEMPLATE.format(label=os.path.basename(path),
                            built=dt.date.today().isoformat(), genome=genome)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    return path


def _eval(genomes, seeds, workers):
    """Returns per-genome dict of cells {(seed, seat): (own, opp)}."""
    paths = []
    for i, g in enumerate(genomes):
        paths.append(_render(g, os.path.join(WORK, f"cand_{i}.py")))
    jobs, meta = [], []
    for i, p in enumerate(paths):
        for s in seeds:
            for seat in (0, 1):
                jobs.append((p, OPPONENT, s, seat))
                meta.append((i, s, seat))
    out = [dict() for _ in genomes]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (i, s, seat), (own, opp) in zip(meta, ex.map(_play, jobs,
                                                         chunksize=1)):
            out[i][(s, seat)] = (own, opp)
    return out


def _paired(cells_c, cells_i):
    ks = [k for k in cells_c if k in cells_i]
    if not ks:
        return 0.0, 0.0, 0.0, 0.0
    db = sum(cells_c[k][0] - cells_i[k][0] for k in ks) / len(ks)
    pos = sum(1 for k in ks if cells_c[k][0] > cells_i[k][0]) / len(ks)

    def sc(c):
        return 1.0 if c[0] > c[1] else (0.5 if c[0] == c[1] else 0.0)
    ds = sum(sc(cells_c[k]) - sc(cells_i[k]) for k in ks) / len(ks)
    own = sum(cells_c[k][0] for k in ks) / len(ks)
    return db, pos, ds, own


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    from kaggriculture.trackp import economy as E

    os.makedirs(WORK, exist_ok=True)
    state_path = os.path.join(STATE_DIR, "state_live.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
        center, gen0, best_ever = st["center"], st["gen"], st.get("best_ever")
        print(f"resuming at gen {gen0}", flush=True)
    else:
        adv = json.load(open(os.path.join(STATE_DIR, "state_adv.json"),
                             encoding="utf-8"))
        center, gen0, best_ever = adv["center"], 0, None
        print("seeded from tape-GA center", flush=True)

    rng = random.Random(90210)
    for gen in range(gen0, args.gens):
        t0 = time.time()
        genomes = [center]
        for _ in range(args.pop):
            child = E.mutate_genome(center, rng)
            if rng.random() < 0.35:
                child = E.mutate_genome(child, rng)
            genomes.append(child)
        seeds = [95000 + gen * 17 + k for k in range(args.seeds)]
        res = _eval(genomes, seeds, args.workers)
        stats = [_paired(res[i], res[0]) for i in range(len(genomes))]
        best_i = max(range(1, len(genomes)),
                     key=lambda i: (stats[i][2], stats[i][0]))
        db, pos, ds, own = stats[best_i]
        inc_own = stats[0][3]
        moved = ""
        if ds > 0 or (db > 1000 and pos >= 0.6):
            cseeds = [97000 + gen * 19 + k for k in range(args.seeds)]
            conf = _eval([genomes[0], genomes[best_i]], cseeds, args.workers)
            cdb, cpos, cds, cown = _paired(conf[1], conf[0])
            if cds > 0 or (cdb > 1000 and cpos >= 0.6):
                center = genomes[best_i]
                moved = f" * (conf {cds:+.3f}/{cdb:+,.0f}/p{cpos:.2f})"
            else:
                moved = f" x (conf {cds:+.3f}/{cdb:+,.0f}/p{cpos:.2f})"
        row = {"gen": gen, "own_bank": own, "genome": copy.deepcopy(
            genomes[best_i])}
        if best_ever is None or row["own_bank"] > best_ever["own_bank"]:
            best_ever = row
        json.dump({"center": center, "gen": gen + 1, "best_ever": best_ever},
                  open(state_path, "w", encoding="utf-8"))
        print(f"gen {gen:>4}: inc-bank {inc_own:>8,.0f}  "
              f"best-child {own:>8,.0f} d{db:+,.0f}/p{pos:.2f}/s{ds:+.3f}"
              f"{moved}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
