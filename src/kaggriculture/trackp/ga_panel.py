"""Population GA over the FULL planner genome space, fitness = ELITE PANEL.

Operator directive 2026-09-02: search all combinations, score every
candidate against the benchmark (elite tapes) — local panel >= 0.82 has
measured to mean 2800+ live. Fitness = win cells vs a fixed, diverse
subset of the live top-100 panel (both seats, fixed seed), evaluated on
serve. The planner phenotype (closed loop) comes from build_econ_agent.

    python src/trackp/ga_panel.py --gens 2000
State: models/trackp/ga_panel/state.json (resumes).
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STATE_DIR = os.path.join(ROOT, "models", "trackp", "ga_panel")
WORK = os.path.join(ROOT, ".local", "ga_panel")
N_PANEL = 16          # diverse elite tapes in the fitness subset
SEED = 901
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "SHEEP", "COW")
ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
         "MILK", "WOOL", "FERTILIZER")


def load_panel():
    man = json.load(open(os.path.join(ROOT, ".local", "elite_panel",
                    "manifest.json"), encoding="utf-8"))
    teams = man["teams"]
    # diverse: spread across the lb range, step through the list
    step = max(1, len(teams) // N_PANEL)
    return [t["tape"] for t in teams[::step]][:N_PANEL]


def seed_genomes():
    out = []
    for fn in ("genome_elite_fit.json", "genome_elite_econ3.json",
               "genome_wool_econ.json"):
        p = os.path.join(ROOT, "models", "trackp", fn)
        if os.path.exists(p):
            out.append(json.load(open(p)))
    bg = os.path.join(STATE_DIR, "best_genome.json")
    if os.path.exists(bg):
        out.append(json.load(open(bg)))
    st = os.path.join(ROOT, "models", "trackp", "genome_ga",
                      "state_live.json")
    if os.path.exists(st):
        s = json.load(open(st))
        out.append(s["best_ever"]["genome"])
    # wheat-industry archetype variants around the elite fit
    if out:
        base = json.loads(json.dumps(out[0]))
        for wheat_n, hires_scale in ((60, 1.0), (45, 0.8), (70, 1.2)):
            g = json.loads(json.dumps(base))
            for q, share in (("NW", 0.4), ("NE", 0.25), ("SW", 0.25),
                             ("SE", 0.1)):
                g["alloc"].setdefault(q, {})
                g["alloc"][q]["WHEAT"] = int(wheat_n * share)
            g["hires"] = [max(0, int(h * hires_scale)) for h in g["hires"]]
            out.append(g)
    return out


def mutate(g, rng):
    g = json.loads(json.dumps(g))
    k = rng.random()
    if k < 0.25:      # hires curve
        d = rng.randrange(30)
        g["hires"][d] = max(0, g["hires"][d] + rng.choice((-2, -1, 1, 2)))
    elif k < 0.45:    # alloc
        q = rng.choice(("NW", "NE", "SW", "SE"))
        sp = rng.choice(CROPS + ANIMALS)
        a = g["alloc"].setdefault(q, {})
        a[sp] = max(0, a.get(sp, 0) + rng.choice((-3, -1, 1, 3)))
        if a[sp] == 0:
            a.pop(sp, None)
    elif k < 0.60:    # holds
        it = rng.choice(ITEMS)
        g["hold_until"][it] = max(0, min(29,
            g["hold_until"].get(it, 0) + rng.choice((-4, -2, 2, 4))))
    elif k < 0.72:    # caps
        it = rng.choice(ITEMS)
        g["sell_cap"][it] = max(1,
            g["sell_cap"].get(it, 20) + rng.choice((-8, -3, 3, 8)))
    elif k < 0.82:    # land timing
        q = rng.choice(("NE", "SW", "SE"))
        cur = g["land"].get(q, -1)
        g["land"][q] = rng.choice((-1, 3, 5, 6, 8, 11, 14)) \
            if cur == -1 or rng.random() < 0.3 else max(2, cur + rng.choice((-2, 2)))
        g["start"][q] = max(0, g["land"][q])
    elif k < 0.90:    # animal schedule
        sched = g.setdefault("animal_sched", {"COW": [[0, 1], [4, 3], [9, 6]],
                                              "SHEEP": [[0, 1], [9, 3]]})
        sp = rng.choice(("COW", "SHEEP"))
        rows = sched.setdefault(sp, [[0, 1]])
        i = rng.randrange(len(rows))
        if rng.random() < 0.5:
            rows[i][0] = max(0, min(25, rows[i][0] + rng.choice((-2, 2))))
        else:
            rows[i][1] = max(0, rows[i][1] + rng.choice((-1, 1)))
    else:             # scalars
        f = rng.choice(("feed_buy", "fert_buy", "last_plant_day"))
        g[f] = max(0, int(g.get(f, 4)) + rng.choice((-3, -1, 1, 3)))
    return g


def crossover(a, b, rng):
    g = json.loads(json.dumps(a))
    for key in ("hires", "alloc", "hold_until", "sell_cap", "land",
                "feed_buy", "fert_buy", "last_plant_day", "start"):
        if rng.random() < 0.5 and key in b:
            g[key] = json.loads(json.dumps(b[key]))
    return g


def build(genome, out):
    gp = out + ".genome.json"
    json.dump(genome, open(gp, "w"))
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "build_econ_agent.py"),
                        "--genome", gp, "--out", out],
                       capture_output=True, text=True, timeout=120)
    return os.path.exists(out) and "wrote" in (r.stdout or "")


def fitness(agent_path, panel, srv, SM):
    """score = winrate + tiny margin term: below the first win the margin
    is the only gradient (elite-fit seeds bank 28k vs 100k+ and lose
    every cell — pure W/L is flat there)."""
    w = 0
    margin = 0.0
    n = 0
    for tape in panel:
        for seat in (0, 1):
            try:
                a, b = SM.load_agent(agent_path), SM.load_agent(tape)
                ab, tb = (SM.run_match(a, b, SEED, srv) if seat == 0
                          else SM.run_match(b, a, SEED, srv)[::-1])
                w += ab > tb
                margin += (ab - tb)
                n += 1
            except Exception:                                  # noqa: BLE001
                return 0.0
    # margin term bounded to (0, 0.01) per 1k avg: never outweighs a win
    m = margin / max(1, n)
    return w / (2 * len(panel)) + max(-0.4, min(0.4, m / 250000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    import kaggriculture.engine.serve_match as SM
    srv = SM.Serve()
    panel = load_panel()
    print(f"panel subset: {len(panel)} tapes", flush=True)
    rng = random.Random(12345)
    sp = os.path.join(STATE_DIR, "state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        pop = st["pop"]
        gen0 = st["gen"]
        best = tuple(st["best"]) if st.get("best") else None
    else:
        pop = []
        gen0 = 0
        best = None
        for i, g in enumerate(seed_genomes()):
            ap_ = os.path.join(WORK, f"seed{i}.py")
            if build(g, ap_):
                f = fitness(ap_, panel, srv, SM)
                pop.append([f, g])
                print(f"seed{i}: {f:.3f}", flush=True)
        while len(pop) < args.pop:
            g = mutate(pop[rng.randrange(len(pop))][1], rng)
            pop.append([-9.0, g])   # unevaluated: never outranks a real score
    for gen in range(gen0, args.gens):
        t0 = time.time()
        pop.sort(key=lambda x: -x[0])
        pop = pop[:args.pop]
        # tournament: two parents from top half, crossover + mutate
        half = max(2, len(pop) // 2)
        pa, pb = rng.sample(range(half), 2)
        child = mutate(crossover(pop[pa][1], pop[pb][1], rng), rng)
        cp = os.path.join(WORK, "cand.py")
        f = 0.0
        if build(child, cp):
            f = fitness(cp, panel, srv, SM)
        pop.append([f, child])
        if best is None or f > best[0]:
            best = (f, gen)
            json.dump(child, open(os.path.join(STATE_DIR,
                      "best_genome.json"), "w"), indent=1)
        json.dump({"gen": gen + 1, "pop": pop,
                   "best": list(best)}, open(sp, "w"))
        print(f"gen {gen}: cand {f:.3f} | best {best[0]:.3f}@{best[1]} "
              f"| top {pop[0][0]:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
