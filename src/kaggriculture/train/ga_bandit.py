"""Chassis-constant search with ELITE PANEL fitness (the 0.80 hunt).

Operator order 2026-09-02: find a bandit candidate that measures 2800+
(calibration: elite panel >= ~0.80). The chassis's adaptive constants
have never been searched against elites. Candidates are the BUILT bandit
file with constant values substituted (no rebuild): mutate -> substitute
-> paired cells vs a diverse elite-tape subset on serve.

    python src/ga_bandit.py --base agents/v42.0_bandit.py --gens 2000
State: models/bandit_ga/state.json (resumes).
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STATE_DIR = os.path.join(ROOT, "models", "bandit_ga")
WORK = os.path.join(ROOT, ".local", "bandit_ga")
N_PANEL = 16
SEED = 901

# constant -> (min, max, is_float)
SPACE = {
    "_OBF_MAX_DELAY": (0, 6, False),
    "_RELAY_STREAK": (6, 24, False),
    "_RELAY_START": (96, 400, False),
    "_RELAY_MAX": (15, 80, False),
    "_RELAY_COOLDOWN": (1, 6, False),
    "_RELAY_TOL": (2, 10, False),
    "_RELAY_LEAD": (1, 6, False),
    "_RELAY_MIN_QTY": (4, 16, False),
    "_RELAY_MIN_QUOTE": (1, 8, False),
    "_RELAY_HORIZON": (4, 20, False),
    "_RELAY_COLLIDE": (3, 12, False),
    "_RELAY_OBS_MIN": (1, 5, False),
    "_RELAY_LEAD_MAX": (4, 16, False),
    "_PULL_RATIO": (1.05, 2.0, True),
    "_PULL_RATIO_CONTESTED": (1.0, 1.6, True),
    "_PULL_WINDOW": (12, 96, False),
    "_PULL_MAX_PER_TURN": (1, 4, False),
}


def substitute(src, consts):
    for k, v in consts.items():
        vv = f"{v:.3f}" if isinstance(v, float) else str(int(v))
        src, n = re.subn(rf"^({k} = )[0-9.]+", rf"\g<1>{vv}", src,
                         count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"constant {k} not found")
    return src


def read_defaults(src):
    out = {}
    for k, (lo, hi, is_f) in SPACE.items():
        m = re.search(rf"^{k} = ([0-9.]+)", src, flags=re.M)
        v = float(m.group(1))
        out[k] = v if is_f else int(v)
    return out


def mutate(c, rng):
    c = dict(c)
    for _ in range(rng.choice((1, 2, 3))):
        k = rng.choice(list(SPACE))
        lo, hi, is_f = SPACE[k]
        if is_f:
            c[k] = round(min(hi, max(lo, c[k] + rng.gauss(0, (hi - lo) / 4))), 3)
        elif rng.random() < 0.2:
            c[k] = rng.randint(int(lo), int(hi))   # jump: escape the plateau
        else:
            step = max(1, int((hi - lo) / 6))
            c[k] = int(min(hi, max(lo, c[k] + rng.choice((-step, step)))))
    return c


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def load_panel():
    man = json.load(open(os.path.join(ROOT, ".local", "elite_panel",
                    "manifest.json"), encoding="utf-8"))
    teams = man["teams"]
    step = max(1, len(teams) // N_PANEL)
    return [t["tape"] for t in teams[::step]][:N_PANEL]


def fitness(agent_path, panel, srv, SM):
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
                margin += ab - tb
                n += 1
            except Exception:                                  # noqa: BLE001
                return -9.0
    return w / (2 * len(panel)) + max(-0.4, min(0.4,
        (margin / max(1, n)) / 250000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "agents",
                                                   "v42.0_bandit.py"))
    ap.add_argument("--gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    import kaggriculture.engine.serve_match as SM
    srv = SM.Serve()
    panel = load_panel()
    base_src = open(args.base, encoding="utf-8").read()
    rng = random.Random(777)
    sp = os.path.join(STATE_DIR, "state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        pop, gen0, best = st["pop"], st["gen"], tuple(st["best"])
    else:
        d = read_defaults(base_src)
        cp = os.path.join(WORK, "cand.py")
        open(cp, "w", encoding="utf-8").write(substitute(base_src, d))
        f0 = fitness(cp, panel, srv, SM)
        print(f"defaults: {f0:.3f}", flush=True)
        pop = [[f0, d]]
        for _ in range(args.pop - 1):
            pop.append([-9.0, mutate(d, rng)])
        gen0, best = 0, (f0, -1)
        json.dump(d, open(os.path.join(STATE_DIR, "best_consts.json"), "w"))
    for gen in range(gen0, args.gens):
        t0 = time.time()
        pop.sort(key=lambda x: -x[0])
        pop = pop[:args.pop]
        half = max(2, len(pop) // 2)
        pa, pb = rng.sample(range(half), 2)
        child = mutate(crossover(pop[pa][1], pop[pb][1], rng), rng)
        cp = os.path.join(WORK, "cand.py")
        open(cp, "w", encoding="utf-8").write(substitute(base_src, child))
        f = fitness(cp, panel, srv, SM)
        pop.append([f, child])
        if f > best[0]:
            best = (f, gen)
            json.dump(child, open(os.path.join(STATE_DIR,
                      "best_consts.json"), "w"), indent=1)
        json.dump({"gen": gen + 1, "pop": pop, "best": list(best)},
                  open(sp, "w"))
        print(f"gen {gen}: cand {f:.3f} | best {best[0]:.3f}@{best[1]} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
