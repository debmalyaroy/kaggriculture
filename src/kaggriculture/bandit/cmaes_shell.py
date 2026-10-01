"""CMA-ES tuning of the bandit shell's knobs with CLOSED-LOOP copy-race fitness (Rust self-play).

Open-loop benchmarks saturate (v63 ~0.98 against any recorded opponent); the tests that separate candidates are
closed-loop races against reacting copies of our lineage, which is also where v63 loses on the ladder. Fitness of a
knob vector = mean over games vs a copy population (v63, v62.1, v62, v61.1, the base profile itself; both seats) and
random-knob lineage clones (seat A only) of  win + 0.25 * tanh(margin / 3000). Every candidate of a generation plays
the same seeds; the seed block rotates each generation. Every --check generations the incumbent best plays a fixed
HELD-OUT seed set against the base profile (paired), which is the only number that decides anything.

    python -m kaggriculture.bandit.cmaes_shell --name cma1 [--gens 50] [--lam 12] [--seeds 6] [--workers 16] [--base 100]
    python -m kaggriculture.bandit.cmaes_shell --name cma2 --gens 30 --bank .local/w64/bank.json   # all 64 worlds

--bank: {"train": {world: [seeds]}, "heldout": {world: [seeds]}} (world = the REALIZED first two shops; inside our
lineage the realized world is a function of the seed alone, verified 600/600 across four knob pairings). Each
generation plays one training seed per world (rotating); the held-out check plays every held-out seed and reports
better/worse per world.

--standins FILE: extra OPPONENT profiles (Rust stand-ins of public agents of our family, see bandit/port_public.py and
configs/bandit/profiles/standins_*.json) appended to the table and played from both seats, in the fitness AND the
held-out check. --worlds-per-gen N (with --bank): each generation plays N worlds (alternating slices of the sorted world
list, so ceil(64/N) generations cover all 64); the held-out check always plays every world.

Writes .local/cmaes/<name>/{log.jsonl, best.json, profiles.json}; the best vector is appended to
configs/bandit/profiles/v4.json as "cma_<name>" only at the end (then it still needs the 64-world tournament).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kaggriculture.paths import ROOT

BIN = os.path.join(ROOT, "rustengine", "v62", "target-x", "release", "selfplay.exe")
BASE_DIR = os.path.join(ROOT, "configs", "bandit", "bases", "v61.1")
V4 = os.path.join(ROOT, "configs", "bandit", "profiles", "v4.json")
OPPS = [71, 19, 13, 0]  # plus the base profile itself (mirror) and random-knob clones

# (knob, lo, hi, integer)
SPACE = [
    ("v92_h", 12, 96, True), ("v92_k", 2, 6, True), ("v92_top", 1, 5, True), ("v92_ext_window", 0, 12, True),
    ("rsa_look", 2, 24, True), ("rsa_min_frac", 0.2, 0.9, False),
    ("ev_h", 4, 48, True), ("dp_h", 4, 48, True), ("mp_h", 4, 48, True),
    ("v9_race_default", 20, 96, True), ("v9_race_max", 24, 120, True), ("v9_race_margin", 0, 30, True),
    ("racepx_margin", -20, 40, True), ("ca_margin", -40.0, 0.0, False), ("or2_slot_margin", 0.0, 30.0, False),
    ("adv_look", 1, 6, True), ("tsell_window", 8, 48, True), ("tsell_min", 0.0, 150.0, False),
]
DEFAULTS = {"rsa_min_frac": 0.5, "v9_race_margin": 12, "racepx_margin": 0, "ca_margin": -20.0, "or2_slot_margin": 12.0,
            "adv_look": 3, "tsell_window": 24, "v92_h": 48}


def decode(x):
    out = {}
    for (k, lo, hi, integer), u in zip(SPACE, np.clip(x, 0, 1)):
        v = lo + u * (hi - lo)
        out[k] = int(round(v)) if integer else round(float(v), 3)
    return out


def encode(knobs):
    return np.array([(float(knobs.get(k, DEFAULTS.get(k, lo))) - lo) / (hi - lo) for k, lo, hi, _ in SPACE])


class CMA:
    """Minimal (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial parameters)."""

    def __init__(self, x0, sigma, lam):
        n = len(x0)
        self.n, self.lam, self.mean, self.sigma = n, lam, np.array(x0, float), sigma
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self, rng):
        vals, vecs = np.linalg.eigh(self.C)
        self.B, self.D = vecs, np.sqrt(np.maximum(vals, 1e-20))
        return [self.mean + self.sigma * self.B @ (self.D * rng.standard_normal(self.n)) for _ in range(self.lam)]

    def tell(self, xs, fits):
        # maximise fitness
        order = np.argsort(fits)[::-1][: self.mu]
        old = self.mean.copy()
        sel = np.array([xs[i] for i in order])
        self.mean = self.w @ sel
        y = (self.mean - old) / self.sigma
        inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ y
        self.gen += 1
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        artmp = (sel - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * artmp.T @ np.diag(self.w) @ artmp)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))


def play(args):
    """One selfplay run: (cand_id, opp, seat, seeds, profiles_path) -> list of (win, margin) from the candidate's side."""
    cand, opp, seat, seeds, prof = args
    cmd = [BIN, "--a", BASE_DIR, "--b", BASE_DIR, "--profiles", prof, "--seeds", ",".join(map(str, seeds)), "--threads", "1"]
    if opp == "rand":
        cmd += ["--pa", str(cand), "--pb", "0", "--rand-b", "77"]
    elif seat == 0:
        cmd += ["--pa", str(cand), "--pb", str(opp)]
    else:
        cmd += ["--pa", str(opp), "--pb", str(cand)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res = []
    for ln in out.splitlines():
        p = ln.split("\t")
        if len(p) >= 3 and p[0].lstrip("-").isdigit():
            us, them = (float(p[1]), float(p[2])) if (opp == "rand" or seat == 0) else (float(p[2]), float(p[1]))
            res.append(((1.0 if us > them else 0.0 if us < them else 0.5), us - them, int(p[0])))
    return args, res


def evaluate(cands, seeds, prof, workers, base_id, opps=None):
    """cands: list of profile ids -> fitness list (and raw per-game results keyed by (cand, opp, seat, seed))."""
    opps = OPPS + [base_id] + list(opps or [])
    jobs = [(c, o, s, seeds, prof) for c in cands for o in opps for s in (0, 1)] + [(c, "rand", 0, seeds, prof) for c in cands]
    raw = {}
    with ThreadPoolExecutor(workers) as ex:
        for (c, o, s, _, _), res in ex.map(play, jobs):
            for win, margin, seed in res:
                raw[(c, str(o), s, seed)] = (win, margin)
    fits = []
    for c in cands:
        v = [w + 0.25 * math.tanh(m / 3000) for (cc, _, _, _), (w, m) in raw.items() if cc == c]
        fits.append(sum(v) / max(1, len(v)))
    return fits, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cma1")
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--lam", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6, help="seeds per (opponent, seat) per generation")
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base", type=int, default=100, help="profile id in v4.json to start from and compare against")
    ap.add_argument("--check", type=int, default=10)
    ap.add_argument("--heldout", type=int, default=30, help="held-out seeds per (opponent, seat) for the check")
    ap.add_argument("--bank", default=None, help="world-stratified seed bank (see module doc)")
    ap.add_argument("--standins", default=None, help="JSON {profiles: [...]} of extra opponent profiles")
    ap.add_argument("--worlds-per-gen", type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(ROOT, ".local", "cmaes", a.name)
    os.makedirs(out, exist_ok=True)
    v4 = json.load(open(V4, encoding="utf-8"))
    base = v4["profiles"][a.base]
    extra_opps = []
    if a.standins:
        sp = json.load(open(a.standins, encoding="utf-8"))["profiles"]
        extra_opps = list(range(len(v4["profiles"]), len(v4["profiles"]) + len(sp)))
        v4["profiles"] += sp
        print(f"[cma] stand-in opponents: {[(i, p['name']) for i, p in zip(extra_opps, sp)]}", flush=True)
    rng = np.random.default_rng(20260927)
    es = CMA(encode(base), a.sigma, a.lam)
    best = (-1e9, None, None)
    prof_path = os.path.join(out, "profiles.json")
    heldout = list(range(990000, 990000 + a.heldout))
    bank, world_of = None, {}
    if a.bank:
        bank = json.load(open(a.bank, encoding="utf-8"))
        heldout = sorted(x for v in bank["heldout"].values() for x in v)
        world_of = {x: w for part in ("train", "heldout") for w, v in bank[part].items() for x in v}
    log = open(os.path.join(out, "log.jsonl"), "a", encoding="utf-8")
    for g in range(a.gens):
        xs = es.ask(rng)
        table = copy.deepcopy(v4)
        n0 = len(table["profiles"])
        ids = []
        for i, x in enumerate(xs + [es.mean]):
            p = {**{k: v for k, v in base.items() if k != "name"}, **decode(x), "name": f"{a.name}_g{g}_{i}"}
            table["profiles"].append(p)
            ids.append(n0 + i)
        json.dump(table, open(prof_path, "w", encoding="utf-8"))
        if bank:
            ws = sorted(bank["train"].items())
            if a.worlds_per_gen and a.worlds_per_gen < len(ws):
                k = -(-len(ws) // a.worlds_per_gen)
                ws = ws[g % k::k]
            seeds = sorted(v[g % len(v)] for _, v in ws)
        else:
            seeds = list(range(900000 + g * 1000, 900000 + g * 1000 + a.seeds))
        fits, _ = evaluate(ids, seeds, prof_path, a.workers, a.base, extra_opps)
        es.tell(xs, fits[:-1])
        mean_fit = fits[-1]
        if mean_fit > best[0]:
            best = (mean_fit, decode(es.mean), g)
        rec = {"gen": g, "sigma": es.sigma, "fit_max": max(fits[:-1]), "fit_mean_cands": float(np.mean(fits[:-1])), "fit_of_mean": mean_fit,
               "mean_knobs": decode(es.mean)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[cma] gen {g}: sigma {es.sigma:.3f} fit max {rec['fit_max']:.3f} cands {rec['fit_mean_cands']:.3f} mean-vector {mean_fit:.3f}", flush=True)
        if (g + 1) % a.check == 0 or g == a.gens - 1:
            # held-out paired check: the current mean vector vs the base profile, same opponents / seats / seeds
            table = copy.deepcopy(v4)
            table["profiles"].append({**{k: v for k, v in base.items() if k != "name"}, **decode(es.mean), "name": f"{a.name}_check_g{g}"})
            json.dump(table, open(prof_path, "w", encoding="utf-8"))
            cid = len(table["profiles"]) - 1
            _, raw = evaluate([cid, a.base], heldout, prof_path, a.workers, a.base, extra_opps)
            c = {k[1:]: v[0] for k, v in raw.items() if k[0] == cid}
            b = {k[1:]: v[0] for k, v in raw.items() if k[0] == a.base}
            ks = set(c) & set(b)
            bw, ww = sum(c[k] > b[k] for k in ks), sum(c[k] < b[k] for k in ks)
            n = bw + ww
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(bw, ww) + 1)) / 2 ** n) if n else 1.0
            chk = {"gen": g, "check": True, "n": len(ks), "score": sum(c[k] for k in ks) / max(1, len(ks)),
                   "base_score": sum(b[k] for k in ks) / max(1, len(ks)), "better": bw, "worse": ww, "p": p, "knobs": decode(es.mean)}
            if world_of:
                pw = {}
                for k in ks:
                    w = world_of.get(k[2], "?")
                    x = pw.setdefault(w, [0, 0])
                    x[0] += c[k] > b[k]
                    x[1] += c[k] < b[k]
                chk["per_world"] = pw
                worse_w = sorted(((x[1] - x[0], w) for w, x in pw.items() if x[1] > x[0]), reverse=True)
                print(f"[cma] worlds net-worse: {len(worse_w)}/{len(pw)} {worse_w[:6]}", flush=True)
            log.write(json.dumps(chk) + "\n")
            log.flush()
            print(f"[cma] HELD-OUT gen {g}: {chk['score']:.3f} vs base {chk['base_score']:.3f}  +{bw}/-{ww} p {p:.3g} (n {len(ks)})", flush=True)
            json.dump(chk, open(os.path.join(out, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
