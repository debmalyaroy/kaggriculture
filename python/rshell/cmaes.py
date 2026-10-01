"""CMA-ES over reactive shell v2's tunables and the chain's timing constants (operator order 2026-09-27).

    python python/rshell/cmaes.py --shell weights/rshell/NAME/rshell.json [--gens 40] [--lam 16] [--threads 24]
                                  [--chain-off-file data/rshell/screen/chain_off.txt] [--out data/rshell/cma]

Every generation plays all 64 worlds, one seed per world rotating through the bank's train split (generation g
uses rotation g), against the 7 lineage opponents (panel.OPPONENTS) from both seats, plus the open-loop sets
(training ladder half, 600 training band tapes, our real losses split A). Fitness of a candidate =
paired (better - worse) vs the reference agent (i790 + big1) on the SAME games, closed loop + open loop.
The reference is replayed on each generation's seeds. Starts from the trained config (x0 = the file's values,
defaults elsewhere); an independent restart landing on similar values is the evidence that a vector is real.
Every generation's best is kept; at the end the best mean-of-generations candidate is written to
OUT/best_rshell.json + OUT/best_knobs.json and checked by python/rshell/test.py (held-out, per world).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402

# (name, lo, hi, kind)  kind: f = float, i = int, b = bool (> 0.5), m = mode 0/1/2; dotted = array element
SPACE = [
    ("tau", 0.3, 0.99, "f"), ("stay", -2.0, 6.0, "f"), ("model_w", 0.0, 2.0, "f"), ("beta_price", 0.0, 20.0, "f"),
    *[(f"bias.{i}", -3.0, 3.0, "f") for i in range(5)],
    *[(f"urge_w.{i}", -3.0, 3.0, "f") for i in range(15)],
    *[(f"item_urge.{i}", -3.0, 3.0, "f") for i in range(9)],
    *[(f"group_tau.{i}", -0.3, 0.3, "f") for i in range(3)],
    *[(f"group_mode.{i}", 0.0, 2.0, "m") for i in range(3)],
    *[(f"group_on.{i}", 0.0, 1.0, "b") for i in range(3)],
    ("from", 25, 400, "i"), ("to", 500, 717, "i"), ("glut_floor", 0.0, 1.5, "f"), ("margin_gate", -1.0, 2.0, "f"),
    ("reorder", 0.0, 1.0, "b"), ("prio_urge", -2.0, 2.0, "f"), ("front", 0.0, 10.0, "f"), ("horizon", 8, 96, "i"),
    *[(f"fc_w.{i}", 0.0, 2.0, "f") for i in range(4)], ("lin_min", 0.6, 1.0, "f"), ("fert_reserve", 0, 10, "i"),
    ("item_on.8", 0.0, 1.0, "b"),
    # chain constants no PPO profile sets (knob-over on every profile)
    ("K.r36_from", 144, 288, "i"), ("K.r37_base", 1, 4, "i"), ("K.r37_streak", 2, 6, "i"), ("K.r37_late", 2, 8, "i"),
    ("K.r37_late_from", 192, 400, "i"), ("K.r37_sim", 0.8, 0.98, "f"), ("K.race_from", 144, 300, "i"),
    ("K.lead_frac", 0.4, 1.0, "f"), ("K.lead_from", 48, 200, "i"),
]
DEFAULTS = {"tau": 0.6, "stay": 2.0, "model_w": 1.0, "beta_price": 0.0, "from": 25, "to": 717, "glut_floor": 0.0, "margin_gate": -1.0,
            "reorder": 0.0, "prio_urge": 0.0, "front": 10.0, "horizon": 48, "lin_min": 0.9, "fert_reserve": 0,
            "K.r36_from": 192, "K.r37_base": 2, "K.r37_streak": 3, "K.r37_late": 4, "K.r37_late_from": 288, "K.r37_sim": 0.9,
            "K.race_from": 216, "K.lead_frac": 0.75, "K.lead_from": 96}
ARR_DEFAULT = {"bias": [0.0] * 5, "urge_w": [0.0] * 15, "item_urge": [0.0] * 9, "group_tau": [0.0] * 3, "group_mode": [0] * 3, "group_on": [1] * 3,
               "fc_w": [0.0, 1.0, 1.0, 0.0], "item_on": [1] * 9}


def start_values(cfg):
    x = []
    for name, lo, hi, kind in SPACE:
        if "." in name and not name.startswith("K."):
            arr, i = name.split(".")
            src = list(cfg.get(arr, ARR_DEFAULT[arr]))
            v = src[int(i)] if int(i) < len(src) else ARR_DEFAULT[arr][int(i)]
        else:
            v = cfg.get(name, DEFAULTS.get(name, (lo + hi) / 2))
        v = float(v) if not isinstance(v, bool) else float(v)
        x.append((min(max(v, lo), hi) - lo) / (hi - lo))
    return np.array(x)


def decode(z, base_cfg):
    """z in [0,1]^n (clipped) -> (shell config, chain knob overrides)."""
    cfg = json.loads(json.dumps(base_cfg))
    knobs = {}
    for (name, lo, hi, kind), u in zip(SPACE, np.clip(z, 0, 1)):
        v = lo + u * (hi - lo)
        v = int(round(v)) if kind in ("i", "m") else bool(v > 0.5 * (lo + hi)) if kind == "b" else round(float(v), 4)
        if name.startswith("K."):
            knobs[name[2:]] = v
        elif "." in name:
            arr, i = name.split(".")
            cur = list(cfg.get(arr, ARR_DEFAULT[arr]))
            cur += ARR_DEFAULT[arr][len(cur):]
            cur[int(i)] = (1 if v else 0) if kind == "b" else v
            cfg[arr] = cur
        else:
            cfg[name] = v
    if cfg.get("front", 10) >= 9.5:
        cfg["front"] = 99.0
    return cfg, knobs


class CMA:
    """Minimal (mu/mu_w, lambda)-CMA-ES (Hansen 2016 tutorial defaults), minimising."""

    def __init__(self, x0, sigma, lam):
        n = len(x0)
        self.n, self.lam, self.mu = n, lam, lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m, self.sigma = np.array(x0, float), sigma
        self.C, self.pc, self.ps = np.eye(n), np.zeros(n), np.zeros(n)
        self.g = 0

    def ask(self, rng):
        self.D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(self.D2, 1e-20))
        self.z = rng.standard_normal((self.lam, self.n))
        self.y = self.z * self.D @ self.B.T
        return self.m + self.sigma * self.y

    def tell(self, f):
        idx = np.argsort(f)[: self.mu]
        yw = (self.w[:, None] * self.y[idx]).sum(0)
        self.m = self.m + self.sigma * yw
        cinv = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * cinv @ yw
        self.g += 1
        hs = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank = sum(wi * np.outer(self.y[i], self.y[i]) for wi, i in zip(self.w, idx))
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", required=True, help="trained rshell.json (its folder holds net.json + lineage.json)")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--lam", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--v1-shell", default=P.BIG1, help="keep the v1 sales shell under shell v2 ('' = none); diagnostic 2026-09-27: big1 is worth +253 net closed loop")
    ap.add_argument("--it", type=int, default=790, help="PPO snapshot the candidate plays with")
    ap.add_argument("--chain-off-file", default=None)
    ap.add_argument("--out", default=os.path.join(P.RL, "data", "rshell", "cma"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume-center", default=None, help="folder with mean_rshell.json + mean_knobs.json of an interrupted run: start from that center")
    ap.add_argument("--gen0", type=int, default=0, help="first generation index (seed rotation continues from here after a resume)")
    ap.add_argument("--x0", default="tau=0.85,stay=3.0", help="start values overriding the trained config, name=value,... "
                    "(a cautious start: the shell overrides the chain only when confident; CMA-ES opens it up if that wins)")
    ap.add_argument("--ref-extra", default=None, help="reference = PPO --ref-it + big1 + these args (e.g. v63.7_rl's --rshell/--knob-over/--endg/--chain-off) instead of i790 + big1")
    ap.add_argument("--ref-it", type=int, default=910)
    ap.add_argument("--cand-extra", default="", help="args every candidate carries besides its --rshell/--knob-over (e.g. --endg F)")
    a = ap.parse_args()
    if a.ref_extra is not None:
        P.REF = P.ppo_args(a.ref_it, extra=a.ref_extra)
        print(f"[cma] reference: {P.REF}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    shell_dir = os.path.dirname(os.path.abspath(a.shell))
    base_cfg = json.load(open(a.shell))
    chain_off = open(a.chain_off_file).read().strip() if a.chain_off_file and os.path.exists(a.chain_off_file) else ""
    sets = P.tape_sets("A")
    start = dict(base_cfg)
    for kv in filter(None, (a.x0 or "").split(",")):
        k, v = kv.split("=")
        start[k] = float(v)
    if a.resume_center:
        start.update({k: v for k, v in json.load(open(os.path.join(a.resume_center, "mean_rshell.json"))).items() if k not in ("net_file", "lineage_file")})
        start.update({"K." + k: v for k, v in json.load(open(os.path.join(a.resume_center, "mean_knobs.json"))).items()})
        print(f"[cma] resumed from the center in {a.resume_center}", flush=True)
    es = CMA(start_values(start), a.sigma, a.lam)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    hist = []
    print(f"[cma] {len(SPACE)} dims, lambda {a.lam}, {a.gens} generations; chain-off: {chain_off or '-'}", flush=True)
    for g in range(a.gen0, a.gen0 + a.gens):
        t0 = time.time()
        seeds = P.bank_seeds("train", 1, g)
        ref_cl, _ = P.closed_loop(P.REF, seeds, a.threads, opponents=P.FIT)
        ref_ol = P.open_loop(P.REF, sets, a.threads)
        X = es.ask(rng)
        fit, rows = [], []
        for k, x in enumerate(list(X) + [es.m]):
            cfg, knobs = decode(x, base_cfg)
            cf = os.path.join(shell_dir, f"cma_g{g}_{k}.json")
            kf = os.path.join(a.out, f"knobs_g{g}_{k}.json")
            json.dump(cfg, open(cf, "w"))
            json.dump(knobs, open(kf, "w"))
            extra = f"--rshell {cf} --knob-over {kf}" + (f" --chain-off {chain_off}" if chain_off else "") + (f" {a.cand_extra}" if a.cand_extra else "")
            agent = P.ppo_args(a.it, shell=a.v1_shell or None, extra=extra)
            cl, worlds = P.closed_loop(agent, seeds, a.threads, opponents=P.FIT)
            ol = P.open_loop(agent, sets, a.threads)
            dc, pc = P.score_delta(cl, ref_cl)
            do = {n: P.score_delta(ol[n], ref_ol[n]) for n in sets}
            f = dc + sum(v[0] for v in do.values())
            mism = sum(str(w).startswith("MISMATCH") for w in worlds.values())
            if k < len(X):
                fit.append(-f)
            rows.append({"g": g, "k": k, "fitness": f, "closed": pc, "open": {n: v[1] for n, v in do.items()}, "world_mismatch": mism, "cfg": cf, "knobs": knobs})
        es.tell(np.array(fit))
        for r in rows:
            if os.path.exists(r["cfg"]):
                os.remove(r["cfg"])
        mean_row = rows[-1]
        rows = rows[:-1] + [dict(mean_row, k="mean")]
        best = max(rows[:-1], key=lambda r: r["fitness"])
        for r in rows:
            log.write(json.dumps({kk: vv for kk, vv in r.items() if kk != "cfg"}) + "\n")
        log.flush()
        mcfg, mknobs = decode(es.m, base_cfg)
        hist.append({"g": g, "best": best["fitness"], "mean_fit": float(-np.mean(fit)), "center": mean_row["fitness"], "center_closed": mean_row["closed"], "sigma": es.sigma})
        json.dump(mcfg, open(os.path.join(a.out, "mean_rshell.json"), "w"), indent=1)
        json.dump(mknobs, open(os.path.join(a.out, "mean_knobs.json"), "w"), indent=1)
        print(f"[cma] gen {g + 1}/{a.gens}: center {mean_row['fitness']:+d} (closed +{mean_row['closed']['better']}/-{mean_row['closed']['worse']}), "
              f"best {best['fitness']:+d} (closed {best['closed']['better']}/{best['closed']['worse']}), "
              f"mean {-np.mean(fit):+.1f}, sigma {es.sigma:.3f}, {time.time() - t0:.0f}s", flush=True)
    # the final mean of the distribution is the candidate (a single generation's best is noisy)
    cfg, knobs = decode(es.m, base_cfg)
    cfg["cma"] = {"gens": a.gens, "lam": a.lam, "hist": hist}
    for k in ("net_file", "lineage_file"):
        if k in base_cfg:
            cfg[k] = os.path.join(shell_dir, base_cfg[k])
    json.dump(cfg, open(os.path.join(a.out, "best_rshell.json"), "w"), indent=1)
    json.dump(knobs, open(os.path.join(a.out, "best_knobs.json"), "w"), indent=1)
    open(os.path.join(a.out, "chain_off.txt"), "w").write(chain_off)
    print(f"[cma] -> {a.out}/best_rshell.json + best_knobs.json", flush=True)


if __name__ == "__main__":
    main()
