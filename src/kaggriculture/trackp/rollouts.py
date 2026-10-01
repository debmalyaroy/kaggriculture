"""Shared self-play rollout engine on `kagg serve` (Rust, bit-exact).

Persistent worker processes, each owning one ServeEnv and one compiled copy
of the planner template. A policy is (PARAMS overrides, L1 weights or None,
sample_temp); building one is an exec of the template into a fresh namespace
-- no files, no imports, ~2 ms.

For PPO, `record=True` swaps the namespace's _macro for a sampling wrapper
that draws each head from softmax(logits/temp) and records
(day, features32, bins, logprob) per decision.

Rewards are GAP-SHAPED, never bank-shaped: win + GAP_LAMBDA * clip(gap).
"""
from __future__ import annotations

import json
import math
import os
import random

try:
    from . import common, macro
    from .serve_env import ServeEnv, seat_view
except ImportError:  # script mode
    import sys
    from kaggriculture.trackp import common, macro
    from kaggriculture.trackp.serve_env import ServeEnv, seat_view

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "planner_template.py")

_W: dict = {}


def _worker_init():
    with open(TEMPLATE, encoding="utf-8") as fh:
        src = fh.read()
    _W["code"] = compile(src, "planner_template_inline", "exec")
    _W["env"] = ServeEnv()


def _build_ns(policy: dict) -> dict:
    ns: dict = {}
    exec(_W["code"], ns)  # noqa: S102 -- our own template
    if policy.get("params"):
        ns["PARAMS"].update(policy["params"])
    if policy.get("l1") is not None:
        ns["L1_WEIGHTS"] = policy["l1"]
    if policy.get("record"):
        _wrap_recording(ns, policy)
    return ns


def _softmax_sample(logits, temp, rng):
    mx = max(logits)
    exps = [math.exp((v - mx) / max(1e-6, temp)) for v in logits]
    z = sum(exps)
    r = rng.random() * z
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return i, math.log(max(1e-12, e / z))
    return len(logits) - 1, math.log(max(1e-12, exps[-1] / z))


def _wrap_recording(ns: dict, policy: dict):
    """Replace _macro with a sampling+recording version (training only)."""
    rng = random.Random(policy.get("rng_seed", 0))
    temp = policy.get("sample_temp", 1.0)
    log: list = []
    ns["_TRACKP_DECISIONS"] = log
    orig_features = ns["_l1_features"]
    weights = ns["L1_WEIGHTS"]

    def _macro_sampling(obs):
        day = obs["day"]
        shops = len(obs["town"]["unlocked_shops"])
        S = ns["S"]
        if (S.get("macro") is not None and S.get("macro_day") == day
                and S.get("macro_shops") == shops):
            return S["macro"]
        f = orig_features(obs)
        if weights is None:
            m = ns["_macro_rules"](obs, f)
            log.append({"day": day, "f": f,
                        "bins": [m[h] for h, _ in macro.HEADS],
                        "logp": 0.0})
        else:
            logits = _forward_logits(weights, f)
            m = {}
            off = 0
            lp = 0.0
            bins = []
            for name, k in macro.HEADS:
                b, l1p = _softmax_sample(logits[off:off + k], temp, rng)
                m[name] = b
                bins.append(b)
                lp += l1p
                off += k
            log.append({"day": day, "f": f, "bins": bins, "logp": lp})
        S["macro"] = m
        S["macro_day"] = day
        S["macro_shops"] = shops
        return m

    ns["_macro"] = _macro_sampling
    # _agent captured _macro at def time? No -- it resolves the global at
    # call time from ns, so the swap takes effect.


def _forward_logits(W, f):
    h = f
    for li in (1, 2):
        w, b = W["w%d" % li], W["b%d" % li]
        nh = []
        for j in range(len(b)):
            s = b[j]
            wj = w[j]
            for i in range(len(h)):
                s += wj[i] * h[i]
            nh.append(math.tanh(s))
        h = nh
    wo, bo = W["wo"], W["bo"]
    out = []
    for j in range(len(bo)):
        s = bo[j]
        wj = wo[j]
        for i in range(len(h)):
            s += wj[i] * h[i]
        out.append(s)
    return out


def gap_reward(bank_me: float, bank_opp: float) -> float:
    gap = bank_me - bank_opp
    win = 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
    return win + macro.GAP_LAMBDA * max(-1.0, min(1.0, gap / macro.GAP_CLIP))


def play(job: dict) -> dict:
    """One episode in this worker.

    job = {seed, me: policy, opp: {kind: 'tape', tape, ...} |
           {kind: 'policy', policy}, max_steps?}
    Returns {banks, reward, decisions?, opp_decisions?}.
    """
    if "env" not in _W:
        _worker_init()
    env: ServeEnv = _W["env"]
    seed = job["seed"]
    me = _build_ns(job["me"])
    opp = job["opp"]
    try:
        if opp["kind"] == "tape":
            obs = env.reset(seed, opp_tape=opp["tape"], opp_seat=1)
            while not obs.get("done"):
                a = me["agent"](seat_view(obs, 0))
                obs = env.step(a)
        else:
            on = _build_ns(opp["policy"])
            obs = env.reset(seed)
            while not obs.get("done"):
                a0 = me["agent"](seat_view(obs, 0))
                a1 = on["agent"](seat_view(obs, 1))
                obs = env.step_both(a0, a1)
    except Exception as e:  # noqa: BLE001 -- one broken episode, not the run
        env.close()
        return {"error": f"{type(e).__name__}: {e}", "seed": seed}
    b0 = float(obs["farms"][0]["money"])
    b1 = float(obs["farms"][1]["money"])
    out = {"banks": [b0, b1], "reward": gap_reward(b0, b1), "seed": seed}
    if job["me"].get("record"):
        out["decisions"] = me.get("_TRACKP_DECISIONS", [])
    return out


class Runner:
    """Process pool of persistent serve workers."""

    def __init__(self, jobs: int = 0):
        from concurrent.futures import ProcessPoolExecutor
        self.n = jobs or max(2, (os.cpu_count() or 8) - 2)
        self.pool = ProcessPoolExecutor(max_workers=self.n,
                                        initializer=_worker_init)

    def run(self, jobs: list) -> list:
        return list(self.pool.map(play, jobs, chunksize=1))

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    import time
    from kaggriculture.trackp import league
    anchors = league.load_anchors()
    jobs = [{"seed": 1000 + i, "me": {}, "opp": {"kind": "tape",
             "tape": anchors[i % len(anchors)]["tape"]}}
            for i in range(12)]
    r = Runner()
    t0 = time.time()
    res = r.run(jobs)
    dt = time.time() - t0
    ok = [x for x in res if "banks" in x]
    print(json.dumps({"episodes": len(ok), "wall_s": round(dt, 1),
                      "ep_per_s": round(len(ok) / dt, 2),
                      "mean_reward": round(sum(x["reward"] for x in ok)
                                           / max(1, len(ok)), 3)}))
    r.close()
