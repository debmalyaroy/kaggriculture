"""Track P's own official-engine match runner (fresh code, no shared parse).

Runs paired matches on the VENDORED OFFICIAL interpreter -- the only engine
whose numbers count. Both seats, fixed seeds, per-turn latency capture.
Emits stable machine lines:  RESULT\tseed\tseat\tbank_a\tbank_b\twinner

Usage:
  python src/trackp/arena.py A.py --vs B.py -n 4 [--seed0 11] [--swap]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))

from kaggriculture.trackp import common  # noqa: E402


def load_agent(path: str):
    """Import an agent file in its own namespace; returns agent(obs)."""
    name = "trackp_arena_" + os.path.basename(path).replace(
        ".", "_").replace("-", "_") + str(abs(hash(os.path.abspath(path))))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


class _PassAgent:
    def __call__(self, obs):
        return {"farmer": ["PASS"], "hands": [], "market": []}


def make_env(seed: int):
    common.vendored_env()
    from kaggle_environments import make
    return make("kaggriculture",
                configuration={"episodeSteps": common.EPISODE_STEPS,
                               "actTimeout": 60, "runTimeout": 1000000,
                               "seed": seed},
                info={"seed": seed})


def play(agent_a, agent_b, seed: int, collect_latency=None,
         collect_actions=None):
    """One episode, A in seat 0. Returns (bank_a, bank_b)."""
    env = make_env(seed)
    env.reset(2)
    agents = (agent_a, agent_b)
    while not env.done:
        actions = []
        step = int(dict(env.state[0].observation).get("step", 0))
        for i in (0, 1):
            obs = dict(env.state[i].observation)
            view = {"step": step,
                    "day": obs.get("day", step // common.TURNS_PER_DAY),
                    "hour": obs.get("hour", step % common.TURNS_PER_DAY),
                    "player": i, "farms": obs["farms"],
                    "market": obs["market"], "town": obs["town"],
                    "private": obs["private"]}
            t0 = time.perf_counter()
            try:
                a = agents[i](view)
            except Exception:  # noqa: BLE001 -- a crash is a PASS, like live
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            dt = time.perf_counter() - t0
            if collect_latency is not None:
                collect_latency[i].append(dt)
            if not isinstance(a, dict):
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            if collect_actions is not None:
                collect_actions[i].append(a)
            actions.append(a)
        env.step(actions)
    return tuple(float(env.state[i].observation.farms[i]["money"])
                 for i in (0, 1))


def paired(path_a: str, path_b: str, n: int = 4, seed0: int = 11,
           latency: bool = False) -> dict:
    """n seeds x both seats. Returns the full result table."""
    rows = []
    lat = [[], []] if latency else None
    agent_a = load_agent(path_a) if path_a != "pass" else _PassAgent()
    agent_b = load_agent(path_b) if path_b != "pass" else _PassAgent()
    for k in range(n):
        seed = seed0 + 1000 * k
        for seat in (0, 1):
            la = [[], []] if latency else None
            if seat == 0:
                ba, bb = play(agent_a, agent_b, seed, la)
            else:
                bb, ba = play(agent_b, agent_a, seed, la)
            if la:
                lat[0].extend(la[seat])
                lat[1].extend(la[1 - seat])
            w = "A" if ba > bb else ("B" if bb > ba else "D")
            rows.append({"seed": seed, "seat": seat, "bank_a": ba,
                         "bank_b": bb, "winner": w})
            print(f"RESULT\t{seed}\t{seat}\t{ba}\t{bb}\t{w}", flush=True)
    wins = sum(r["winner"] == "A" for r in rows)
    draws = sum(r["winner"] == "D" for r in rows)
    out = {"a": path_a, "b": path_b, "games": len(rows), "wins_a": wins,
           "draws": draws, "score_a": (wins + 0.5 * draws) / len(rows),
           "rows": rows}
    if latency and lat[0]:
        out["latency_a_ms"] = {
            "mean": round(1000 * statistics.fmean(lat[0]), 2),
            "p99": round(1000 * sorted(lat[0])[
                int(0.99 * (len(lat[0]) - 1))], 2),
            "max": round(1000 * max(lat[0]), 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_a")
    ap.add_argument("--vs", required=True)
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=11)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    res = paired(a.agent_a, a.vs, a.n, a.seed0, a.latency)
    summary = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(summary, indent=1))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
