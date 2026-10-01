"""Play two Python agent FILES on the Rust engine via `kagg serve`.

The engine steps in Rust (bit-identical to the official interpreter, verified
by tests/test_rust_engine.py); the agents run in THIS process as real Python
-- every closed-loop layer (floor guard, weed repair, adaptive sell) behaves
exactly as on the ladder. This is the substrate for fast tournament rounds:
same fidelity as evaluate.py, a fraction of the wall time, and one light
process instead of a pure-Python engine per game.

    python src/serve_match.py A.py B.py --seed 3
    python src/serve_match.py A.py B.py --seeds 3,4,5 --compare-official

--compare-official replays each seed on the official vendored engine too and
asserts the BANKS MATCH EXACTLY -- the substrate-equivalence check the crown
migration is gated on. Independent of src/trackp/ by design (isolation).
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import subprocess
import sys
import time

def _resolve_kagg():
    """Cross-platform engine path: KAGG_BIN env, else a Windows kagg.exe or a
    Linux kagg (rented VM), else the .exe default."""
    env = os.environ.get("KAGG_BIN")
    if env and os.path.exists(env):
        return env
    for c in ("rustengine/kagg.exe", "rustengine/kagg",
              "rustengine/target/release/kagg", "rustengine/target/release/kagg.exe"):
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "rustengine", "kagg.exe")


KAGG = _resolve_kagg()
# exec-loaded agents (no __file__) resolve their own kagg via KAGG_BIN --
# without this the big-trackp searcher silently RUST-FAILs in every local
# instrument and the gate measures the bare chassis (found 2026-09-06)
if not os.environ.get("KAGG_BIN") and os.path.exists(KAGG):
    os.environ["KAGG_BIN"] = KAGG


# --- Official invocation contract (matches vendor kaggle_environments) -------
# episodeSteps from the env spec (kaggriculture.json). The official framework
# solicits agent actions for steps 0..episodeSteps-2 and reads the final bank
# from the terminal step-(episodeSteps-1) state (core.py:296 marks DONE at
# step>=episodeSteps-1; interpreter:960 at step>=episodeSteps-2). See S5 fix
# in run_match.
EPISODE_STEPS = 720

# Resolved default configuration the official runner passes to agents as the
# second positional arg (scalar defaults from kaggriculture.json; `seed` is
# cleared by the interpreter, marketParams defaults to {}). Agents that take a
# 2nd `configuration` param read from this. (S3)
CONFIG = {
    "episodeSteps": 720, "actTimeout": 1, "boardSize": 10, "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24, "shedCapacity": 100,
    "weedSpawnChance": 0.005, "townShopUnlockInterval": 3, "townShopSellInterval": 4,
    "townCenterSellInterval": 24, "farmHandCostMult": 1, "marketParams": {},
}


class _Struct(dict):
    """dict with attribute access -- a local mirror of kaggle_environments'
    Struct so serve agents see the SAME object type as on the ladder, without
    importing vendor on the fast path."""
    def __init__(self, **entries):
        entries = {k: v for k, v in entries.items() if k != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        self[attr] = value


def _structify(o):
    """Recursively REBUILD into Struct/list -- deep-copies (per-seat isolation,
    S4) AND gives attribute access, byte-for-byte matching vendor structify."""
    if isinstance(o, list):
        return [_structify(v) for v in o]
    if isinstance(o, dict):
        return _Struct(**{k: _structify(v) for k, v in o.items()})
    return o


def _call(agent, obs):
    """Invoke like the official runner: args truncated to the agent's
    co_argcount (single-arg agents get obs only; 2-arg agents also get a fresh
    structified configuration). (S3)"""
    args = [obs, _structify(CONFIG)]
    code = getattr(agent, "__code__", None)
    if code is not None and hasattr(code, "co_argcount"):
        args = args[: code.co_argcount]
    return agent(*args)


def load_agent(path):
    ns = {"__file__": os.path.abspath(path), "__name__": "__agent__"}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    if "agent" not in ns:
        raise SystemExit(f"{path} does not export agent(obs)")
    return ns["agent"]


def _tok(op):
    if not op:
        return "PASS"
    return " ".join(str(t) for t in op)


def action_to_line(action):
    """The single-seat tape line `farmer\\thands\\tmarket` kagg parses."""
    action = action or {}
    farmer = _tok(action.get("farmer"))
    hands = ";".join(_tok(h) for h in (action.get("hands") or []))
    # Every market entry keeps its slot: `[]` (and any non-list) encodes as an
    # empty segment, which kagg's `engine::positional` keeps. Dropping it
    # re-pairs the per-index market race (cha22-lineage agents use [] holes).
    market = ";".join(" ".join(str(t) for t in o) if isinstance(o, (list, tuple)) else ""
                      for o in (action.get("market") or []))
    return f"{farmer}\t{hands}\t{market}"


def obs_for(seat, js):
    # structify deep-copies farms/market/town/private into a fresh per-seat
    # object (S4: no cross-seat aliasing / in-place mutation), exactly as the
    # official runner does via structify(observation).
    return _structify({
        "step": js["step"], "day": js["day"], "hour": js["hour"],
        "player": seat, "farms": js["farms"], "market": js["market"],
        "town": js["town"], "private": js["private"][seat]})


class Serve:
    def __init__(self):
        if not os.path.exists(KAGG):
            raise SystemExit(f"kagg.exe not built: {KAGG}")
        self.p = subprocess.Popen([KAGG, "serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  encoding="utf-8", cwd=ROOT)

    def cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
        except OSError:
            pass
        self.p.wait(timeout=10)


def run_match(agent_a, agent_b, seed, srv=None):
    own = srv is None
    if own:
        srv = Serve()
    try:
        js = srv.cmd(f"RESET {seed}")
        # S5 -- last-turn parity. Official solicits actions for steps
        # 0..episodeSteps-2 (0..718) and reads the final bank from the
        # step-(episodeSteps-1) state. Driving serve to its own `done`
        # (step>=720) would apply an EXTRA step-719 action and an extra
        # day-30 end-of-day, banking differently for last-turn liquidators.
        # Stop one step early so serve == official exactly.
        while js["step"] < EPISODE_STEPS - 1:
            la = action_to_line(_call(agent_a, obs_for(0, js)))
            lb = action_to_line(_call(agent_b, obs_for(1, js)))
            js = srv.cmd(f"STEP2 {la}\x1e{lb}")
            if "error" in js:
                raise RuntimeError(js["error"])
        money = [f.get("money") for f in js["farms"]]
        return float(money[0]), float(money[1])
    finally:
        if own:
            srv.close()


def official_banks(path_a, path_b, seed):
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([load_agent(path_a), load_agent(path_b)])
    return tuple(float(env.steps[-1][i]["reward"]) for i in range(2))


def eval_mode(args):
    """evaluate.py-compatible eval: same CLI shape, same RESULT lines.

    For each opponent, n games: seed0 + i//2, candidate alternating seats --
    the same (opponent, seed, seat) grid for every candidate, which is what
    makes tournament cells comparable. Emits the 10-field RESULT lines
    win_metric.parse_eval() reads, so refresh_cycle's tournament can swap
    the engine substrate without touching a parser.
    """
    pa = os.path.join(ROOT, args.a) if not os.path.isabs(args.a) else args.a
    srv = Serve()
    try:
        for opp in args.vs:
            po = os.path.join(ROOT, opp) if not os.path.isabs(opp) else opp
            w = d = lo = 0
            bank = opp_bank = 0.0
            for i in range(args.n):
                seed = args.seed0 + i // 2
                A, B = load_agent(pa), load_agent(po)
                if i % 2 == 0:
                    b0, b1 = run_match(A, B, seed, srv)
                else:
                    b1, b0 = run_match(B, A, seed, srv)
                bank += b0
                opp_bank += b1
                w += b0 > b1
                d += b0 == b1
                lo += b0 < b1
            g = max(1, args.n)
            score = (w + 0.5 * d) / g
            print(f"RESULT\t{po}\t{score:.4f}\t{w}\t{d}\t{lo}\t{args.n}"
                  f"\t{bank / g:.1f}\t{opp_bank / g:.1f}"
                  f"\t{bank - opp_bank:.1f}", flush=True)
    finally:
        srv.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated; overrides --seed")
    ap.add_argument("--compare-official", action="store_true")
    # evaluate.py-compatible eval mode (RESULT lines):
    ap.add_argument("--vs", nargs="*", default=None)
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=60000)
    ap.add_argument("--no-record", action="store_true")   # accepted, no-op
    ap.add_argument("--workers", type=int, default=0)     # accepted, no-op
    args = ap.parse_args()
    if args.vs:
        return eval_mode(args)
    if not args.b:
        ap.error("need a second agent (or --vs for eval mode)")

    pa = os.path.join(ROOT, args.a) if not os.path.isabs(args.a) else args.a
    pb = os.path.join(ROOT, args.b) if not os.path.isabs(args.b) else args.b
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed if args.seed is not None else 3])

    mismatches = 0
    srv = Serve()
    try:
        for seed in seeds:
            a = load_agent(pa)          # fresh module state per episode
            b = load_agent(pb)
            t0 = time.time()
            banks = run_match(a, b, seed, srv)
            dt_serve = time.time() - t0
            line = (f"SERVE\t{seed}\t{banks[0]:.0f}\t{banks[1]:.0f}"
                    f"\t{dt_serve:.1f}s")
            if args.compare_official:
                t0 = time.time()
                off = official_banks(pa, pb, seed)
                dt_off = time.time() - t0
                ok = banks == off
                mismatches += 0 if ok else 1
                line += (f"\tOFFICIAL\t{off[0]:.0f}\t{off[1]:.0f}"
                         f"\t{dt_off:.1f}s\t{'MATCH' if ok else 'MISMATCH'}")
            print(line, flush=True)
    finally:
        srv.close()
    if args.compare_official:
        print(f"substrate equivalence: {len(seeds) - mismatches}/{len(seeds)} "
              f"exact-bank matches")
        return 1 if mismatches else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
