"""Play the ASSEMBLED compiled artefact on the OFFICIAL engine and measure it.

This loads `main.py` out of a staging directory EXACTLY as the tarball unpacks
it -- main.py beside the binary -- so the thing measured is the thing shipped,
including the subprocess spawn and the `__file__`-relative binary lookup.

    python src/trackp/compiled/harness.py --stage <dir> \
        --vs agents/v43.0_bandit.py data/gauntlet/pub_v16rc5.py --seeds 3,4,5

Reports per cell: own bank, opponent bank, engine status for both seats,
mean / p95 / WORST turn latency in ms, and how many turns the pure-Python
fallback answered (must be 0) plus how many actions the validator had to
repair (must be 0).

The engine is the vendored official interpreter, never the Rust one: the Rust
engine is a pre-ranker, and a compiled agent measured on the engine it embeds
would be marking its own homework.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_agent_fn(path):
    ns = {}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    if "agent" not in ns:
        raise SystemExit(f"{path} does not export agent(obs)")
    return ns["agent"]


class Rec:
    """Per-call wall times and a legality audit for one seat."""

    def __init__(self):
        self.ms = []
        self.illegal = 0
        self.hand_mismatch = 0
        self.over_orders = 0

    def summary(self):
        if not self.ms:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "worst": 0.0,
                    "illegal": 0, "hand_mismatch": 0, "over_orders": 0}
        s = sorted(self.ms)
        return {
            "n": len(s),
            "mean": sum(s) / len(s),
            "p50": s[len(s) // 2],
            "p95": s[int(len(s) * 0.95)],
            "worst": s[-1],
            "illegal": self.illegal,
            "hand_mismatch": self.hand_mismatch,
            "over_orders": self.over_orders,
        }


def timed(fn, rec):
    """Wrap an agent for timing WITHOUT changing what the engine sees.

    This must be a plain two-argument function, not a callable object:
    kaggle_environments introspects the agent's signature to decide whether to
    pass `configuration`, and a class instance made it pass only the
    observation. That silently crippled every opponent that reads config --
    pub_v16rc5 banked exactly its 3,000 starting money and the candidate
    looked like it was winning 100k games. An opponent that scores its
    starting money is a broken harness, never a result.
    """
    # Some published agents export `def agent(obs)` with no configuration
    # parameter (pub_v16rc5 does). The wrapper must keep the two-argument
    # signature the env introspects, and call the inner function with the
    # arity it actually has.
    try:
        import inspect
        takes_cfg = len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        takes_cfg = True

    def wrapped(observation, configuration):
        t0 = time.perf_counter()
        a = fn(observation, configuration) if takes_cfg else fn(observation)
        rec.ms.append((time.perf_counter() - t0) * 1000.0)
        try:
            me = int(observation["player"])
            n = len(observation["farms"][me]["hands"] or [])
            if len(a.get("hands") or []) != n:
                rec.hand_mismatch += 1
            if len(a.get("market") or []) > 10:
                rec.over_orders += 1
            for op in [a.get("farmer")] + list(a.get("hands") or []):
                if not op or not isinstance(op[0], str):
                    rec.illegal += 1
        except Exception:                                     # noqa: BLE001
            rec.illegal += 1
        return a
    return wrapped


def play(stage, opp_path, seed, seat, ref=None):
    """One cell. `ref` plays a plain Python agent file in OUR seat instead of
    the compiled artefact -- same engine, same seeds, same seats, so the two
    runs pair cell for cell and `win_metric.paired_test` is valid on them."""
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    from kaggle_environments import make

    if ref:
        rec = Rec()
        me = timed(load_agent_fn(ref), rec)
        stats = {k: (0 if not isinstance(v, str) else "")
                 for k, v in _ZERO_STATS.items()}
    else:
        # Reloading main.py gives a FRESH module (and a fresh STATS), but the
        # kagg process the old module spawned is still blocked on its stdin and
        # would leak for the whole run. Retire it explicitly.
        old = sys.modules.pop("compiled_main", None)
        if old is not None:
            try:
                old._kill("harness: next cell")
            except Exception:                                 # noqa: BLE001
                pass
        mod = load_module(os.path.join(stage, "main.py"), "compiled_main")
        rec = Rec()
        me = timed(mod.agent, rec)
        stats = None
    opp = load_agent_fn(opp_path)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    order = [me, opp] if seat == 0 else [opp, me]
    env.run(order)
    last = env.steps[-1]
    banks = [float(last[i]["reward"] or 0) for i in range(2)]
    status = [last[i]["status"] for i in range(2)]
    return {
        "seed": seed,
        "seat": seat,
        "opp": os.path.basename(opp_path),
        "bank": banks[seat],
        "opp_bank": banks[1 - seat],
        "status": status[seat],
        "opp_status": status[1 - seat],
        "steps": len(env.steps),
        "lat": rec.summary(),
        "stats": stats if stats is not None else dict(mod.STATS),
        "who": os.path.basename(ref) if ref else "artefact",
    }


# Shape of bridge.STATS, so a reference-agent row prints in the same columns.
_ZERO_STATS = {"turns": 0, "bridge": 0, "fallback": 0, "spawn_fail": 0,
               "timeouts": 0, "bad_reply": 0, "repairs": 0, "worst_ms": 0.0,
               "reason": "", "budget_ms": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default=os.path.join(
        ROOT, ".local", "candidates", "trackp_compiled", "stage_linux"))
    ap.add_argument("--vs", nargs="+", default=[
        os.path.join(ROOT, "agents", "v43.0_bandit.py"),
        os.path.join(ROOT, "data", "gauntlet", "pub_v16rc5.py"),
    ])
    ap.add_argument("--seeds", default="3,4,5")
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--json", default=None, help="write the raw rows here")
    ap.add_argument("--budget-ms", type=int, default=None,
                    help="Phase-B search budget per turn (sets TRACKP_BUDGET_MS "
                         "for both the bridge's watchdog and the binary). "
                         "Omit to use the value compiled into bridge.py.")
    ap.add_argument("--ref", default=None,
                    help="play this plain Python agent in our seat instead of "
                         "the compiled artefact. Same seeds, same seats, same "
                         "engine, so the rows pair with an artefact run for "
                         "win_metric.paired_test.")
    ap.add_argument("--workers", type=int, default=1,
                    help="cells to play concurrently, one process each. The "
                         "search budget is wall clock, so oversubscribing the "
                         "box measures a WEAKER searcher than the one that "
                         "ships -- keep this well under the core count.")
    ap.add_argument("--worst-gate", type=float, default=250.0,
                    help="worst-turn latency the transport gate allows, in ms. "
                         "The default 250 is 4x margin on the 1,000 ms "
                         "actTimeout; a larger SEARCH budget must raise it "
                         "DELIBERATELY, never by accident.")
    args = ap.parse_args()
    if args.budget_ms is not None:
        os.environ["TRACKP_BUDGET_MS"] = str(args.budget_ms)

    seeds = [int(s) for s in args.seeds.split(",")]
    seats = [int(s) for s in args.seats.split(",")]
    cells = [(args.stage, opp, seed, seat, args.ref)
             for opp in args.vs for seed in seeds for seat in seats]
    print(f"{'opponent':<22} {'seed':>4} {'seat':>4} {'bank':>9} {'opp':>9} "
          f"{'status':>8} {'mean':>7} {'p95':>7} {'worst':>8} {'fb':>4} "
          f"{'rep':>4}")

    def emit(r):
        lat = r["lat"]
        st = r["stats"]
        warn = ""
        if st["fallback"]:
            warn += f"   !! fallback: {st['reason']}"
        if r["opp_status"] != "DONE" or r["opp_bank"] <= 3000.0:
            # An opponent on exactly its starting money never acted: the cell
            # measures nothing.
            warn += f"   !! DEAD OPPONENT ({r['opp_status']})"
        print(f"{r['opp']:<22} {r['seed']:>4} {r['seat']:>4} {r['bank']:>9.0f} "
              f"{r['opp_bank']:>9.0f} {r['status']:>8} "
              f"{lat['mean']:>6.1f}m {lat['p95']:>6.1f}m "
              f"{lat['worst']:>7.1f}m {st['fallback']:>4} "
              f"{st['repairs']:>4}" + warn, flush=True)

    rows = []
    if args.workers > 1:
        # One PROCESS per cell, never one thread: main.py keeps module-level
        # state and owns a subprocess, so cells cannot share an interpreter.
        #
        # The search budget is WALL CLOCK, so a worker that is fighting for a
        # core buys fewer rollouts, not more milliseconds. Keep workers well
        # under the core count or the measurement quietly becomes a weaker
        # searcher than the one that would ship.
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(play, *c) for c in cells]
            for f in futs:
                r = f.result()
                rows.append(r)
                emit(r)
    else:
        for c in cells:
            r = play(*c)
            rows.append(r)
            emit(r)

    if rows:
        worst = max(r["lat"]["worst"] for r in rows)
        mean = sum(r["lat"]["mean"] for r in rows) / len(rows)
        fb = sum(r["stats"]["fallback"] for r in rows)
        rep = sum(r["stats"]["repairs"] for r in rows)
        bad = sum(1 for r in rows if r["status"] != "DONE")
        dead = sum(1 for r in rows
                   if r["opp_status"] != "DONE" or r["opp_bank"] <= 3000.0)
        banks = sorted(r["bank"] for r in rows)
        med = banks[len(banks) // 2]
        wins = sum(1 for r in rows if r["bank"] > r["opp_bank"])
        print("\n== per opponent ==")
        print(f"{'opponent':<22} {'cells':>5} {'W-D-L':>9} {'score':>6} "
              f"{'own med':>9} {'opp med':>9}   [clean cells only]")
        for opp in sorted({r["opp"] for r in rows}):
            sub = [r for r in rows
                   if r["opp"] == opp and r["status"] == "DONE"
                   and r["opp_status"] == "DONE" and r["opp_bank"] > 3000.0]
            if not sub:
                print(f"{opp:<22} {0:>5}   (no clean cells)")
                continue
            w = sum(1 for r in sub if r["bank"] > r["opp_bank"])
            d = sum(1 for r in sub if r["bank"] == r["opp_bank"])
            lo = len(sub) - w - d
            om = sorted(r["bank"] for r in sub)[len(sub) // 2]
            pm = sorted(r["opp_bank"] for r in sub)[len(sub) // 2]
            print(f"{opp:<22} {len(sub):>5} {w:>3}-{d}-{lo:<3} "
                  f"{(w + 0.5 * d) / len(sub):>6.3f} {om:>9,.0f} {pm:>9,.0f}")

        print("\n== GATE ==")
        print(f"  cells                 {len(rows)}  (wins {wins})")
        print(f"  status != DONE        {bad}                  "
              f"[gate: 0]")
        print(f"  dead-opponent cells   {dead}                  "
              f"[gate: 0 -- a cell whose opponent banked its starting money "
              f"measures nothing]")
        print(f"  fallback fires        {fb}                  [gate: 0]")
        print(f"  validator repairs     {rep}                  [gate: 0]")
        print(f"  mean turn latency     {mean:.2f} ms")
        print(f"  WORST turn latency    {worst:.2f} ms          "
              f"[gate: < {args.worst_gate:.0f} ms, i.e. "
              f"{1000.0 / max(worst, 1e-9):.1f}x margin on actTimeout]")
        print(f"  search budget         "
              f"{rows[0]['stats'].get('budget_ms', 0)} ms/turn")
        print(f"  median own bank       {med:,.0f}          [gate: >= 85,000]")
        print(f"  min own bank          {banks[0]:,.0f}")
        ok = bad == 0 and fb == 0 and rep == 0 and worst < args.worst_gate
        print(f"\n  transport gate: {'PASS' if ok else 'FAIL'}"
              f"   strength gate: {'PASS' if med >= 85000 else 'FAIL'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
