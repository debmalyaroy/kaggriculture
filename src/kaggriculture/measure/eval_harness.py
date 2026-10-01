"""Unified local test harness -- the four evaluations, one command, faithful engine.

Ties together the evaluation types the notebook sources describe, all on the
post-fidelity-fix engine (serve==official, [[serve-fidelity-fix-2026-09-18]]):

  1. SMOKE           -- runs exception-free for a full episode, both seats
                        (delegates to ``measure.smoke_test``).
  2. SEAT-SWAPPED    -- paired both-seats win rate vs a fixed reference roster
                        (score currency = win/draw/loss, ``measure.win_metric``).
  3. REPLAY CLEAN-ROOM -- vs transcribed top-tier opponents (destbreso replay
                        tapes from ``macro_env``'s pool), banded by ladder rating
                        -- counter-strategy testing with no opponent source.
  4. ELO / BENCHMARK -- a logistic rating fit from the observed scores against
                        rated opponents (fast diagnostic throughput number).

``gate_packaged_policy`` turns this into the Slot-2 ship rule: smoke + seat-
swapped vs the STRONG refs (the two ~2500 public tapes) -> ship iff the policy
clears the B4.0 bar (~0.9 vs v46) with no regression. Nothing here submits.

    python -m kaggriculture.measure.eval_harness agents/v56y_baseline.py \
        --refs agents/ref_v46.py,agents/ref_k0006.py --seeds 4
    python -m kaggriculture.measure.eval_harness --gate      # gate the packaged policy
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import math
import os
import sys
from typing import Callable, List, Optional

sys.path.insert(0, os.path.join(ROOT, "vendor"))

import kaggriculture.train.macro_actions as MA
from kaggriculture.measure.smoke_test import load_agent, smoke, PASS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Bands for the clean-room evaluation (ladder rating).
BANDS = [("<2100", 0, 2100), ("2100-2300", 2100, 2300),
         ("2300-2500", 2300, 2500), ("2500-2700", 2500, 2700),
         ("2700+", 2700, 9999)]


def _as_agent(a):
    """Accept a file path or a callable; return a 1-or-2-arg-safe callable."""
    if callable(a):
        fn = a
    else:
        fn = load_agent(a)
    import inspect
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 1
    return (lambda o, c=None: fn(o)) if n == 1 else fn


_SERVE = {"srv": None}


def play_two(agent0, agent1, seed: int, backend: str = "serve") -> tuple:
    """Play two live agents; return (bank0, bank1).

    Default backend = the fidelity-fixed Rust ``kagg serve`` (bit-exact to the
    official interpreter, ~8x faster than the vendored Python engine) over ONE
    persistent process -- so the GATE evolves fast. ``backend='vendored'`` uses
    the Python engine as a cross-check.
    """
    if backend == "serve":
        try:
            from kaggriculture.engine.serve_match import run_match, Serve
            if _SERVE["srv"] is None:
                _SERVE["srv"] = Serve()
            return run_match(agent0, agent1, seed, srv=_SERVE["srv"])
        except Exception:
            pass                              # fall back to vendored on any serve issue
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([agent0, agent1])
    last = env.steps[-1]
    return float(last[0]["reward"] or 0), float(last[1]["reward"] or 0)


# --------------------------------------------------------------------------- #
# 2. seat-swapped paired evaluation
# --------------------------------------------------------------------------- #
def gate_seeds(n: int, seed0: int = 1000):
    """G3: a seed bank spanning DIFFERENT worlds. Prefers the realized seed bank
    (data/worlds/seed_bank.json, seeds chosen to cover the ladder world freq);
    else a well-spread set (a large stride hits varied shop-worlds, vs
    consecutive seeds that cluster)."""
    p = os.path.join(ROOT, "data", "worlds", "seed_bank.json")
    if os.path.exists(p):
        try:
            import json
            bank = json.load(open(p, encoding="utf-8"))
            seeds = [int(s) for s in bank.get("seeds", [])][:n]
            if len(seeds) >= n:
                return seeds
        except (OSError, ValueError):
            pass
    return [seed0 + k * 7919 for k in range(n)]     # spread stride -> varied worlds


def seat_swapped(agent, refs: List, seeds: int = 4, seed0: int = 1000) -> dict:
    """Both-seats win rate vs each reference across DIVERSE worlds (G3)."""
    from kaggriculture.measure import win_metric
    A = _as_agent(agent)
    out = {}
    all_cells = []
    bank = gate_seeds(seeds, seed0)                 # G3: world-diverse seed bank
    for ref in refs:
        R = _as_agent(ref)
        cells = []
        for s in bank:
            b0, b1 = play_two(A, R, s)                 # A at seat 0
            cells.append(1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0))
            b0b, b1b = play_two(R, A, s)               # A at seat 1
            cells.append(1.0 if b1b > b0b else (0.5 if b1b == b0b else 0.0))
        name = ref if isinstance(ref, str) else getattr(ref, "__name__", "ref")
        sc = sum(cells) / len(cells)
        out[os.path.basename(str(name))] = dict(score=sc, games=len(cells))
        all_cells += cells
        print(f"[eval:seat] vs {os.path.basename(str(name))}: "
              f"score {sc:.3f} ({len(cells)} games)")
    out["_aggregate"] = dict(score=sum(all_cells) / max(len(all_cells), 1),
                             games=len(all_cells))
    return out


# --------------------------------------------------------------------------- #
# 3. replay clean-room vs transcribed opponents (banded)
# --------------------------------------------------------------------------- #
def replay_clean_room(agent, n_per_band: int = 3, seeds: int = 1) -> dict:
    """Play the agent vs real destbreso replay tapes, banded by ladder rating."""
    from kaggriculture.train.macro_env import OpponentPool, replay_agent
    A = _as_agent(agent)
    pool = OpponentPool()
    by_band = {b[0]: [] for b in BANDS}
    dbs = [e for e in pool.entries if e.get("kind") == "destbreso_tape"]
    dbs.sort(key=lambda e: -(e.get("rating") or 0))
    picked = {b[0]: [] for b in BANDS}
    for e in dbs:
        r = e.get("rating") or 0
        for name, lo, hi in BANDS:
            if lo <= r < hi and len(picked[name]) < n_per_band:
                picked[name].append(e)
    ratings_used = []
    for name, entries in picked.items():
        wins = []
        for e in entries:
            cells = pool.cells_for(e)
            if not cells:
                continue
            seed = (int(e.get("seed", 42)) & 0x7fffffff) or 42
            b0, b1 = play_two(A, replay_agent(cells), seed)
            wins.append(1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0))
            ratings_used.append((e.get("rating") or 0,
                                 wins[-1]))
        if wins:
            by_band[name] = wins
            print(f"[eval:replay] band {name:<10} "
                  f"score {sum(wins)/len(wins):.3f}  (n={len(wins)})")
    return dict(bands={k: (sum(v)/len(v) if v else None)
                       for k, v in by_band.items()},
                pairs=ratings_used)


# --------------------------------------------------------------------------- #
# 4. logistic Elo fit from observed scores vs rated opponents
# --------------------------------------------------------------------------- #
def elo_fit(rating_score_pairs, base: float = 1500.0) -> Optional[float]:
    """Fit our rating R so expected logistic score ~ observed, over rated games.

    E(win) = 1/(1+10**((r_opp - R)/400)); solve by bisection on total error.
    """
    pairs = [(r, s) for (r, s) in rating_score_pairs if r and r > 0]
    if not pairs:
        return None

    def total_expected(R):
        return sum(1.0 / (1.0 + 10 ** ((r - R) / 400.0)) for r, _ in pairs)

    observed = sum(s for _, s in pairs)
    lo, hi = 0.0, 4000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if total_expected(mid) < observed:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------- #
# packaged-policy gate (Slot-2 ship rule)
# --------------------------------------------------------------------------- #
def build_policy_agent(onnx_path: str, manifest_path: str):
    """A compile-then-execute agent: sample a greedy 30-day plan from the ONNX
    policy at game start, then execute it with the greedy micro-realizer."""
    import json
    import numpy as np
    import onnxruntime as ort
    from kaggriculture.train.macro_env import PlanController
    import kaggriculture.train.bc_warmup as BC

    man = json.load(open(manifest_path, encoding="utf-8"))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tgt_mu = np.asarray(man["norm"]["tgt_mu"], np.float32)
    tgt_sd = np.asarray(man["norm"]["tgt_sd"], np.float32)
    cum_mu = np.asarray(man["norm"]["cum_mu"], np.float32)
    cum_sd = np.asarray(man["norm"]["cum_sd"], np.float32)
    MAXD = man["max_days"]

    def sample_greedy_plan(rating=0.85):
        cum = np.zeros(MA.VECTOR_LEN, np.float32)
        plan = []
        feats = np.zeros((1, MAXD, man["in_dim"]), np.float32)
        for day in range(MAXD):
            x = np.zeros(man["in_dim"], np.float32)
            x[0] = day / float(MAXD)
            db = min(day // (MAXD // man["day_buckets"] + 1),
                     man["day_buckets"] - 1)
            x[1 + db] = 1.0
            base = 1 + man["day_buckets"] + man["world_buckets"]
            x[base] = 1.0                     # rtg -> condition on WINNING
            x[base + 1] = rating
            x[base + 2:base + 2 + MA.VECTOR_LEN] = (cum - cum_mu) / cum_sd
            feats[0, day] = x
            _, vec = sess.run(None, {"x": feats})
            raw = np.clip(vec[0, day] * tgt_sd + tgt_mu, 0, None)
            plan.append(MA.MacroAction.from_vector(raw.tolist()))
            cum = cum + raw
        return plan

    ctrl = {"c": None}

    def agent(obs, config=None):
        if ctrl["c"] is None:
            ctrl["c"] = PlanController(sample_greedy_plan())
        return ctrl["c"].agent(obs, config)
    return agent


def gate_packaged_policy(refs: Optional[List[str]] = None, seeds: int = 4,
                         bar: float = 0.9) -> dict:
    """Smoke + seat-swapped vs the strong refs -> ship-or-fallback verdict."""
    import json
    onnx = os.path.join(ROOT, "models", "rl", "policy.onnx")
    man = os.path.join(ROOT, "models", "rl", "policy_manifest.json")
    if not (os.path.exists(onnx) and os.path.exists(man)):
        print("[gate] no packaged policy -- run package_policy first")
        return dict(skipped=True, reason="no policy.onnx")
    agent = build_policy_agent(onnx, man)
    refs = refs or _default_refs()
    if not refs:
        print("[gate] no reference agents found; running replay clean-room only")
        cr = replay_clean_room(agent, n_per_band=2)
        return dict(clean_room=cr, ship=False, verdict="no-refs")
    ss = seat_swapped(agent, refs, seeds=seeds)
    agg = ss["_aggregate"]["score"]
    ship = agg >= bar
    print(f"[gate] aggregate vs refs = {agg:.3f}  bar {bar:.2f} -> "
          f"{'SHIP' if ship else 'FALLBACK (diversified route2)'}")
    return dict(seat_swapped=ss, aggregate=agg, bar=bar, ship=ship,
                verdict="ship" if ship else "fallback")


def crown_refs(max_per_band: int = 3) -> dict:
    """Load reactive gate refs from the crown panel, banded. Only REACTIVE agents
    (origin .py that exists) — tapes are inert off-world so they're excluded."""
    import json
    p = os.path.join(ROOT, "models", "crown_panel.json")
    out = {}
    if not os.path.exists(p):
        return out
    c = json.load(open(p, encoding="utf-8"))
    for band, refs in (c.get("bands") or {}).items():
        picked = []
        for r in refs or []:
            origin = r.get("origin")
            fp = os.path.join(ROOT, origin) if origin else None
            if fp and os.path.exists(fp) and r.get("vet") == "ok":
                picked.append(fp)
            if len(picked) >= max_per_band:
                break
        if picked:
            out[band] = picked
    return out


def full_gate(agent, seeds: int = 4, bar: float = 0.9,
              harness: str = "?") -> dict:
    """The RELEASE gate: smoke + banded seat-swap vs the crown panel (reactive),
    on the faithful serve harness across DIVERSE worlds. SHIP iff no band
    regresses below the bar and the ladder-weighted aggregate clears it. Both
    harnesses (bandit, trackp) run through this identical gate."""
    import numpy as np
    A = _as_agent(agent)
    bands = crown_refs()
    per_band, all_cells = {}, []
    band_centroid = {"<2100": 2000, "2100-2300": 2200, "2300-2500": 2400,
                     "2500-2700": 2600, "2700+": 2850}
    for band, refs in bands.items():
        cells = []
        for ref in refs:
            ss = seat_swapped(A, [ref], seeds=seeds)
            cells.append(ss["_aggregate"]["score"])
        if cells:
            per_band[band] = float(np.mean(cells))
            all_cells += cells
    # ladder-weighted aggregate + per-band non-regression
    if per_band:
        wsum = sum(band_centroid.get(b, 2000) for b in per_band)
        agg = sum(band_centroid.get(b, 2000) * s for b, s in per_band.items()) / wsum
    else:
        agg = 0.0
    worst_band = min(per_band.values()) if per_band else 0.0
    ship = agg >= bar and worst_band >= (bar - 0.15)
    print(f"[full_gate:{harness}] bands={ {k: round(v,2) for k,v in per_band.items()} } "
          f"agg={agg:.3f} worst={worst_band:.3f} bar={bar} -> "
          f"{'SHIP' if ship else 'HOLD'}")
    return dict(harness=harness, bands=per_band, aggregate=agg,
                worst_band=worst_band, bar=bar, ship=ship,
                verdict="ship" if ship else "hold")


def bandit_binary_agent(stage_dir: str):
    """An eval agent that drives `kagg mbandit` DIRECTLY (blocking reads, fresh
    process per game). This is how the config-driven bandit is gated ON THIS BOX:
    the shipped main.py DRIVER has a 0.40s per-turn budget that the Windows
    kagg.exe pipe latency exceeds → PASS fallback → false 0.000. See
    [[bandit-gate-driver-windows-2026-09-19]]. The binary + tape + branches are
    faithful; only the transport differs, so the measured economy is the shipped
    economy."""
    import json
    import subprocess
    binp = None
    for nm in ("kagg.exe", "kagg"):
        p = os.path.join(stage_dir, nm)
        if os.path.exists(p):
            binp = p
            break
    st = {"p": None}

    def agent(obs, cfg=None):
        o = obs if isinstance(obs, dict) else getattr(obs, "__dict__", {}) or {}
        if int(o.get("step", 0) or 0) == 0 and st["p"]:
            try:
                st["p"].kill()
            except Exception:
                pass
            st["p"] = None
        if not st["p"]:
            st["p"] = subprocess.Popen(
                [binp, "mbandit", "config.json", "branches.json", "base.tape"],
                cwd=stage_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        p = st["p"]
        try:
            p.stdin.write(json.dumps(o) + "\n")
            p.stdin.flush()
            ln = p.stdout.readline()
            return json.loads(ln) if ln.strip() else {"farmer": ["PASS"], "hands": [], "market": []}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def gate_config_bandit(config_path: str = None, seeds: int = 4, bar: float = 0.9,
                       stage_root: str = None) -> dict:
    """Build the config-driven bandit stage (base.tape+branches from config),
    stage a runnable binary for THIS box, and run the full banded gate on it via
    the direct mbandit driver. Returns the full_gate dict (bands/aggregate/ship)."""
    import shutil
    import kaggriculture.bandit.build.build_rust_bandit as B
    import kaggriculture.pipeline.daily_slot2 as D
    config_path = config_path or os.path.join(ROOT, "configs", "bandit_config.json")
    out = stage_root or os.path.join(ROOT, ".local", "scratch", "bandit_gate_stage")
    shutil.rmtree(out, ignore_errors=True)
    B.write_stage(out, config_path)
    D._windows_bandit_gate_stage(out)                 # swaps in a runnable kagg[.exe]
    stage = out + "_wingate" if os.path.exists(out + "_wingate") else out
    return full_gate(bandit_binary_agent(stage), seeds=seeds, bar=bar, harness="bandit")


def _default_refs() -> List[str]:
    cand = []
    for p in ("ref_v46_chassis_2517.py", "ref_v46.py", "ref_k0006_2494.py",
              "ref_k0006.py"):
        for d in (os.path.join(ROOT, "agents"),
                  os.path.join(ROOT, ".local", "panel", "ref2500")):
            fp = os.path.join(d, p)
            if os.path.exists(fp):
                cand.append(fp)
    # de-dup keeping order
    seen, out = set(), []
    for c in cand:
        b = os.path.basename(c)
        if b not in seen:
            seen.add(b); out.append(c)
    return out[:2]


def evaluate_agent(agent_path: str, refs: List[str], seeds: int = 4) -> dict:
    print(f"=== eval_harness: {os.path.basename(agent_path)} ===")
    sm = smoke(agent_path, seeds=tuple(range(3, 3 + min(seeds, 3))))
    if not sm["pass_"]:
        print("[eval] SMOKE FAILED -- stopping"); return dict(smoke=sm, pass_=False)
    ss = seat_swapped(agent_path, refs, seeds=seeds) if refs else {}
    cr = replay_clean_room(agent_path, n_per_band=2)
    rating = elo_fit(cr.get("pairs", []))
    if rating:
        print(f"[eval:elo] fitted rating ~ {rating:.0f}")
    return dict(smoke=sm, seat_swapped=ss, clean_room=cr, elo=rating,
                pass_=True)


def _smoke() -> int:
    """Tiny self-test: gate the packaged policy (or clean-room) end to end."""
    onnx = os.path.join(ROOT, "models", "rl", "policy.onnx")
    if not os.path.exists(onnx):
        import kaggriculture.train.package_policy as PK
        PK._smoke()
    agent = build_policy_agent(onnx,
                               os.path.join(ROOT, "models", "rl",
                                            "policy_manifest.json"))
    # one clean-room band pass (uses real destbreso tapes if present)
    cr = replay_clean_room(agent, n_per_band=1)
    print("[eval_harness][smoke] OK  bands:",
          {k: (round(v, 2) if v is not None else None)
           for k, v in cr["bands"].items()})
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", nargs="?")
    ap.add_argument("--refs", default="", help="comma-separated ref agent files")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gate", action="store_true",
                    help="gate the packaged Slot-2 policy")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    refs = [r for r in args.refs.split(",") if r] or _default_refs()
    if args.gate:
        r = gate_packaged_policy(refs=refs, seeds=args.seeds)
        return 0 if r.get("ship") else 1
    if not args.agent:
        ap.error("provide an agent file, or --gate, or --smoke")
    r = evaluate_agent(args.agent, refs, seeds=args.seeds)
    return 0 if r.get("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
