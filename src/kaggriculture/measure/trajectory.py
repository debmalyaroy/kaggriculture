"""Optimise a full 720-turn action sequence, and ship it as a tape.

This is the architecture the top of the leaderboard actually uses. Decoding
their public notebooks shows a `main.py` that carries a recorded episode plus a
slip-recovery layer, not a policy -- viable because the environment is
deterministic apart from weed spawns and shop-unlock order, so one good
trajectory transfers between games.

The difference here: the recording is **ours**. We play our own agent to get a
base trajectory, then improve that sequence directly, which is a search a
closed-loop policy cannot do because it cannot see the whole episode at once.

Mutations are deliberately concentrated on the market channel. Two independent
top-ten write-ups isolate sale timing as the remaining edge -- one measured a
promotion that changed 20 field turns and 112 market turns -- so moving a sale
one turn earlier is a higher-yield move than reshuffling a walk.

    python -m kaggriculture.measure.trajectory --record --vs opponents/tape_90036815_s1.py
    python -m kaggriculture.measure.trajectory --optimise --iterations 300
    python -m kaggriculture.measure.trajectory --build --out agents/v12_tape.py

The optimiser keeps a change only when it survives every seed in the panel, so
a tape cannot be tuned into one lucky weed layout.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import copy
import json
import os
import random
import statistics
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.measure.opponents as OPP  # noqa: E402

WORK = os.path.join(ROOT, "data", "traj")
TAPE = os.path.join(WORK, "tape.json")
STATE = os.path.join(WORK, "state.json")
TPD = 24
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER"]


# ------------------------------------------------------------------ record --

def record(agent_path, opponent, seed, steps=720):
    """Play our agent and keep every action it produced."""
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    env.run([os.path.abspath(agent_path), os.path.abspath(opponent)])
    tape = _actions_from_env(env, 0)
    reward = float(env.steps[-1][0]["reward"] or 0)
    return tape, reward


def _actions_from_env(env, seat):
    """steps[i]['action'] is the action that produced state i, so turn i is i+1."""
    steps = env.steps
    blank = {"farmer": ["PASS"], "hands": [], "market": []}
    out = []
    for i in range(len(steps)):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        a = nxt[seat].get("action") if (nxt and seat < len(nxt)) else None
        out.append(copy.deepcopy(a) if isinstance(a, dict) else dict(blank))
    return out


# ------------------------------------------------------------------ mutate --

def mutate(tape, rng, market_only=True):
    """One local edit to the sequence. Returns (new_tape, description)."""
    n = len(tape)
    sell_turns = [i for i, a in enumerate(tape)
                  if any(isinstance(o, list) and o and o[0] == "SELL"
                         for o in (a.get("market") or []))]
    kind = rng.choice(
        ["shift", "resize", "insert", "drop", "reorder"] if sell_turns
        else ["insert", "insert", "insert"])
    new = copy.deepcopy(tape)

    if kind == "shift" and sell_turns:
        t = rng.choice(sell_turns)
        orders = [o for o in new[t]["market"] if o[0] == "SELL"]
        if not orders:
            return new, "noop"
        order = rng.choice(orders)
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        dest = max(0, min(n - 1, t + delta))
        new[t]["market"] = [o for o in new[t]["market"] if o is not order]
        new[dest].setdefault("market", [])
        if len(new[dest]["market"]) < 10:
            new[dest]["market"].append(list(order))
        return new, f"shift {order[1]} t{t}->{dest}"

    if kind == "resize" and sell_turns:
        t = rng.choice(sell_turns)
        for o in new[t]["market"]:
            if o[0] == "SELL" and len(o) >= 3:
                factor = rng.choice([0.5, 0.75, 1.5, 2.0])
                o[2] = max(1, int(round(int(o[2]) * factor)))
                return new, f"resize {o[1]} t{t} x{factor}"
        return new, "noop"

    if kind == "drop" and sell_turns:
        t = rng.choice(sell_turns)
        before = len(new[t]["market"])
        new[t]["market"] = [o for o in new[t]["market"] if o[0] != "SELL"]
        return new, f"drop sells t{t} ({before}->{len(new[t]['market'])})"

    if kind == "reorder" and sell_turns:
        t = rng.choice(sell_turns)
        rng.shuffle(new[t]["market"])
        return new, f"reorder t{t}"

    # insert: sell something a bit earlier than the tape currently does
    t = rng.randrange(TPD, n)
    item = rng.choice(PRODUCTS)
    qty = rng.choice([2, 4, 6, 8, 12])
    new[t].setdefault("market", [])
    if len(new[t]["market"]) < 10:
        new[t]["market"].append(["SELL", item, qty])
    return new, f"insert SELL {item} x{qty} t{t}"


# ---------------------------------------------------------------- evaluate --

def build_tape_agent(tape, path, label="candidate"):
    payload = base64.b85encode(zlib.compress(
        json.dumps(tape, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    src = OPP._TAPE_TEMPLATE.format(
        label=label, episode="optimised", seat=0, rewards="n/a",
        result="candidate", steps=len(tape), payload=payload)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


def _play(job):
    left, right, seed, steps = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def score(paths, opponents, seeds, seed0, pool):
    jobs, meta = [], []
    for ci, p in enumerate(paths):
        for opp in opponents:
            o = os.path.abspath(opp)
            for s in range(seeds):
                jobs.append((p, o, seed0 + s, 720))
                meta.append((ci, 0))
                jobs.append((o, p, seed0 + s, 720))
                meta.append((ci, 1))
    out = list(pool.map(_play, jobs))
    rows = {i: [] for i in range(len(paths))}
    for (ci, seat), (r0, r1) in zip(meta, out):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        rows[ci].append((mine, theirs))
    res = []
    for ci, data in rows.items():
        wins = sum(1 for m, t in data if m > t)
        res.append({"i": ci, "win": wins / max(1, len(data)),
                    "margin": statistics.mean(m - t for m, t in data),
                    "bank": statistics.mean(m for m, _ in data),
                    "worst": min(m - t for m, t in data)})
    return res


# ---------------------------------------------------------------- optimise --

def optimise(opponents, iterations, batch, seeds, seed0, workers, rng_seed=3,
             market_only=True, resume=False):
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(TAPE):
        raise SystemExit("no base tape -- run: python -m kaggriculture.measure.trajectory --record")
    tape = json.load(open(TAPE, encoding="utf-8"))
    rng = random.Random(rng_seed)
    hist = []
    if resume and os.path.exists(STATE):
        st = json.load(open(STATE, encoding="utf-8"))
        hist = st.get("history", [])

    base_path = build_tape_agent(tape, os.path.join(WORK, "base.py"), "base")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        best = score([base_path], opponents, seeds, seed0, pool)[0]
        print(f"base tape: win {best['win'] * 100:.0f}%  margin {best['margin']:+,.0f}  "
              f"bank {best['bank']:,.0f}")
        kept = 0
        for it in range(iterations):
            cands, notes = [], []
            for k in range(batch):
                cand, note = mutate(tape, rng, market_only)
                p = build_tape_agent(cand, os.path.join(WORK, f"m{k}.py"), note)
                cands.append((p, cand, note))
                notes.append(note)
            res = score([c[0] for c in cands], opponents, seeds, seed0, pool)
            res.sort(key=lambda r: -(r["win"] * 1e6 + r["margin"]))
            top = res[0]
            # A change is kept only if it beats the incumbent on the aggregate
            # *and* does not make the worst seed worse -- a tape tuned into one
            # weed layout is not a tape.
            if (top["win"] * 1e6 + top["margin"]) > (best["win"] * 1e6 + best["margin"]) \
                    and top["worst"] >= best["worst"] - 1e-9:
                tape = cands[top["i"]][1]
                best = top
                kept += 1
                json.dump(tape, open(TAPE, "w", encoding="utf-8"),
                          separators=(",", ":"))
                print(f"  it {it + 1:>4}  KEEP  {cands[top['i']][2]:<28} "
                      f"win {top['win'] * 100:>3.0f}%  margin {top['margin']:>+10,.0f}")
            hist.append({"it": it, "win": top["win"], "margin": top["margin"],
                         "kept": kept})
            if (it + 1) % 10 == 0:
                json.dump({"history": hist}, open(STATE, "w", encoding="utf-8"))
                print(f"  it {it + 1:>4}  best margin {best['margin']:>+10,.0f}  "
                      f"kept {kept}  ({(time.time() - t0) / 60:.1f} min)")
    json.dump({"history": hist}, open(STATE, "w", encoding="utf-8"))
    print(f"\nkept {kept} of {iterations} proposals; "
          f"final margin {best['margin']:+,.0f}, bank {best['bank']:,.0f}")
    return best


def main():
    # The ladder's engine, or nothing this prints means anything.
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "v9_cem.py"))
    ap.add_argument("--vs", nargs="+",
                    default=[os.path.join(ROOT, "opponents", "tape_90036815_s1.py")])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--optimise", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "v12_tape.py"))
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=7700)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-drift", type=float, default=0.15,
                    help="reject a recording that does not replay to within this")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    if args.record:
        tape, reward = record(args.agent, args.vs[0], args.seed)
        # A recording made while the box is saturated is not a recording. Under
        # heavy load the engine substitutes default actions on turns that miss
        # actTimeout, and the same agent on the same seed recorded $16,348
        # instead of $80,289 while a 300-episode mine had every core. Replay the
        # tape once and refuse it if it does not reproduce what it recorded.
        json.dump(tape, open(TAPE, "w", encoding="utf-8"), separators=(",", ":"))
        check = build_tape_agent(tape, os.path.join(WORK, "verify.py"), "verify")
        got, _theirs = _play((check, os.path.abspath(args.vs[0]), args.seed, 720))
        drift = abs(got - reward) / max(1.0, reward)
        print(f"recorded {len(tape)} turns from {os.path.basename(args.agent)} "
              f"(banked ${reward:,.0f}) -> {os.path.relpath(TAPE, ROOT)}")
        print(f"tape replays to ${got:,.0f}  ({drift * 100:.1f}% drift)")
        if drift > float(args.max_drift):
            print(f"REJECTED: drift over {float(args.max_drift) * 100:.0f}%. "
                  f"Is something else using the CPU? Re-record on an idle box.")
            return 1
        return 0
    if args.optimise:
        optimise(args.vs, args.iterations, args.batch, args.seeds, args.seed0,
                 args.workers, resume=args.resume)
        return 0
    if args.build:
        tape = json.load(open(TAPE, encoding="utf-8"))
        build_tape_agent(tape, args.out, "optimised trajectory")
        print(f"wrote {os.path.relpath(args.out, ROOT)} "
              f"({os.path.getsize(args.out):,} bytes)")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
