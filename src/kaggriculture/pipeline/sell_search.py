"""Sell-schedule search on the Rust batch engine -- the route factory MVP.

Lever L5/R3 of docs/history/pair-improvement-plan.md: everything we ship today was
mined from opponents, so opponents can mine it back. This tool searches for
a route the ladder has never seen -- keeping the crowned route's FARM PLAN
fixed (planting, watering, animals, hiring: the part where a bad mutation is
a silent illegal no-op) and optimizing only its SELL SCHEDULE (which step,
what quantity: the part the 1.32.7 hinge repriced and the part that is
legality-safe by construction, because market orders never depend on board
position -- an infeasible SELL just fails per-unit in `_commit_unit`).

Search: (1+lambda) elitist evolution over the market channel -- move, split,
merge and resize SELL ops -- scored on `kagg batch` (bit-exact engine,
~140 ep/s/core) against a fixed panel of elite opponent tapes with common
random seeds. The winner then faces the FINAL GATE on fresh holdout seeds:
a paired sign test vs the unmutated base (win_metric.paired_test). Only a
significant winner is rendered into an agent; the agent still has to earn a
release through the normal tournament like any candidate.

    python src/sell_search.py                      # base = the live flagship
    python src/sell_search.py --base 92513718_s1 --iters 200
    python src/sell_search.py --smoke              # 5-iteration wiring check

Compute: ONE Rust process, --threads 3 by default (the machine-wide 3-wide
engine cap; never raise it while a cycle tournament is running).
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# A scheduled task's console is cp1252; a team name like "张广麒"
# must never kill a search run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

KAGG = os.path.join(ROOT, "rustengine", "kagg.exe")
WORK = os.path.join(ROOT, ".local", "sell_search")
OUTDIR = os.path.join(ROOT, "models", "factory")
PRODUCTS = {"STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
            "CARROT", "WHEAT", "FERTILIZER"}


# ------------------------------------------------------------------- tapes

def action_line(a):
    if not isinstance(a, dict):
        return "PASS\t\t"
    farmer = a.get("farmer")
    fs = " ".join(str(t) for t in farmer) if isinstance(farmer, list) and \
        farmer else "PASS"
    hs = ";".join(" ".join(str(t) for t in h)
                  for h in (a.get("hands") or []) if isinstance(h, list) and h)
    ms = ";".join(" ".join(str(t) for t in o)
                  for o in (a.get("market") or []) if isinstance(o, list) and o)
    return f"{fs}\t{hs}\t{ms}"


def write_tape(actions, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("SEED 0\n")
        fh.write("\n".join(action_line(a) for a in actions))
    return path


# ------------------------------------------------------------------- panel

def pick_panel(idx, base_rec, n):
    """Elite opponents: best bank per team, base's team excluded, fresh first."""
    best_by_team = {}
    for r in idx["routes"].values():
        team = r.get("team", "?")
        if team in ("?", base_rec.get("team")):
            continue
        cur = best_by_team.get(team)
        if cur is None or float(r.get("bank", 0)) > float(cur.get("bank", 0)):
            best_by_team[team] = r
    ranked = sorted(best_by_team.values(),
                    key=lambda r: (-float(r.get("bank", 0)), r["id"]))
    return ranked[:n]


def default_base():
    """The live flagship's route id, from the newest v*_route.py header."""
    import glob as g
    agents = sorted(g.glob(os.path.join(ROOT, "agents", "v*_route.py")),
                    key=os.path.getmtime, reverse=True)
    for p in agents:
        head = open(p, encoding="utf-8").read(4000)
        m = re.search(r"(\d{6,}_s[01])", head)
        if m:
            return m.group(1), os.path.basename(p)
    raise SystemExit("no route agent found to take a base from; pass --base")


# ------------------------------------------------------------------- eval

def batch_eval_paired(cand_tapes, pairs, threads):
    """Score each candidate against (opponent, seed) PAIRS, both seats.

    `batch_eval` plays the full cross product of panel x seeds, which is
    right when opponents are open-loop schedules and seeds are free. It is
    WRONG for the hard band (2026-09-04): there each tape is one team's
    recorded play in ONE world, and cross-producting replays a 2400-rated
    schedule in a world it never saw -- exactly the tape degradation the
    band's certification exists to exclude. Pairing keeps every tape in its
    own world and is len(seeds)x cheaper.

    `pairs`: [(tape_path, seed)]. Same return shape as batch_eval.
    """
    jobs, meta = [], []
    for ci, ct in enumerate(cand_tapes):
        for oi, (ot, s) in enumerate(pairs):
            jobs.append(f"{s}\t{ct}\t{ot}")
            meta.append((ci, oi, s, 0))
            jobs.append(f"{s}\t{ot}\t{ct}")
            meta.append((ci, oi, s, 1))
    jp = os.path.join(WORK, "jobs_paired.tsv")
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jobs))
    out = subprocess.run([KAGG, "batch", jp, str(threads)],
                         capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise SystemExit(f"kagg batch failed: {out.stderr[:400]}")
    res = [{"cells": {}, "n": 0, "acc": 0.0, "msum": 0.0, "mcells": {}}
           for _ in cand_tapes]
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        ci, oi, s, flipped = meta[int(parts[0])]
        mine, theirs = (float(parts[3]), float(parts[2])) if flipped             else (float(parts[2]), float(parts[3]))
        sc = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        res[ci]["msum"] += mine - theirs
        res[ci]["cells"].setdefault((oi, s), []).append(sc)
        res[ci]["n"] += 1
        res[ci]["acc"] += sc
        res[ci]["mcells"][(oi, s, flipped)] = mine - theirs
        # ABSOLUTE banks too. A paired score cannot tell a candidate that
        # earned a win from one that merely DESYNCED the opponent's tape:
        # on 2026-09-04 a base scored 47-9 on the hard band with cells like
        # "84,810 vs 0" and opponent banks down 69,630. Callers guard on
        # these against the opponent's RECORDED bank.
        res[ci].setdefault("banks", {})[(oi, s, flipped)] = (mine, theirs)
    for r in res:
        r["cells"] = {k: sum(v) / len(v) for k, v in r["cells"].items()}
        r["score"] = r["acc"] / r["n"] if r["n"] else 0.0
        r["margin"] = r["msum"] / r["n"] if r["n"] else 0.0
    return res


def batch_eval(cand_tapes, panel_tapes, seeds, threads):
    """Score each candidate tape vs every (opponent, seed), both seats.

    Returns per candidate: {"score": mean, "cells": {(opp_i, seed): score}}.
    Draws are 0.5 -- the ladder's currency, never margin.
    """
    jobs, meta = [], []
    for ci, ct in enumerate(cand_tapes):
        for oi, ot in enumerate(panel_tapes):
            for s in seeds:
                jobs.append(f"{s}\t{ct}\t{ot}")
                meta.append((ci, oi, s, 0))
                jobs.append(f"{s}\t{ot}\t{ct}")
                meta.append((ci, oi, s, 1))
    jp = os.path.join(WORK, "jobs.tsv")
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jobs))
    out = subprocess.run([KAGG, "batch", jp, str(threads)],
                         capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise SystemExit(f"kagg batch failed: {out.stderr[:400]}")
    res = [{"cells": {}, "n": 0, "acc": 0.0, "msum": 0.0}
           for _ in cand_tapes]
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        i = int(parts[0])
        ci, oi, s, flipped = meta[i]
        mine, theirs = (float(parts[3]), float(parts[2])) if flipped \
            else (float(parts[2]), float(parts[3]))
        sc = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        res[ci]["msum"] += mine - theirs
        cell = res[ci]["cells"].setdefault((oi, s), [])
        cell.append(sc)
        res[ci]["n"] += 1
        res[ci]["acc"] += sc
        # Per-cell margin keyed by the full (opp, seed, seat) cell: paired
        # child-vs-incumbent deltas on identical cells cancel the opponent-
        # and seed-driven variance (+/-10k) that swamps candidate effects
        # (+/-3k) in unpaired means.
        res[ci].setdefault("mcells", {})[(oi, s, flipped)] = mine - theirs
    for r in res:
        r["score"] = r["acc"] / r["n"] if r["n"] else 0.0
        r["margin"] = r["msum"] / r["n"] if r["n"] else 0.0
        r["cells"] = {k: sum(v) / len(v) for k, v in r["cells"].items()}
    return res


def batch_eval_cells(cand_tapes, cells, threads):
    """Score each candidate on explicit (opp_tape, seed, seat) CELLS.

    The realized-world fix (2026-09-07): shop unlocks share the per-day
    RNG with weeds, so the WORLD a game realizes depends on the seed AND
    both action streams AND the seat order — a seed's PASS-drive world
    (seed_bank.json) is not the world real play visits. Per-world
    evolution must therefore group by cells whose REALIZED shops match,
    and a cell fixes the seat: cross-producting or force-pairing both
    seats would mix realizations back in.

    `cells`: [(opp_tape, seed, seat)] with seat 0 = candidate first.
    Same return shape as batch_eval (score/cells/margin/mcells/banks).
    """
    jobs, meta = [], []
    for ci, ct in enumerate(cand_tapes):
        for oi, (ot, s, seat) in enumerate(cells):
            if seat == 0:
                jobs.append(f"{s}\t{ct}\t{ot}")
            else:
                jobs.append(f"{s}\t{ot}\t{ct}")
            meta.append((ci, oi, s, seat))
    jp = os.path.join(WORK, "jobs_cells.tsv")
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jobs))
    out = subprocess.run([KAGG, "batch", jp, str(threads)],
                         capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise SystemExit(f"kagg batch failed: {out.stderr[:400]}")
    res = [{"cells": {}, "n": 0, "acc": 0.0, "msum": 0.0, "mcells": {},
            "banks": {}} for _ in cand_tapes]
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        ci, oi, s, seat = meta[int(parts[0])]
        mine, theirs = (float(parts[3]), float(parts[2])) if seat \
            else (float(parts[2]), float(parts[3]))
        sc = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        res[ci]["msum"] += mine - theirs
        res[ci]["cells"][(oi, s, seat)] = sc
        res[ci]["n"] += 1
        res[ci]["acc"] += sc
        res[ci]["mcells"][(oi, s, seat)] = mine - theirs
        res[ci]["banks"][(oi, s, seat)] = (mine, theirs)
    for r in res:
        r["score"] = r["acc"] / r["n"] if r["n"] else 0.0
        r["margin"] = r["msum"] / r["n"] if r["n"] else 0.0
    return res


# --------------------------------------------------------------- mutation

def _sell_slots(actions):
    out = []
    for step, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        for j, o in enumerate(a.get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" \
                    and o[1] in PRODUCTS:
                out.append((step, j))
        # normalize: every dict step gets a market list we can insert into
        if a.get("market") is None:
            a["market"] = []
    return out


def _insert(actions, step, op):
    step = max(0, min(len(actions) - 1, step))
    a = actions[step]
    if not isinstance(a, dict):
        return False
    mk = a.setdefault("market", [])
    if len(mk) >= 10:            # engine truncates at maxMarketOrdersPerTurn
        return False
    mk.append(op)
    return True


def _econ_slots(actions):
    """Field-economy orders: labor and purchases (Track-P factory space)."""
    out = []
    for step, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        for j, o in enumerate(a.get("market") or []):
            if isinstance(o, list) and o and o[0] in (
                    "HIRE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"):
                out.append((step, j))
    return out


def mutate_field(actions, rng, window=48, n_ops=2):
    """Track-P phase-2 operators: perturb the field ECONOMY, not the sells.

    Labor is the strongest measured correlate of final bank (74k-game
    field study) and purchase depth bounds every downstream plan, so the
    operators are: add/remove a HIRE, shift a purchase in time, resize a
    purchase. All legality is settled by SIMULATION -- a purchase without
    its downstream unit ops just measures worse and dies in selection.
    """
    acts = copy.deepcopy(actions)
    for _ in range(rng.randint(1, n_ops)):
        kind = rng.choice(("hire_add", "hire_del", "buy_shift",
                           "buy_resize", "buy_resize"))
        if kind == "hire_add":
            step = rng.randrange(0, min(600, len(acts)))
            _insert(acts, step, ["HIRE"])
            continue
        slots = _econ_slots(acts)
        if not slots:
            return acts
        if kind == "hire_del":
            hires = [(s, j) for s, j in slots
                     if acts[s]["market"][j][0] == "HIRE"]
            if hires:
                s, j = hires[rng.randrange(len(hires))]
                del acts[s]["market"][j]
            continue
        buys = [(s, j) for s, j in slots
                if acts[s]["market"][j][0] != "HIRE"]
        if not buys:
            continue
        s, j = buys[rng.randrange(len(buys))]
        op = acts[s]["market"][j]
        if kind == "buy_shift":
            tgt = s + rng.randint(-window, window)
            if _insert(acts, tgt, list(op)):
                del acts[s]["market"][j]
        elif kind == "buy_resize" and len(op) >= 3:
            try:
                q = int(op[2])
            except (TypeError, ValueError):
                continue
            dq = max(1, int(round(q * 0.25)))
            nq = q + rng.choice((-dq, dq))
            if nq >= 1:
                acts[s]["market"][j] = [op[0], op[1], nq]
    return acts


def mutate(actions, rng, window=60, n_ops=2, field_ops=False):
    if field_ops and rng.random() < 0.5:
        return mutate_field(actions, rng)
    acts = copy.deepcopy(actions)
    for _ in range(rng.randint(1, n_ops)):
        slots = _sell_slots(acts)
        if not slots:
            return acts
        step, j = slots[rng.randrange(len(slots))]
        op = acts[step]["market"][j]
        kind = rng.choice(("move", "move", "split", "resize", "merge"))
        if kind == "move":
            tgt = step + rng.randint(-window, window)
            if _insert(acts, tgt, list(op)):
                del acts[step]["market"][j]
        elif kind == "split" and int(op[2]) >= 2:
            q = int(op[2])
            q1 = rng.randint(1, q - 1)
            tgt = step + rng.randint(-window, window)
            if _insert(acts, tgt, [op[0], op[1], q - q1]):
                acts[step]["market"][j] = [op[0], op[1], q1]
        elif kind == "resize":
            q = int(op[2])
            dq = max(1, int(round(q * 0.2)))
            nq = q + rng.choice((-dq, dq))
            if nq >= 1:
                acts[step]["market"][j] = [op[0], op[1], nq]
        elif kind == "merge":
            mates = [(s2, j2) for s2, j2 in slots
                     if (s2, j2) != (step, j)
                     and acts[s2]["market"][j2][1] == op[1]
                     and abs(s2 - step) <= 48]
            if mates:
                s2, j2 = mates[rng.randrange(len(mates))]
                o2 = acts[s2]["market"][j2]
                first, second = ((step, j), (s2, j2)) if step <= s2 \
                    else ((s2, j2), (step, j))
                acts[first[0]]["market"][first[1]] = \
                    [op[0], op[1], int(op[2]) + int(o2[2])]
                del acts[second[0]]["market"][second[1]]
    return acts


# ------------------------------------------------------------------ build

def build_agent(actions, base_rec, out_path, label):
    import base64
    import datetime as dt
    import zlib
    import kaggriculture.engine.tape_runtime as tape_runtime
    payload = base64.b85encode(zlib.compress(
        json.dumps(actions, separators=(",", ":")).encode("utf-8"),
        9)).decode("ascii")
    src = tape_runtime.TEMPLATE.format(
        label=label, route_id=f"{base_rec['id']}+sellsearch",
        team=base_rec.get("team", "?"), episode=base_rec.get("episode", "?"),
        seat=base_rec.get("seat", 0), bank=base_rec.get("bank", 0.0),
        opp_bank=base_rec.get("opp_bank", 0.0),
        built=dt.date.today().isoformat(), payload=payload,
        weed_catchup=8, impact_slots=True, mirror_tiebreak=True,
        floor_guard=False, endgame_pull=False, guard_ratio=0.0,
        swap_advance=False, sell_first=True, feed_pin=True,
        premium_lead=False, deposit_advance=False, floor_seller=False,
        branchpack="")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    code = subprocess.run([sys.executable, "-m", "py_compile", out_path])
    if code.returncode != 0:
        raise SystemExit(f"built agent does not compile: {out_path}")
    return out_path


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="route id to optimize (default: the "
                                   "live flagship's embedded route)")
    ap.add_argument("--base-tape", default=None,
                    help="optimize a raw .tape file (e.g. the shipped "
                         "rustengine/src/bandit_egg.tape) instead of a route id")
    ap.add_argument("--panel-size", type=int, default=6)
    ap.add_argument("--hard-panel", action="store_true",
                    help="search against the HARD band: 56 real 2300+ ladder "
                         "opponents in their OWN recorded seeds, paired. "
                         "Overrides --panel-ids/--panel-size/--seeds.")
    ap.add_argument("--panel-ids", nargs="*", default=None,
                    help="explicit opponent route ids (Track-P: the "
                         "current-elite tapes; overrides the bank-ranked "
                         "pick, which is the anti-predictive measure)")
    ap.add_argument("--panel-tapes", nargs="*", default=None,
                    help="explicit opponent .tape FILES (globs ok, e.g. the "
                         "harvested strong_tapes) — evolve against these; "
                         "highest priority (overrides hard-panel/panel-ids)")
    ap.add_argument("--field-ops", action="store_true",
                    help="Track-P phase 2: include labor/purchase "
                         "mutations (hire add/del, buy shift/resize)")
    ap.add_argument("--seeds", type=int, default=8,
                    help="common random seeds per evaluation")
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lam", type=int, default=16,
                    help="mutants per generation")
    ap.add_argument("--patience", type=int, default=40,
                    help="stop after this many generations without a gain")
    ap.add_argument("--threads", type=int, default=3,
                    help="kagg batch threads -- the machine-wide engine cap")
    ap.add_argument("--holdout-seeds", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "agents",
                                                  "factory_sell.py"))
    ap.add_argument("--resume", action="store_true",
                    help="continue from .local/sell_search/best_actions.json "
                         "(same base only)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the wiring")
    args = ap.parse_args()
    if args.smoke:
        args.iters, args.lam, args.seeds, args.panel_size = 3, 4, 2, 2
        args.holdout_seeds = 4
    if not os.path.exists(KAGG):
        raise SystemExit("rustengine not built: cargo build --release")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(20260818)

    import kaggriculture.data.routes as R
    idx = R.load_index()
    if args.base_tape:
        # load base actions straight from a raw .tape (farmer\thands\tmarket),
        # numeric tokens -> ints to match the route-schema mutators expect.
        def _tok(s):
            out = []
            for t in s.split():
                try:
                    out.append(int(t))
                except ValueError:
                    out.append(t)
            return out
        base_actions = []
        for i, ln in enumerate(open(args.base_tape, encoding="utf-8").read().splitlines()):
            if i == 0 and ln.startswith("SEED"):
                continue
            p = (ln.split("\t") + ["", "", ""])[:3]
            base_actions.append({
                "farmer": _tok(p[0]) or ["PASS"],
                "hands": [_tok(h) for h in p[1].split(";") if h.strip()],
                "market": [_tok(o) for o in p[2].split(";") if o.strip()],
            })
        base_id, base_src = os.path.basename(args.base_tape), "--base-tape"
        base_rec = {"team": "__ours__", "bank": 0}
        print(f"base: {base_id} (raw tape, {len(base_actions)} rows)")
    else:
        if args.base:
            base_id, base_src = args.base, "--base"
        else:
            base_id, base_src = default_base()
        base_rec = idx["routes"].get(base_id)
        if base_rec is None:
            raise SystemExit(f"route {base_id!r} not in the index")
        base_actions = R.load_route(base_id)
        print(f"base: {base_id} (from {base_src}), bank {base_rec.get('bank')}")

    hard = None
    if args.panel_tapes:
        import glob as _glob
        tapes = []
        for pat in args.panel_tapes:
            p = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
            tapes += sorted(_glob.glob(p)) if any(c in pat for c in "*?[") else [p]
        panel_tapes = [t for t in tapes if os.path.exists(t)]
        if not panel_tapes:
            raise SystemExit(f"no --panel-tapes matched: {args.panel_tapes}")
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        print(f"EXPLICIT PANEL: {len(panel_tapes)} tapes x {args.seeds} seeds")
    elif args.hard_panel:
        # 56 REAL games against 2300+ opposition, each opponent tape kept in
        # the world it was actually played in (2026-09-04). The default panel
        # is mined elite routes on arbitrary seeds; those tape worlds run
        # 56-63k median shared bank against a ladder at ~85k.
        import json as _json
        _m = _json.load(open(os.path.join(
            ROOT, ".local", "hardband", "factory_panel", "manifest.json"),
            encoding="utf-8"))
        hard = [(o["tape"], o["seed"]) for o in _m["opponents"]]
        panel_tapes = [t for t, _s in hard]
        seeds = [sd for _t, sd in hard]
        print(f"HARD PANEL: {len(hard)} real 2300+ opponents, paired with "
              f"their recorded ladder seeds")
    else:
        if args.panel_ids:
            panel = [idx["routes"][i] for i in args.panel_ids
                     if i in idx["routes"]]
            if not panel:
                raise SystemExit("none of --panel-ids are in the index")
        else:
            panel = pick_panel(idx, base_rec, args.panel_size)
        print(f"panel: {[(r['id'], r.get('team')) for r in panel]}")
        panel_tapes = []
        for i, r in enumerate(panel):
            panel_tapes.append(write_tape(R.load_route(r["id"]),
                                          os.path.join(WORK, f"opp_{i}.tape")))
        seeds = list(range(args.seed0, args.seed0 + args.seeds))

    def _ev(cands, sds):
        """Paired on the hard panel; full cross product otherwise."""
        if hard is not None:
            keep = set(sds)
            return batch_eval_paired(
                cands, [(t, sd) for t, sd in hard if sd in keep],
                args.threads)
        return batch_eval(cands, panel_tapes, sds, args.threads)
    base_tape = write_tape(base_actions, os.path.join(WORK, "base.tape"))
    if hard is not None:
        seeds, hold_override = seeds[0::2], seeds[1::2]
    else:
        hold_override = None
    base_eval = _ev([base_tape], seeds)[0]
    print(f"base score on the panel: {base_eval['score']:.4f} "
          f"({base_eval['n']} games)")

    ckpt = os.path.join(WORK, "best_actions.json")
    best_actions, best_score = base_actions, base_eval["score"]
    if args.resume and os.path.exists(ckpt):
        try:
            saved = json.load(open(ckpt, encoding="utf-8"))
            if saved.get("base") == base_id and saved.get("actions"):
                rt = write_tape(saved["actions"],
                                os.path.join(WORK, "resume.tape"))
                rs = _ev([rt], seeds)[0]["score"]
                if rs > best_score:
                    best_actions, best_score = saved["actions"], rs
                    print(f"resumed from checkpoint at {rs:.4f}")
        except (OSError, ValueError):
            pass
    stale = 0
    for gen in range(args.iters):
        cands = [mutate(best_actions, rng, field_ops=args.field_ops)
                 for _ in range(args.lam)]
        tapes = [write_tape(c, os.path.join(WORK, f"c{i}.tape"))
                 for i, c in enumerate(cands)]
        evals = _ev(tapes, seeds)
        gi = max(range(len(evals)), key=lambda i: evals[i]["score"])
        if evals[gi]["score"] > best_score:
            best_score = evals[gi]["score"]
            best_actions = cands[gi]
            stale = 0
            print(f"  gen {gen:>4}: NEW BEST {best_score:.4f}", flush=True)
            with open(ckpt, "w", encoding="utf-8") as fh:
                json.dump({"base": base_id, "score": best_score,
                           "gen": gen, "actions": best_actions}, fh)
        else:
            stale += 1
            if gen % 10 == 0:
                print(f"  gen {gen:>4}: best {best_score:.4f} "
                      f"(gen-best {evals[gi]['score']:.4f}, stale {stale})")
        if stale >= args.patience:
            print(f"  no gain in {args.patience} generations -- stopping")
            break

    # FINAL GATE: fresh holdout seeds, paired sign test vs the base.
    import kaggriculture.measure.win_metric as WM
    hseeds = hold_override if hold_override else list(range(args.seed0 + 10000,
                        args.seed0 + 10000 + args.holdout_seeds))
    best_tape = write_tape(best_actions, os.path.join(WORK, "best.tape"))
    hb, hc = _ev([base_tape, best_tape], hseeds)
    cells = sorted(set(hb["cells"]) & set(hc["cells"]))
    t = WM.paired_test([hc["cells"][k] for k in cells],
                       [hb["cells"][k] for k in cells])
    passed = bool(cells) and t["score_diff"] > 0 and t["significant"]
    print(f"\nHOLDOUT: candidate {hc['score']:.4f} vs base {hb['score']:.4f} "
          f"on {len(cells)} fresh cells; diff {t['score_diff']:+.4f}, "
          f"discordant {t['better_a']}-{t['better_b']}, "
          f"p={t['p_value']:.4f}")
    verdict = {"when": __import__("datetime").datetime.now()
               .isoformat(timespec="seconds"),
               "base": base_id, "search_score": best_score,
               "base_search_score": base_eval["score"],
               "holdout": {"candidate": hc["score"], "base": hb["score"],
                           "cells": len(cells),
                           "diff": t["score_diff"], "p": t["p_value"]},
               "passed": passed,
               "panel": (["hard-band:%d" % len(hard)] if hard is not None
                         else ([os.path.basename(t) for t in panel_tapes]
                               if args.panel_tapes else [r["id"] for r in panel]))}
    if passed:
        out = build_agent(best_actions, base_rec, args.out,
                          os.path.basename(args.out))
        verdict["agent"] = os.path.relpath(out, ROOT)
        print(f"PASS -- built {out}; it still has to win the tournament")
    else:
        print("HOLDOUT FAIL -- no agent built (searched gains did not "
              "survive fresh seeds)")
    json.dump(verdict, open(os.path.join(OUTDIR, "sell_search.json"), "w",
                            encoding="utf-8"), indent=1)
    print(f"wrote models/factory/sell_search.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
