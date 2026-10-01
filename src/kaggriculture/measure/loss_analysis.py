"""Why did this model lose that game?

Post-mortem for a single episode. Plays (or reads) a match, tracks both banks
turn by turn, finds where the gap opened, and attributes it to what the two
sides were actually doing at the time.

    python -m kaggriculture.measure.loss_analysis --agent agents/agent_v4_*.py --vs agents/v2_tuned.py --seed 7400
    python -m kaggriculture.measure.loss_analysis --replay data/episodes/mine/12345.json --seat 0
    python -m kaggriculture.measure.loss_analysis --agent agents/agent_v4_*.py --scan 8      # find its losses
    python -m kaggriculture.measure.loss_analysis --json                                     # for the dashboard

The point is to answer "what should I change?", so the output is ordered by how
much money each cause is worth, not by when it happened. A loss is rarely one
bad turn; it is usually a structural gap -- fewer hands, a herd that starved on
day 18, produce left in inventories at the end -- that a per-turn diff buries.
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.progress as pr  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "losses")

SELLABLE = ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY", "POTATO",
            "MILK", "WOOL", "EGG", "FERTILIZER")


# ------------------------------------------------------------------ capture --

def play_and_record(agent, opponent, seed, steps=720):
    """Run one match, keeping a per-turn snapshot of both farms."""
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([agent, opponent])
    return env.steps


def _farm_snapshot(obs, seat):
    """The handful of numbers that explain a farm's trajectory."""
    try:
        farm = obs["farms"][seat]
    except (KeyError, IndexError, TypeError):
        return None
    tiles = farm.get("tiles") or []
    crops = animals = 0
    for t in tiles:
        if not isinstance(t, dict):
            continue
        if t.get("crop"):
            crops += 1
        if t.get("animal"):
            animals += 1
    return {
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands") or []),
        "crops": crops,
        "animals": animals,
        "quadrants": len(farm.get("unlocked_quadrants") or []),
    }


def trace(steps):
    """Per-turn series for both seats, from a live run or a stored replay."""
    out = []
    for i, step in enumerate(steps):
        if not step:
            continue
        obs = step[0].get("observation") or {}
        row = {"turn": i, "day": obs.get("day"), "hour": obs.get("hour")}
        for seat in (0, 1):
            src = step[seat].get("observation") if seat < len(step) else None
            snap = _farm_snapshot(src if src and "farms" in src else obs, seat)
            if snap:
                row[f"p{seat}"] = snap
        if "p0" in row and "p1" in row:
            row["gap"] = row["p0"]["money"] - row["p1"]["money"]
            out.append(row)
    return out


# ------------------------------------------------------------- attribution --

def analyse(steps, seat=0):
    """Where the game was lost, and to what.

    Returns a dict the dashboard renders directly.
    """
    tr = trace(steps)
    if not tr:
        return {"error": "no usable per-turn observations in this episode"}

    me, them = f"p{seat}", f"p{1 - seat}"
    final = tr[-1]
    result = final[me]["money"] - final[them]["money"]

    # Biggest single-day swing against us: where the game actually turned.
    by_day = {}
    for r in tr:
        d = r.get("day")
        if d is None:
            continue
        g = r[me]["money"] - r[them]["money"]
        cur = by_day.setdefault(d, {"first": g, "last": g})
        cur["last"] = g
    swings = sorted(((d, v["last"] - v["first"]) for d, v in by_day.items()),
                    key=lambda kv: kv[1])

    causes = []

    # 1. labour -- the single biggest lever in this game
    my_hands = max(r[me]["hands"] for r in tr)
    their_hands = max(r[them]["hands"] for r in tr)
    if their_hands > my_hands:
        causes.append({
            "cause": "fewer hands",
            "detail": (f"peak {my_hands} vs their {their_hands}. Labour is the "
                       f"binding constraint on how much land can be worked at "
                       f"all; every downstream number follows from it."),
            "weight": (their_hands - my_hands) * 4000.0,
        })

    # 2. a herd that died -- the classic silent killer
    peak_animals = max(r[me]["animals"] for r in tr)
    end_animals = tr[-1][me]["animals"]
    if peak_animals >= 3 and end_animals < peak_animals * 0.6:
        died_at = next((r["day"] for r in tr
                        if r[me]["animals"] < peak_animals * 0.6), "?")
        causes.append({
            "cause": "herd shrank",
            "detail": (f"peaked at {peak_animals} animals, ended with "
                       f"{end_animals}, first dropped around day {died_at}. "
                       f"Animals that starve cost their purchase price and "
                       f"every future yield."),
            "weight": (peak_animals - end_animals) * 2500.0,
        })

    # 3. land not opened
    my_q = max(r[me]["quadrants"] for r in tr)
    their_q = max(r[them]["quadrants"] for r in tr)
    if their_q > my_q:
        causes.append({
            "cause": "fewer quadrants unlocked",
            "detail": f"{my_q} vs their {their_q}: less land to work all game.",
            "weight": (their_q - my_q) * 6000.0,
        })

    # 4. crop density
    my_crops = sum(r[me]["crops"] for r in tr) / len(tr)
    their_crops = sum(r[them]["crops"] for r in tr) / len(tr)
    if their_crops > my_crops * 1.15:
        causes.append({
            "cause": "fewer tiles under crop",
            "detail": (f"averaged {my_crops:.1f} planted tiles vs their "
                       f"{their_crops:.1f}. Either the plan was too "
                       f"conservative or labour could not keep up with it."),
            "weight": (their_crops - my_crops) * 900.0,
        })

    # 5. late collapse
    if len(tr) > 50:
        late = tr[int(len(tr) * 0.85):]
        drift = (late[-1][me]["money"] - late[-1][them]["money"]) - \
                (late[0][me]["money"] - late[0][them]["money"])
        if drift < -1500:
            causes.append({
                "cause": "lost it in the endgame",
                "detail": (f"the gap moved ${drift:,.0f} against us over the "
                           f"last 15% of the game. Usually produce stranded in "
                           f"inventories: SELL only draws from the shed."),
                "weight": abs(drift),
            })

    causes.sort(key=lambda c: -c["weight"])

    return {
        "seat": seat,
        "result": "won" if result > 0 else "lost",
        "final_bank": final[me]["money"],
        "opp_bank": final[them]["money"],
        "margin": result,
        "turns": len(tr),
        "worst_day": swings[0][0] if swings else None,
        "worst_day_swing": swings[0][1] if swings else None,
        "peak_hands": [my_hands, their_hands],
        "series": [{"turn": r["turn"], "day": r.get("day"),
                    "me": r[me]["money"], "them": r[them]["money"]}
                   for r in tr[::max(1, len(tr) // 240)]],
        "causes": causes,
    }


def report_text(a):
    if "error" in a:
        return a["error"]
    lines = [
        f"{a['result'].upper()}  ${a['final_bank']:,.0f} vs ${a['opp_bank']:,.0f} "
        f"({a['margin']:+,.0f})",
        f"turns {a['turns']}  |  worst day {a['worst_day']} "
        f"({a['worst_day_swing']:+,.0f} that day)",
        "",
    ]
    if not a["causes"]:
        lines.append("No structural cause stood out: this was decided by "
                     "small margins, not by one thing going wrong. Look at "
                     "market timing rather than the plan.")
    else:
        lines.append("Ranked causes, by how much money each is worth:")
        for c in a["causes"]:
            lines.append(f"  ~${c['weight']:>9,.0f}  {c['cause']}")
            lines.append(f"              {c['detail']}")
    return "\n".join(lines)


# -------------------------------------------------------------------- entry --

def analyse_match(agent, opponent, seed, steps=720, save=True):
    a = analyse(play_and_record(agent, opponent, seed, steps), seat=0)
    a.update({"agent": os.path.relpath(agent, ROOT) if os.path.isabs(agent) else agent,
              "opponent": os.path.relpath(opponent, ROOT) if os.path.isabs(opponent) else opponent,
              "seed": seed})
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        name = f"{os.path.basename(str(agent))[:-3]}_vs_{os.path.basename(str(opponent))[:-3]}_{seed}.json"
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(a, f, indent=1)
        a["report_path"] = os.path.join("data", "losses", name)
    return a


def stored_reports():
    """Every saved post-mortem, newest first. Used by the dashboard."""
    out = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.json")),
                    key=os.path.getmtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            d["file"] = os.path.basename(p)
            out.append(d)
        except (OSError, ValueError):
            continue
    return out


def _resolve(pattern, default=None):
    if not pattern:
        return default
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(p))
    return hits[-1] if hits else default


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="agents/agent_v*.py")
    ap.add_argument("--vs", default="agents/v2_tuned.py")
    ap.add_argument("--seed", type=int, default=7400)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--scan", type=int, default=0,
                    help="play N seeds and post-mortem every loss")
    ap.add_argument("--replay", help="analyse a stored replay instead of playing")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    pr.reset()
    if args.list:
        reports = stored_reports()
        print(json.dumps(reports, indent=1) if args.json else
              "\n".join(f"{r['file']:<56} {r.get('result', '?'):<5} "
                        f"{r.get('margin', 0):+,.0f}" for r in reports)
              or "no post-mortems yet")
        return 0

    if args.replay:
        with open(_resolve(args.replay), encoding="utf-8") as f:
            d = json.load(f)
        a = analyse(d.get("steps") or [], seat=args.seat)
        print(json.dumps(a, indent=1) if args.json else report_text(a))
        return 0

    agent = _resolve(args.agent)
    opp = _resolve(args.vs)
    if not agent or not opp:
        print("could not resolve agent or opponent", file=sys.stderr)
        return 2

    if args.scan:
        losses = []
        for i in range(args.scan):
            seed = args.seed + i * 101
            a = analyse_match(agent, opp, seed, args.steps)
            tag = "LOST" if a["result"] == "lost" else "won "
            pr.log(f"seed {seed}: {tag} {a['margin']:+,.0f}")
            if a["result"] == "lost":
                losses.append(a)
        pr.log(f"{len(losses)} loss(es) of {args.scan}")
        for a in losses:
            print("\n" + "=" * 70)
            print(f"seed {a['seed']}")
            print(report_text(a))
        return 0

    a = analyse_match(agent, opp, args.seed, args.steps)
    print(json.dumps(a, indent=1) if args.json else report_text(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
