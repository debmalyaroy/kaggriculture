"""Who else is running our route, and what did they change?

    python -m kaggriculture.data.forks --agent agents/v18_route.py
    python -m kaggriculture.data.forks --route 90521287_s0 --threshold 0.85
    python -m kaggriculture.data.forks --agent agents/v18_route.py --detail 3

Everything here reads Kaggle's **published episode archive** -- the same public
data the whole field mines, already on disk from `src/kaggriculture/data/routes.py`. There is no
telemetry in the agent and there could not be: a submission runs in a sandbox
with no network, and putting a callback in a published notebook would be a
serious breach of the competition and of the people forking it. The only honest
channel is the one Kaggle already gives everybody, which turns out to be enough.

What it finds, and why that is worth having:

* **Forks.** A route we shipped, replayed by someone else, shows up as a game
  whose action sequence matches ours on most turns. The turns where it does
  *not* match are precisely their edit -- their optimisation, in the open.
* **Siblings.** Anyone running the same donor route we mined, whether or not
  they came via us. Same diff, same value.
* **Outcomes.** Each match carries its recorded banks, so a change is not just
  visible, it is scored: this is what they altered, and this is what it earned.

The per-channel breakdown matters more than the headline number. A fork that
differs on 4 market turns and 0 field turns has done something small and
surgical to the sale schedule; one that differs on 200 field turns has replaced
the farm plan. Those deserve very different amounts of attention.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402


def route_of_agent(agent_path):
    """Pull the embedded route out of a built agent, so we compare what shipped."""
    import base64
    import zlib
    with open(agent_path, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'b85decode\("([^"]+)"\)', src)
    if not m:
        raise SystemExit(f"{agent_path} carries no embedded route")
    return json.loads(zlib.decompress(base64.b85decode(m.group(1))).decode("utf-8"))


def _turn_key(turn, channel):
    if not isinstance(turn, dict):
        return ()
    if channel == "field":
        ops = [turn.get("farmer")] + list(turn.get("hands") or [])
        return tuple(tuple(o) if isinstance(o, list) else () for o in ops)
    return tuple(tuple(o) if isinstance(o, list) else ()
                 for o in (turn.get("market") or []))


def compare(ours, theirs):
    """Per-channel agreement, and the turns where they diverge."""
    n = min(len(ours), len(theirs))
    field_same = market_same = 0
    field_diff, market_diff = [], []
    for t in range(n):
        if _turn_key(ours[t], "field") == _turn_key(theirs[t], "field"):
            field_same += 1
        else:
            field_diff.append(t)
        if _turn_key(ours[t], "market") == _turn_key(theirs[t], "market"):
            market_same += 1
        else:
            market_diff.append(t)
    return {
        "turns": n,
        "field_match": field_same / max(1, n),
        "market_match": market_same / max(1, n),
        "overall": (field_same + market_same) / max(1, 2 * n),
        "field_diff": field_diff,
        "market_diff": market_diff,
    }


def describe_market_edit(ours, theirs, turns, limit=12):
    """What actually changed in the market channel, in plain terms."""
    added, removed, resized, reordered = {}, {}, {}, 0
    for t in turns:
        a = [o for o in (ours[t].get("market") or []) if isinstance(o, list)]
        b = [o for o in (theirs[t].get("market") or []) if isinstance(o, list)]
        av = {}
        bv = {}
        for o in a:
            if len(o) >= 3:
                av[(o[0], o[1])] = av.get((o[0], o[1]), 0) + int(o[2] or 0)
        for o in b:
            if len(o) >= 3:
                bv[(o[0], o[1])] = bv.get((o[0], o[1]), 0) + int(o[2] or 0)
        if av == bv and len(a) == len(b):
            reordered += 1          # same volumes, different slot order
            continue
        for key in set(av) | set(bv):
            d = bv.get(key, 0) - av.get(key, 0)
            if d == 0:
                continue
            label = f"{key[0]} {key[1]}"
            if key not in av:
                added[label] = added.get(label, 0) + d
            elif key not in bv:
                removed[label] = removed.get(label, 0) - av[key]
            else:
                resized[label] = resized.get(label, 0) + d
    return {"added": added, "removed": removed, "resized": resized,
            "reordered_turns": reordered}


def scan(ours, threshold=0.80, exclude_id=None):
    idx = R.assign_windows(R.load_index(), verbose=False)
    R.save_index(idx)
    hits = []
    for rid, rec in idx["routes"].items():
        if rid == exclude_id:
            continue
        try:
            theirs = R.load_route(rid)
        except OSError:
            continue
        cmp = compare(ours, theirs)
        if cmp["overall"] >= threshold:
            hits.append({"id": rid, "rec": rec, "cmp": cmp})
    hits.sort(key=lambda h: -h["cmp"]["overall"])
    return hits, len(idx["routes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None, help="a built route agent")
    ap.add_argument("--route", default=None, help="or a mined route id")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="minimum overall action agreement to report")
    ap.add_argument("--detail", type=int, default=3,
                    help="show the market edit for the top N matches")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.route:
        ours = R.load_route(args.route)
        label, exclude = args.route, args.route
    else:
        agent = args.agent or os.path.join(ROOT, "agents", "v18_route.py")
        agent = agent if os.path.isabs(agent) else os.path.join(ROOT, agent)
        ours = route_of_agent(agent)
        label, exclude = os.path.basename(agent), None
        m = re.search(r"Route: (\S+)", open(agent, encoding="utf-8").read())
        if m:
            exclude = m.group(1)

    hits, total = scan(ours, args.threshold, exclude)
    print(f"{label}: {len(ours)} turns, compared against {total} mined route(s) "
          f"at >= {100 * args.threshold:.0f}% agreement\n")
    if not hits:
        print("no close matches. Either nobody in the mined set is running this "
              "route, or we have not mined their games yet -- coverage is the "
              "limit here, not the method.")
        return 0

    print(f"{'route':<16}{'rank':>5} {'team':<24}{'overall':>9}{'field':>8}"
          f"{'market':>8}{'bank':>11}{'vs':>11}")
    print("-" * 92)
    for h in hits:
        c, rec = h["cmp"], h["rec"]
        print(f"{h['id']:<16}{rec.get('rank') or '?':>5} {rec.get('team', '?')[:23]:<24}"
              f"{100 * c['overall']:>8.1f}%{100 * c['field_match']:>7.1f}%"
              f"{100 * c['market_match']:>7.1f}%{rec.get('bank', 0):>11,.0f}"
              f"{rec.get('opp_bank', 0):>11,.0f}")

    for h in hits[:max(0, args.detail)]:
        c, rec = h["cmp"], h["rec"]
        print(f"\n--- {h['id']}  {rec.get('team', '?')} "
              f"(rank {rec.get('rank', '?')}) ---")
        print(f"  differs on {len(c['field_diff'])} field turn(s) and "
              f"{len(c['market_diff'])} market turn(s) of {c['turns']}")
        if not c["market_diff"]:
            print("  market channel identical -- their edit is entirely in the field")
            continue
        theirs = R.load_route(h["id"])
        edit = describe_market_edit(ours, theirs, c["market_diff"])
        if edit["reordered_turns"]:
            print(f"  {edit['reordered_turns']} turn(s) are pure re-ordering "
                  f"(same volumes, different slots)")
        for name, d in (("added", edit["added"]), ("removed", edit["removed"]),
                        ("resized", edit["resized"])):
            if d:
                top = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:6]
                print(f"  {name:<9} " + ", ".join(f"{k} {v:+,}" for k, v in top))
        days = sorted({t // 24 for t in c["market_diff"]})
        print(f"  market edits fall on day(s): "
              f"{', '.join(str(d) for d in days[:14])}"
              + (" ..." if len(days) > 14 else ""))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{"id": h["id"], "team": h["rec"].get("team"),
                        "rank": h["rec"].get("rank"),
                        "bank": h["rec"].get("bank"),
                        "opp_bank": h["rec"].get("opp_bank"),
                        **{k: v for k, v in h["cmp"].items()
                           if k not in ("field_diff", "market_diff")}}
                       for h in hits], fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
