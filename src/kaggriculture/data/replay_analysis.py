"""Stage 2: turn episode replays into a strategy profile.

A Kaggriculture replay is the whole match. Note the offset: `steps[i][p]["action"]`
is the action that *produced* state i, so the action taken from `steps[i]` is
what agent p returned on turn i, and `steps[i][0]["observation"]["farms"]` is
the public state of *both* farms. So a top player's replay tells us their
entire strategy -- portfolio, hiring curve, land timing, what they sold and
when -- without needing their code.

This module extracts a per-player profile from one replay, then aggregates
profiles across many replays into winner/loser medians that `refine.py` turns
into agent parameters.

Replays are ~27 MB each, so they are parsed one at a time and released.
"""
import json
import os
import statistics
from collections import Counter, defaultdict

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]


def _census(tiles):
    c = {k: 0 for k in CROPS + ANIMALS}
    c["COOP"] = c["PASTURE"] = c["WEED"] = c["EMPTY"] = 0
    for row in tiles:
        for t in row:
            if t is None:
                c["EMPTY"] += 1
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind == "PLANT":
                    c[t["crop"]] = c.get(t["crop"], 0) + 1
                elif "animal" in t:
                    c[t["animal"]] = c.get(t["animal"], 0) + 1
                elif kind in ("COOP", "PASTURE"):
                    c[kind] += 1
                elif kind == "WEED":
                    c["WEED"] += 1
    return c


def profile_replay(path):
    """Return {'meta': ..., 'players': [profile0, profile1]} for one replay."""
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)

    steps = rep.get("steps") or []
    if not steps:
        return None
    cfg = rep.get("configuration") or {}
    tpd = int(cfg.get("turnsPerDay", 24) or 24)
    info = rep.get("info") or {}
    rewards = rep.get("rewards") or [0, 0]
    n_players = 2

    players = []
    for p in range(n_players):
        players.append({
            "final_bank": float(rewards[p] or 0),
            "team": (info.get("TeamNames") or ["?", "?"])[p]
            if len(info.get("TeamNames") or []) > p else "?",
            "peak": {k: 0 for k in CROPS + ANIMALS},
            "tile_days": defaultdict(float),
            "hands_by_day": {},
            "land_days": [],
            "market_ops": Counter(),
            "sold_qty": Counter(),
            "bought_qty": Counter(),
            "unit_ops": Counter(),
            "money_by_day": {},
            "peak_herd": 0,
            "peak_crops": 0,
            "first_animal_day": None,
            "quadrants_by_day": {},
        })

    prev_quads = [1, 1]
    for i, step in enumerate(steps):
        obs0 = (step[0] or {}).get("observation") or {}
        day = int(obs0.get("day", i // tpd))
        hour = int(obs0.get("hour", i % tpd))
        farms = obs0.get("farms") or []

        for p in range(min(n_players, len(farms))):
            farm = farms[p] or {}
            prof = players[p]
            tiles = farm.get("tiles") or []
            if tiles:
                c = _census(tiles)
                herd = sum(c.get(a, 0) for a in ANIMALS)
                crops = sum(c.get(k, 0) for k in CROPS)
                prof["peak_herd"] = max(prof["peak_herd"], herd)
                prof["peak_crops"] = max(prof["peak_crops"], crops)
                if herd > 0 and prof["first_animal_day"] is None:
                    prof["first_animal_day"] = day
                for k in CROPS + ANIMALS:
                    prof["peak"][k] = max(prof["peak"][k], c.get(k, 0))
                    prof["tile_days"][k] += c.get(k, 0) / float(tpd)
            hires = int(farm.get("hires_today", 0) or 0)
            prof["hands_by_day"][day] = max(prof["hands_by_day"].get(day, 0), hires)
            quads = len(farm.get("unlocked_quadrants") or ["NW"])
            prof["quadrants_by_day"][day] = max(prof["quadrants_by_day"].get(day, 1), quads)
            if quads > prev_quads[p]:
                prof["land_days"].append(day)
                prev_quads[p] = quads
            if hour == 0:
                prof["money_by_day"][day] = float(farm.get("money", 0) or 0)

        for p in range(n_players):
            if p >= len(step) or not step[p]:
                continue
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            action = (nxt[p].get("action") if nxt and p < len(nxt) else None)
            if not isinstance(action, dict):
                continue
            prof = players[p]
            ops = [action.get("farmer")] + list(action.get("hands") or [])
            for op in ops:
                if isinstance(op, list) and op:
                    prof["unit_ops"][op[0]] += 1
            for order in (action.get("market") or []):
                if not isinstance(order, list) or not order:
                    continue
                prof["market_ops"][order[0]] += 1
                if order[0] == "SELL" and len(order) >= 3:
                    try:
                        prof["sold_qty"][order[1]] += int(order[2])
                    except (TypeError, ValueError):
                        pass
                elif order[0] in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT") and len(order) >= 3:
                    try:
                        prof["bought_qty"][order[1]] += int(order[2])
                    except (TypeError, ValueError):
                        pass

    for prof in players:
        prof["tile_days"] = dict(prof["tile_days"])
        prof["market_ops"] = dict(prof["market_ops"])
        prof["sold_qty"] = dict(prof["sold_qty"])
        prof["bought_qty"] = dict(prof["bought_qty"])
        prof["unit_ops"] = dict(prof["unit_ops"])
        prof["peak_hands"] = max(prof["hands_by_day"].values() or [0])
        prof["mean_hands_late"] = statistics.mean(
            [v for d, v in prof["hands_by_day"].items() if d >= 10] or [0])

    return {
        "file": os.path.basename(path),
        "episode_id": info.get("EpisodeId"),
        "seed": info.get("seed"),
        "configuration": cfg,
        "rewards": [float(r or 0) for r in rewards],
        "players": players,
    }


def _med(values, default=0.0):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else default


def aggregate(profiles, only_winners=True):
    """Median strategy profile across replays."""
    rows = []
    for rep in profiles:
        if not rep:
            continue
        for p, prof in enumerate(rep["players"]):
            won = rep["rewards"][p] >= max(rep["rewards"])
            if only_winners and not won:
                continue
            rows.append(prof)
    if not rows:
        return None

    agg = {
        "n": len(rows),
        "final_bank": _med([r["final_bank"] for r in rows]),
        "peak_herd": _med([r["peak_herd"] for r in rows]),
        "peak_crops": _med([r["peak_crops"] for r in rows]),
        "peak_hands": _med([r["peak_hands"] for r in rows]),
        "mean_hands_late": _med([r["mean_hands_late"] for r in rows]),
        "first_animal_day": _med([r["first_animal_day"] for r in rows
                                  if r["first_animal_day"] is not None], 99),
        "peak_tiles": {k: _med([r["peak"].get(k, 0) for r in rows])
                       for k in CROPS + ANIMALS},
        "tile_days": {k: _med([r["tile_days"].get(k, 0) for r in rows])
                      for k in CROPS + ANIMALS},
        "sold_qty": {k: _med([r["sold_qty"].get(k, 0) for r in rows])
                     for k in PRODUCTS},
        "unit_ops": {},
        "land_days": [],
    }
    op_keys = set()
    for r in rows:
        op_keys |= set(r["unit_ops"])
    total_ops = sum(sum(r["unit_ops"].values()) for r in rows) or 1
    for k in sorted(op_keys):
        agg["unit_ops"][k] = sum(r["unit_ops"].get(k, 0) for r in rows) / total_ops
    for slot in range(3):
        days = [r["land_days"][slot] for r in rows if len(r["land_days"]) > slot]
        agg["land_days"].append(_med(days, None) if days else None)
    return agg


def render_report(top_agg, mine_agg, ours_agg, out_path):
    """Human-readable comparison, written to `out_path`."""
    lines = ["# Replay analysis", ""]

    def block(title, a):
        # `lines` is only ever mutated in place here -- rebinding it with += would
        # make it a local and break the closure.
        if not a:
            lines.extend([f"## {title}", "", "_no data_", ""])
            return
        lines.extend([
            f"## {title}  (n={a['n']})", "",
            f"* median final bank **${a['final_bank']:,.0f}**",
            f"* peak herd {a['peak_herd']:.0f} · peak crop tiles {a['peak_crops']:.0f}",
            f"* peak hands/day {a['peak_hands']:.0f} · mean hands/day after day 10 "
            f"{a['mean_hands_late']:.1f}",
            f"* first animal on day {a['first_animal_day']:.0f}",
            f"* land bought on days {a['land_days']}",
            "",
            "| asset | peak tiles | tile-days | units sold |",
            "|---|---|---|---|",
        ])
        for k in CROPS + ANIMALS:
            sold = a["sold_qty"].get(k if k in PRODUCTS else
                                     {"GOOSE": "EGG", "COW": "MILK",
                                      "SHEEP": "WOOL"}.get(k, k), 0)
            lines.append(f"| {k} | {a['peak_tiles'][k]:.0f} | "
                         f"{a['tile_days'][k]:.0f} | {sold:.0f} |")
        lines.append("")
        ops = sorted(a["unit_ops"].items(), key=lambda kv: -kv[1])[:8]
        lines.append("* action mix: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in ops))
        lines.append("")

    block("Top-ladder winners", top_agg)
    block("Your previous submissions", mine_agg)
    block("Your current agent (local match)", ours_agg)

    if top_agg and ours_agg:
        lines.extend(["## Gap vs the top ladder", "",
                      "| metric | top | ours | delta |", "|---|---|---|---|"])
        for label, key in [("final bank", "final_bank"), ("peak herd", "peak_herd"),
                           ("peak crop tiles", "peak_crops"),
                           ("peak hands/day", "peak_hands"),
                           ("first animal day", "first_animal_day")]:
            t, o = top_agg[key], ours_agg[key]
            lines.append(f"| {label} | {t:,.0f} | {o:,.0f} | {o - t:+,.0f} |")
        lines.append("")
        lines.append("| asset | top peak tiles | our peak tiles | delta |")
        lines.append("|---|---|---|---|")
        for k in CROPS + ANIMALS:
            t = top_agg["peak_tiles"][k]
            o = ours_agg["peak_tiles"][k]
            lines.append(f"| {k} | {t:.0f} | {o:.0f} | {o - t:+.0f} |")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    profs = []
    for path in args.replays:
        print(f"parsing {os.path.basename(path)} ...", flush=True)
        profs.append(profile_replay(path))
    agg = aggregate(profs)
    print(json.dumps(agg, indent=1, default=str))
    if args.out:
        render_report(agg, None, None, args.out)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
