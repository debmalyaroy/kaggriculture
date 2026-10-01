"""CROWN-2 Phase C: harvest sell-schedule overlays, sweep hybrids on Rust.

The portfolio-crown idea, made concrete: keep ONE farm plan (the crowned
base) and try the sell schedules of the field's best fresh routes on top of
it, via the same market_overrides mechanism arms use (SELL retiming only;
structural orders always the base's). The Rust engine makes the sweep free;
the official engine confirms only the winner (C4, separate step).

Donor selection is itself Rust-ranked: every fresh route plays the panel's
SOURCE ROUTES (the opponent seats of today's loss episodes, HOLDOUT episodes
excluded -- the reserve never selects). Feasibility is not estimated but
PLAYED: an overlay selling what the base farm never produces simply realises
fewer sells in the engine and scores accordingly.

    python src/experiments/overlay_lab.py --donors 6
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

BASE_ID = "92513718_s1"
HOLDOUT_EPS = {"92736483", "92675044", "92648008", "92795539"}
OUT = os.path.join(ROOT, "models", "lab", "overlay_sweep.json")
SEEDS = (60000, 150000)


def panel_ref_ids():
    """Route ids behind today's loss tapes (opponent seats), holdout excluded."""
    refs = []
    for p in sorted(glob.glob(os.path.join(
            ROOT, "data", "refresh", "2026-08-14", "loss_tapes", "*.py"))):
        m = re.match(r"tape_(\d+)_s(\d)\.py", os.path.basename(p))
        if m and m.group(1) not in HOLDOUT_EPS:
            refs.append(f"{m.group(1)}_s{m.group(2)}")
    return refs


def retime_overlay(base, donor, max_orders=10):
    """Transfer the donor's sell TIMING, never their basket.

    The naive transplant (donor's absolute schedule on our farm) measured
    0/32 with -51k/game: their basket assumes THEIR production. What is
    plausibly transferable is WHEN a strong farm monetizes -- so redistribute
    the base's own per-product sell totals across turns proportional to the
    donor's per-product timing curve. Structural orders stay the base's;
    totals are conserved exactly (remainders land on the base's original
    turns); the 10-order cap is respected.
    """
    def sells_by_turn(route):
        out = {}
        for t, turn in enumerate(route):
            for o in (turn.get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"):
                    try:
                        q = int(o[2])
                    except (TypeError, ValueError):
                        continue
                    if q > 0:
                        out.setdefault(o[1], []).append((t, q))
        return out

    base_sells = sells_by_turn(base)
    donor_sells = sells_by_turn(donor)
    new_sched = {}                       # {turn: {product: qty}}
    for item, base_list in base_sells.items():
        total = sum(q for _, q in base_list)
        d_list = donor_sells.get(item)
        if not d_list or total <= 0:
            for t, q in base_list:       # no donor signal: keep base timing
                new_sched.setdefault(t, {}).setdefault(item, 0)
                new_sched[t][item] += q
            continue
        d_total = sum(q for _, q in d_list)
        placed = 0
        for t, dq in d_list:
            q = int(total * dq / d_total)
            if q > 0:
                new_sched.setdefault(t, {}).setdefault(item, 0)
                new_sched[t][item] += q
                placed += q
        rem = total - placed             # conserve the basket exactly
        if rem > 0:
            t0 = base_list[-1][0]
            new_sched.setdefault(t0, {}).setdefault(item, 0)
            new_sched[t0][item] += rem

    hybrid = []
    for t, turn in enumerate(base):
        keep = [o for o in (turn.get("market") or [])
                if not (isinstance(o, list) and o and o[0] == "SELL")]
        sells = [["SELL", item, q] for item, q in
                 sorted(new_sched.get(t, {}).items()) if q > 0]
        if not sells and keep == (turn.get("market") or []):
            hybrid.append(turn)
            continue
        merged = dict(turn)
        room = max(0, max_orders - len(keep))
        merged["market"] = sells[:room] + keep
        hybrid.append(merged)
    return hybrid


def hybrid_actions(base, overlay):
    out = []
    for t, turn in enumerate(base):
        ov = overlay.get(str(t))
        if ov is None:
            out.append(turn)
        else:
            merged = dict(turn)
            merged["market"] = ov
            out.append(merged)
    return out


def sweep(cands, refs, loaded, log=print):
    """{name: (score_sum, games, margin)} open-loop on the Rust engine."""
    import kaggriculture.engine.rust_prerank as RP
    import kaggriculture.measure.win_metric as WM
    table = {}
    for name, acts in cands.items():
        pts = games = 0
        margin = 0.0
        for rid in refs:
            ref_acts = loaded.get(rid)
            if ref_acts is None:
                continue
            for seed in SEEDS:
                for seat in (0, 1):
                    a, b = (acts, ref_acts) if seat == 0 else (ref_acts, acts)
                    tape = RP.build_tape(a, b, seed,
                                         f"ovl_{name}_{rid}_{seed}_{seat}"[:110])
                    b0, b1 = RP.play_tape(tape)
                    us, them = (b0, b1) if seat == 0 else (b1, b0)
                    pts += WM.score(us, them)
                    games += 1
                    margin += us - them
                    os.remove(tape)
        table[name] = (pts, games, margin / max(1, games))
        log(f"  {name:<26} score {pts:.1f}/{games} "
            f"({100 * pts / max(1, games):.0f}%)  margin/g {margin / max(1, games):+,.0f}")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donors", type=int, default=6)
    ap.add_argument("--fresh", type=int, default=40,
                    help="fresh routes entering the donor pre-rank")
    args = ap.parse_args()

    import kaggriculture.data.routes as R
    import kaggriculture.engine.rust_prerank as RP
    import kaggriculture.agentbuild.v22_agent as v22_agent

    refs = panel_ref_ids()
    print(f"panel source routes (holdout excluded): {len(refs)}")

    idx = R.load_index()
    newest = max(r.get("date", "") for r in idx["routes"].values())
    # Same-day-ingested routes carry no archive "rank"; pick donors by the
    # TEAM'S CURRENT BOARD RATING instead -- the field's strongest farms are
    # exactly whose monetization we want to try. Winning seats only: a
    # schedule that lost its own game is a poor donor.
    from kaggriculture.measure.band_panel import board_ratings
    ratings = board_ratings()
    pool = [r for r in idx["routes"].values()
            if r.get("date") == newest and r.get("won")
            and str(r.get("episode", "")).isdigit()
            and ratings.get(r.get("team"))]
    # DIVERSIFY BEFORE the cut: the top-rated team dominates the day's
    # harvest, so a rating-sorted head is a one-team pool (measured: 40/40
    # from one team). Keep each team's single best (newest) route first.
    best_per_team = {}
    for r in sorted(pool, key=lambda r: r.get("episode", ""), reverse=True):
        best_per_team.setdefault(r["team"], r)
    fresh = sorted(best_per_team.values(),
                   key=lambda r: -ratings[r["team"]])[:args.fresh]
    print(f"donor pool: {len(fresh)} teams' best fresh WINNING routes "
          f"(newest day {newest}; top rating "
          f"{ratings[fresh[0]['team']]:.0f})" if fresh else "donor pool: 0")

    # C1 donor selection: Rust-rank the pool against the panel routes.
    order = RP.rank([r["id"] for r in fresh], refs, log=lambda *_: None)
    # Family dedupe: one donor per team.
    donors, seen_team = [], set()
    for rid in order:
        team = (idx["routes"].get(rid) or {}).get("team") or rid
        if team in seen_team:
            continue
        seen_team.add(team)
        donors.append(rid)
        if len(donors) >= args.donors:
            break
    print(f"donors (deduped by team): {donors}")

    base = R.load_route(BASE_ID)
    loaded = {rid: R.load_route(rid) for rid in refs + donors
              if _safe_load(R, rid)}

    # C2 harvest + C3 sweep. Plain base rides along as the control. Two
    # variants per donor: the raw transplant (kept for the record -- expected
    # bad) and the RETIMED overlay (donor timing, our basket).
    cands = {"BASE": base}
    harvest_meta = {}
    for rid in donors:
        if rid not in loaded:
            continue
        ov = v22_agent.market_overrides(base, loaded[rid])
        cands[f"raw:{rid}"] = hybrid_actions(base, ov)
        cands[f"retime:{rid}"] = retime_overlay(base, loaded[rid])
        harvest_meta[rid] = {"override_turns": len(ov)}
        print(f"  overlay {rid}: raw {len(ov)} turns; retimed variant built")

    print(f"\nC3 Rust hybrid sweep ({len(cands)} candidates x {len(refs)} "
          f"refs x {len(SEEDS)} seeds x 2 seats):")
    table = sweep(cands, refs, loaded)

    base_pts, base_games, base_m = table["BASE"]
    ranked = sorted(((v[0] / max(1, v[1]), k) for k, v in table.items()),
                    reverse=True)
    best_score, best = ranked[0]
    verdict = {
        "base_score": base_pts / max(1, base_games),
        "best": best, "best_score": best_score,
        "table": {k: {"score": v[0], "games": v[1], "margin_pg": v[2]}
                  for k, v in table.items()},
        "harvest": harvest_meta, "donors": donors, "refs": refs,
        "beats_base_openloop": bool(best != "BASE"
                                    and best_score > base_pts
                                    / max(1, base_games)),
    }
    json.dump(verdict, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nbest: {best} at {best_score:.3f} vs BASE "
          f"{verdict['base_score']:.3f} -> "
          f"{'PROCEED to C4 official test' if verdict['beats_base_openloop'] else 'PARK (no overlay beats the base open-loop)'}")
    print(f"-> {OUT}")
    return 0


def _safe_load(R, rid):
    try:
        R.load_route(rid)
        return True
    except Exception:                                           # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
