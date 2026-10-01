"""Track-P factory: search an OWNED schedule (the phase-2 refit).

The 2026-08-29 planner_v0 baseline read 0.000 (0-64) against the reactive
frontier panel -- per-turn judgment is dead as a Track-P route to a seat.
This factory is the replacement: (1+lambda) evolution over a frozen elite
production core's ECONOMY (labor + purchases + sells; `sell_search.py
--field-ops`) on the Rust engine, against tapes of the CURRENT elite --
picked by source-team ladder standing, never by recorded bank (the
anti-predictive measure). The holdout paired sign test is the only door to
an emitted agent, and that agent still faces the gauntlet, strict-future,
and the Track-P graduation gate like any candidate. Never submits.

    python src/trackp/factory.py                 # full search
    python src/trackp/factory.py --smoke         # wiring check
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The current elite whose programs beat or pace us, stable-first (the
# adaptive top-5 are uncopyable and unpanelable offline -- ladder-only).
ELITE_TEAMS = ("Kaileh57", "YJR", "senkin13", "Mingkang YAN", "tyz123456",
               "Erfan Eshratifar", "Gronk", "vkhydras")
OUT_DIR = os.path.join(ROOT, "models", "trackp", "factory")


def elite_panel_ids(idx, base_team, n=6):
    """Freshest winning route per elite team, engine-current, base excluded."""
    picks = []
    for team in ELITE_TEAMS:
        if team == base_team:
            continue
        rs = [r for r in idx["routes"].values()
              if r.get("team") == team and r.get("engine") == "1.32.7"
              and r.get("won")]
        if rs:
            picks.append(max(rs, key=lambda r: str(r.get("date")))["id"])
        if len(picks) >= n:
            break
    return picks


def hard_panel():
    """The 56 certified hard-band opponents + their recorded ladder seeds.

    The default panel is mined ELITE route tapes played on seeds 9000+. Both
    halves are the wrong target: tape worlds run 56-63k median shared bank
    against the ladder's ~85k, and the seeds are worlds we have never played.
    This panel is 56 REAL games against opponents rated 2300-2548, in the
    worlds they were actually played in, 56/58 of which reproduce the
    recorded banks to the dollar. Our live record against this field is
    19-37, so unlike the reactive panel it has headroom.
    """
    man = os.path.join(ROOT, ".local", "hardband", "factory_panel",
                       "manifest.json")
    if not os.path.exists(man):
        raise SystemExit("no hard panel -- run "
                         ".local/hardband/make_factory_panel.py")
    m = json.load(open(man, encoding="utf-8"))
    tapes = [o["tape"] for o in m["opponents"]]
    seeds = [o["seed"] for o in m["opponents"]]
    print(f"HARD PANEL: {len(tapes)} real 2300+ opponents, "
          f"{len(set(seeds))} recorded ladder seeds", flush=True)
    return tapes, seeds


def run_packages(args, idx, rec, panel):
    """A2 search: production-growth packages over the frozen base, with the
    same batch evaluation and holdout sign gate as every factory search."""
    import kaggriculture.pipeline.sell_search as SS
    from kaggriculture.trackp import guard as WM
    from kaggriculture.trackp import packages as PK
    from kaggriculture.trackp import routes_io as R
    import os as _os

    base_actions = R.load_route(args.base)
    rng = random.Random(20260830)
    _os.makedirs(SS.WORK, exist_ok=True)
    print("simulating the base once for the farm-state timeline...", flush=True)
    state = PK.day_state(base_actions)
    print(f"  empty-tile snapshots for {len(state['empties'])} days", flush=True)

    hard_seeds = None
    if getattr(args, "hard_panel", False):
        panel_tapes, hard_seeds = hard_panel()
    else:
        panel_tapes = [SS.write_tape(R.load_route(r["id"]),
                                     _os.path.join(SS.WORK, f"opp_{i}.tape"))
                       for i, r in enumerate(panel)]
    n_seeds = 2 if args.smoke else 24
    base_tape = SS.write_tape(base_actions,
                              _os.path.join(SS.WORK, "base.tape"))
    # With the hard panel each opponent belongs to ONE world, so seeds are
    # not a free dimension: the search half and the holdout half are disjoint
    # SLICES OF THE REAL SEEDS, never synthetic ranges.
    if hard_seeds:
        search_seeds = hard_seeds[0::2]
        holdout_seeds = hard_seeds[1::2]
    else:
        search_seeds = holdout_seeds = None
    def ev(cands, seeds):
        """Paired on the hard panel (each tape in its OWN world); the full
        panel x seed cross product otherwise."""
        if hard_seeds:
            pairs = [(t, sd) for t, sd in zip(panel_tapes, hard_seeds)
                     if sd in set(seeds)]
            return SS.batch_eval_paired(cands, pairs, 3)
        return SS.batch_eval(cands, panel_tapes, seeds, 3)

    base_eval = ev([base_tape],
                   search_seeds if hard_seeds
                   else list(range(9000, 9000 + n_seeds)))[0]
    print(f"base score: {base_eval['score']:.4f} ({base_eval['n']} games)",
          flush=True)

    best, best_score, stale = base_actions, base_eval["score"], 0
    best_margin = base_eval.get("margin", 0.0)
    best_claims = frozenset()
    lam = 4 if args.smoke else 12
    iters = 3 if args.smoke else args.iters
    for gen in range(iters):
        cands, claims = [], []
        attempts = 0
        while len(cands) < lam and attempts < lam * 6:
            attempts += 1
            c = PK.mutate_packages(best, state, rng, max_packages=3,
                                   claimed=best_claims)
            if c is not None:
                cands.append(c[0])
                claims.append(c[1])
        if not cands:
            print("  no placeable packages remain -- stopping", flush=True)
            break
        # Seeds ROTATE per generation and the incumbent is re-scored on the
        # same fresh set -- gains must generalize, not memorize (the fixed-
        # seed protocol overfit twice on 2026-08-29). Margin is the
        # SELECTION gradient (a +1.2k package flips no cell on a saturated
        # panel); score remains the only currency at the holdout gate.
        if hard_seeds:
            # rotate through the real seeds so a package cannot memorise one
            k = len(search_seeds)
            gseeds = [search_seeds[(gen * 7 + i) % k]
                      for i in range(min(k, n_seeds))]
        else:
            gseeds = list(range(9000 + gen * 31, 9000 + gen * 31 + n_seeds))
        tapes = [SS.write_tape(c, _os.path.join(SS.WORK, f"c{i}.tape"))
                 for i, c in enumerate(cands)]
        allev = ev(
            [SS.write_tape(best, _os.path.join(SS.WORK, "inc.tape"))] + tapes,
            gseeds)
        inc, evals = allev[0], allev[1:]
        best_score, best_margin = inc["score"], inc.get("margin", 0.0)
        gi = max(range(len(evals)),
                 key=lambda i: (evals[i]["score"], evals[i].get("margin", 0)))
        better = (evals[gi]["score"], evals[gi].get("margin", 0)) >                  (best_score, best_margin)
        if better:
            best, best_score, stale = cands[gi], evals[gi]["score"], 0
            best_margin = evals[gi].get("margin", 0.0)
            best_claims = claims[gi]
            print(f"  gen {gen:>4}: NEW BEST score {best_score:.4f} "
                  f"margin {best_margin:+,.0f}", flush=True)
        else:
            stale += 1
            if gen % 10 == 0:
                print(f"  gen {gen:>4}: best {best_score:.4f} (stale {stale})",
                      flush=True)
        if stale >= args.patience:
            print(f"  no gain in {args.patience} generations -- stopping",
                  flush=True)
            break

    hseeds = (holdout_seeds if hard_seeds
              else list(range(19000, 19000 + (4 if args.smoke else 60))))
    best_tape = SS.write_tape(best, _os.path.join(SS.WORK, "best.tape"))
    hb, hc = ev([base_tape, best_tape], hseeds)
    cells = sorted(set(hb["cells"]) & set(hc["cells"]))
    t = WM.paired_test([hc["cells"][k] for k in cells],
                       [hb["cells"][k] for k in cells])
    passed = bool(cells) and t["score_diff"] > 0 and t["significant"]
    print(f"HOLDOUT: candidate {hc['score']:.4f} vs base {hb['score']:.4f} "
          f"on {len(cells)} cells; diff {t['score_diff']:+.4f}, "
          f"p={t['p_value']:.4f}", flush=True)
    out_agent = _os.path.join(ROOT, ".local", "candidates",
                              "trackp_packages_cand.py")
    verdict = {"lane": "trackp-packages", "base": args.base,
               "search_score": best_score,
               "base_score": base_eval["score"],
               "holdout": {"candidate": hc["score"], "base": hb["score"],
                           "diff": t["score_diff"], "p": t["p_value"]},
               "passed": passed}
    if passed:
        SS.build_agent(best, rec, out_agent, "trackp_packages_cand.py")
        verdict["agent"] = out_agent
        print(f"PASS -- built {out_agent}; gauntlet + strict-future + "
              f"graduation still ahead", flush=True)
    else:
        print("HOLDOUT FAIL -- no agent built", flush=True)
    json.dump(verdict, open(_os.path.join(OUT_DIR, "packages_verdict.json"),
                            "w", encoding="utf-8"), indent=1)
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="mv_Kaileh57_2026-08-29")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--mode", choices=("orders", "packages"),
                    default="orders",
                    help="orders = sell_search single-order space (exhausted "
                         "2026-08-29); packages = A2 production growth")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hard-panel", action="store_true",
                    help="search against the HARD band -- 56 real 2300+ "
                         "ladder opponents in their own recorded seeds "
                         "(.local/hardband/factory_panel) instead of mined "
                         "elite route tapes on arbitrary seeds")
    args = ap.parse_args()

    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    rec = idx["routes"].get(args.base)
    if rec is None:
        raise SystemExit(f"base {args.base!r} not in the route index")
    panel = elite_panel_ids(idx, rec.get("team"))
    if len(panel) < 4:
        raise SystemExit(f"only {len(panel)} elite panel tapes -- refresh the "
                         f"index first")
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.mode == "packages":
        return run_packages(args, idx, rec,
                            [idx["routes"][i] for i in panel])
    out_agent = os.path.join(ROOT, ".local", "candidates",
                             "trackp_factory_cand.py")
    cmd = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "sell_search.py"),
           "--base", args.base, "--field-ops",
           "--panel-ids", *panel,
           "--iters", str(args.iters), "--patience", str(args.patience),
           "--out", out_agent]
    if args.smoke:
        cmd.append("--smoke")
    print(f"factory: base {args.base}, panel {panel}", flush=True)
    code = subprocess.run(cmd, cwd=ROOT).returncode
    # Mirror the verdict into Track-P's own model space for graduation.
    src_v = os.path.join(ROOT, "models", "factory", "sell_search.json")
    if os.path.exists(src_v):
        v = json.load(open(src_v, encoding="utf-8"))
        v["lane"] = "trackp-factory"
        json.dump(v, open(os.path.join(OUT_DIR, "search_verdict.json"), "w",
                          encoding="utf-8"), indent=1)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
