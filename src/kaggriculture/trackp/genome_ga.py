"""Genome GA over the owned economy (Track P, T-2).

(1+lambda) hill-climb in plan-space: mutate the incumbent genome, compile
each child with the co-simulated compiler (economy.compile_cosim -- plans
every day against the engine's real state), and score it ADVERSARIALLY:
kagg-batch episodes vs the current elite tape panel, both seats, rotating
seeds, fitness = (score, margin) lexicographic -- the factory protocol.

Mirror bank was stage 1's fitness and hit 63k, but the mirror market lies:
elites flood the shared inventory (kaileh banks 115-134k adversarially)
and town-shop scarcity rewards different products than a mirror does.
Rust batch here is the PRE-RANKER role; any ship-decision still goes
through python-engine panels + the holdout sign gate downstream.

Discipline carried over from every burned lesson:
  * rotating seeds each generation, incumbent re-scored on the SAME seeds
    (fixed-seed search overfit twice: +19pp selection -> +1.4pp holdout);
  * the centre only moves on a strict paired improvement;
  * checkpointed every generation, resumable after a crash.

    python src/trackp/genome_ga.py --gens 400 --pop 8 --workers 6
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

STATE_DIR = os.path.join(ROOT, "models", "trackp", "genome_ga")
TAPE_DIR = os.path.join(ROOT, ".local", "trackp", "ga_tapes")


def bootstrap_genome():
    """Stage-1 (mirror) GA winner, gen 11, 63,370 mirror bank: dense NW
    engine (wheat+geese+cows: eggs, milk, fertilizer and feed at home from
    day 0), NE day 3 for fertilized melon/strawberry, SW/SE never bought,
    CARE on."""
    from kaggriculture.trackp import economy as E
    st_path = os.path.join(STATE_DIR, "state.json")
    if os.path.exists(st_path):
        st = json.load(open(st_path, encoding="utf-8"))
        if st.get("best_ever"):
            return st["best_ever"]["genome"]
    return E.default_genome()


def _compile(job):
    genome, seed = job
    from kaggriculture.trackp import economy as E
    try:
        tape, bank = E.compile_cosim(genome, seed=seed, return_bank=True)
        return tape, bank
    except Exception:
        return None, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--pop", type=int, default=8, help="children per gen")
    ap.add_argument("--seeds", type=int, default=2,
                    help="eval seeds per gen (x panel x 2 seats)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=10, help="kagg threads")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--hard-panel", action="store_true",
                    help="tune against the HARD band: 56 real 2300+ ladder "
                         "opponents in their OWN recorded seeds, paired. The "
                         "elite-tape panel plays worlds we never see.")
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    import kaggriculture.pipeline.sell_search as SS
    from kaggriculture.trackp import routes_io as R
    from kaggriculture.trackp import economy as E
    from kaggriculture.trackp import factory as F

    os.makedirs(TAPE_DIR, exist_ok=True)
    os.makedirs(SS.WORK, exist_ok=True)
    idx = R.load_index()
    hard = None
    if args.hard_panel:
        # THE GENOME IS THE RUNTIME POLICY. Tuning it against mined elite
        # tapes on arbitrary seeds optimises for worlds we never play: those
        # tape worlds run 56-63k median shared bank against a ladder at ~85k.
        # This panel is 56 REAL games against opponents rated 2302-2548, each
        # kept in the world it was actually played in, 56/58 of which
        # reproduce the recorded banks to the dollar.
        _m = json.load(open(os.path.join(
            ROOT, ".local", "hardband", "factory_panel", "manifest.json"),
            encoding="utf-8"))
        hard = [(o["tape"], o["seed"]) for o in _m["opponents"]]
        # search on one half, keep the other untouched for the holdout gate
        hard_search = hard[0::2]
        panel_tapes = [t for t, _s in hard_search]
        print(f"HARD PANEL: {len(hard_search)} real 2300+ opponents "
              f"(holdout half withheld)", flush=True)
    panel_ids = [] if hard else F.elite_panel_ids(idx, None)
    if not hard and len(panel_ids) < 4:
        raise SystemExit(f"only {len(panel_ids)} elite panel tapes")
    # The CURRENT 2400-band meta: teams whose routes beat v36 in its drift
    # window (2026-08-31 loss forensics). Training only on the old top-10
    # leaves the live mid-elite unmodeled.
    for team in ("ShadowT_T", "rui xiong", "A Poor Vul"):
        rs = [r for r in idx["routes"].values()
              if r.get("team") == team and r.get("won")
              and r.get("engine") == "1.32.7"]
        if rs:
            pid = max(rs, key=lambda r: str(r.get("date")))["id"]
            if pid not in panel_ids:
                panel_ids.append(pid)
    panel_ids = panel_ids[:8]
    if not hard:
        panel_tapes = [SS.write_tape(R.load_route(pid),
                                 os.path.join(TAPE_DIR, f"opp_{i}.tape"))
                       for i, pid in enumerate(panel_ids)]
        print(f"elite panel: {panel_ids}", flush=True)

    state_path = os.path.join(STATE_DIR, "state_adv.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
        center, gen0, best_ever = st["center"], st["gen"], st.get("best_ever")
        print(f"resuming at gen {gen0}", flush=True)
    else:
        center, gen0, best_ever = bootstrap_genome(), 0, None

    rng = random.Random(777)
    t_start = time.time()
    for gen in range(gen0, args.gens):
        if args.hours and (time.time() - t_start) > args.hours * 3600:
            print("hour budget reached; checkpoint stands", flush=True)
            break
        t0 = time.time()
        genomes = [center]
        for _ in range(args.pop):
            child = E.mutate_genome(center, rng)
            if rng.random() < 0.35:
                child = E.mutate_genome(child, rng)
            genomes.append(child)
        cseed = 9000 + gen                       # compile-world seed rotates
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            compiled = list(ex.map(_compile, [(g, cseed) for g in genomes],
                                   chunksize=1))
        cand_tapes, keep = [], []
        for i, (tape, _bank) in enumerate(compiled):
            if tape is None:
                continue
            path = os.path.join(TAPE_DIR, f"cand_{i}.tape")
            SS.write_tape(tape, path)
            cand_tapes.append(path)
            keep.append(i)
        if hard:
            # rotate through the REAL seeds; each opponent stays in its own
            # world, so seeds are not a free dimension.
            k = len(hard_search)
            pick = [hard_search[(gen * 5 + i) % k]
                    for i in range(min(k, max(4, args.seeds * 4)))]
            res = SS.batch_eval_paired(cand_tapes, pick, args.threads)
        else:
            eseeds = [23000 + gen * 11 + k for k in range(args.seeds)]
            res = SS.batch_eval(cand_tapes, panel_tapes, eseeds, args.threads)
        by_idx = {keep[j]: r for j, r in enumerate(res)}
        if 0 not in by_idx:
            print(f"gen {gen}: incumbent failed to compile?!", flush=True)
            continue
        inc = by_idx[0]

        def paired(r):
            """(score delta, margin delta, positive fraction) vs the
            incumbent on IDENTICAL (opp, seed, seat) cells -- opponent and
            seed variance (+/-10k) cancels; candidate effects (+/-3k)
            survive. The unpaired accept was a random walk (mirror bank
            drifted 63k -> 31k while 'improving' margin)."""
            mc, mi = r.get("mcells", {}), inc.get("mcells", {})
            keys = [k for k in mc if k in mi]
            if not keys:
                return (-1.0, 0.0, 0.0)
            def sc(m):
                return 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            ds = sum(sc(mc[k]) - sc(mi[k]) for k in keys) / len(keys)
            dm = sum(mc[k] - mi[k] for k in keys) / len(keys)
            pos = sum(1 for k in keys if mc[k] > mi[k]) / len(keys)
            return (ds, dm, pos)

        fit = {i: paired(r) for i, r in by_idx.items()}
        best_i = max(fit, key=lambda i: (fit[i][0], fit[i][1]))
        ds, dm, pos = fit[best_i]
        moved = ""
        better = ds > 0 or (ds == 0 and dm > 500 and pos >= 0.55)
        if best_i != 0 and better:
            # TWO-STAGE ACCEPT: gen-seed paired wins overfit those very
            # cells (holdout after 60 gens of '*': +1,253/cell, p0.53 --
            # nothing transferred). The would-be winner must confirm its
            # paired gain on FRESH seeds before taking the centre.
            cseeds = [61000 + gen * 13 + k for k in range(args.seeds)]
            conf = SS.batch_eval(
                [cand_tapes[keep.index(0)], cand_tapes[keep.index(best_i)]],
                panel_tapes, cseeds, args.threads)
            m0 = conf[0].get("mcells", {})
            m1 = conf[1].get("mcells", {})
            ks = [k for k in m1 if k in m0]
            cdm = sum(m1[k] - m0[k] for k in ks) / max(1, len(ks))
            cpos = sum(1 for k in ks if m1[k] > m0[k]) / max(1, len(ks))
            # Winner's-curse control (2026-08-31 audit: naive accept-sum
            # ~+80k, real cumulative +8-16k): barely-passing confirms are
            # half noise, so demand a solid fresh-seed effect. SCORE is
            # the currency though: a fresh-seed score gain accepts with
            # margin merely non-collapsing (gen 1197: a p1.00 score gain
            # was wrongly refused for missing the dollar floor).
            def _sc(m):
                return 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            cds = sum(_sc(m1[k]) - _sc(m0[k]) for k in ks) / max(1, len(ks))
            # The collapse guard scales with the score gain: three
            # borderline vetoes (gens 1388/1712/1714) blocked +0.03-0.05
            # fresh score for ~-1.7k margin -- the currency-correct trade.
            guard = -4000 if cds >= 0.03 else -1500
            if (cds > 0 and cdm > guard) or (cdm > 500 and cpos >= 0.60):
                center = genomes[best_i]
                moved = f" * (conf {cds:+.3f}/{cdm:+,.0f}/p{cpos:.2f})"
            else:
                moved = f" x (conf {cds:+.3f}/{cdm:+,.0f}/p{cpos:.2f})"
        row = {"gen": gen, "score": by_idx[best_i]["score"],
               "margin": by_idx[best_i].get("margin", 0.0),
               "bank": compiled[best_i][1],
               "genome": copy.deepcopy(genomes[best_i])}
        if best_ever is None or (row["score"], row["margin"]) > \
                (best_ever["score"], best_ever["margin"]):
            best_ever = row
        json.dump({"center": center, "gen": gen + 1, "best_ever": best_ever},
                  open(state_path, "w", encoding="utf-8"))
        print(f"gen {gen:>3}: inc {inc['score']:.3f}/"
              f"{inc.get('margin', 0.0):>8,.0f}  "
              f"pdelta {ds:+.3f}/{dm:>+7,.0f}/p{pos:.2f}{moved}  "
              f"mirror {compiled[best_i][1]:>8,.0f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    json.dump(best_ever, open(os.path.join(STATE_DIR, "best_adv.json"), "w",
                              encoding="utf-8"), indent=1)
    print(f"done: best score {best_ever['score']:.3f} "
          f"margin {best_ever['margin']:,.0f}", flush=True)


if __name__ == "__main__":
    main()
