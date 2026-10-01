"""Paired reactive gate on the RUST SERVE substrate (exact-bank verified
2026-09-01; equivalence self-check is serve_match --compare-official).

    python src/serve_gate.py <candidate> <incumbent> [seeds...] [--workers N]

Prints per-candidate records, discordant-cell counts, and a final VERDICT
line (IMPROVEMENT / TIE / REGRESSION) that scripts key on.

REPAIRED 2026-09-03 (`docs/history/instrument-repair-2026-09-03.md`). This gate was
returning 96-0 for the candidate AND 96-0 for the incumbent: a test nothing
fails cannot discriminate, and it is what let v42.0 ship. Three changes, none
of which touch the CLI:

* MARGIN. Every record now carries the median paired margin per game and, for
  the pair, the count of cells where the margin moved but the winner did not.
  When the winner channel is saturated that is the only resolution left.
* HELD-OUT. The opponent roster is split into a SELECTION half and a HELD-OUT
  half; the VERDICT is decided on the held-out half, which no candidate search
  ever selected against. The overall record is still printed, labelled as NOT
  the decision.
* MIRROR DEDUPE. Both seats of one (opponent, seed) are the same world played
  from opposite sides, and with deterministic agents they reproduce to the
  dollar. Counting both doubled the sample and halved every p-value; the
  paired tests now run over distinct WORLDS.
"""
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = r"D:/codebase/kaggriculture"

OPPS = ["data/gauntlet/kaito_v48.py", "data/gauntlet/kaito_v43.py",
        "data/gauntlet/mutoy.py", "data/gauntlet/munib.py",
        "data/gauntlet/moon.py", "data/gauntlet/pub_rayk_c94.py",
        "data/gauntlet/pub_rayk_c95.py", "data/gauntlet/pub_v16rc5.py",
        "data/gauntlet/pub_adaptive_r1.py", "data/gauntlet/pub_kaito_v27.py",
        "data/gauntlet/teacher_v21r1.py", "data/gauntlet/v16rc5_harish.py"]

# Deterministic, recorded, stratified only by roster order: opponents at even
# positions select, odd positions score. Written down here rather than derived
# at run time so a selection sweep and the gate that judges it agree.
SPLIT = {o: ("selection" if i % 2 == 0 else "holdout")
         for i, o in enumerate(OPPS)}

_SRV = None


def play(job):
    cand, opp, seed, seat = job
    import sys as _s
    import os
    os.chdir(ROOT)
    global _SRV
    import kaggriculture.engine.serve_match as SM
    if _SRV is None:
        _SRV = SM.Serve()
    a = SM.load_agent(cand if seat == 0 else opp)
    b = SM.load_agent(opp if seat == 0 else cand)
    banks = SM.run_match(a, b, seed, srv=_SRV)
    return (banks[seat], banks[1 - seat])


def summarise(cells):
    """(w, l, score, median margin, mean margin) over {key: (own, opp)}."""
    vs = list(cells.values())
    w = sum(1 for v in vs if v[0] > v[1])
    l = sum(1 for v in vs if v[0] < v[1])
    mar = [v[0] - v[1] for v in vs]
    return {"n": len(vs), "w": w, "l": l,
            "score": (w + 0.5 * (len(vs) - w - l)) / len(vs) if vs else None,
            "median_margin": statistics.median(mar) if mar else None,
            "mean_margin": statistics.fmean(mar) if mar else None}


def dedupe_mirrored(keys, *maps):
    """Keep one key per (opponent, seed) world when both seats agree exactly."""
    keep, seen = [], {}
    for k in sorted(keys):
        base = k[:-1]
        sig = tuple((m[k][0], m[k][1]) for m in maps)
        if base in seen and seen[base] == sig:
            continue
        seen[base] = sig
        keep.append(k)
    return keep


def compare(a_cells, b_cells, keys):
    """Winner (McNemar) + margin (sign test) comparison over shared worlds."""
    import kaggriculture.measure.win_metric as WM
    sa = [WM.score(*a_cells[k]) for k in keys]
    sb = [WM.score(*b_cells[k]) for k in keys]
    t = WM.paired_test(sa, sb)
    dm = [(a_cells[k][0] - a_cells[k][1]) - (b_cells[k][0] - b_cells[k][1])
          for k in keys]
    pos = sum(1 for x in dm if x > 0)
    neg = sum(1 for x in dm if x < 0)
    from kaggriculture.measure.band_panel import sign_test
    t.update({"margin_moved_winner_same":
              sum(1 for x, y, z in zip(dm, sa, sb) if x != 0 and y == z),
              "identical": sum(1 for x in dm if x == 0),
              "median_margin_delta": statistics.median(dm) if dm else None,
              "margin_better_a": pos, "margin_better_b": neg,
              "margin_sign_p": sign_test(pos, neg)})
    return t


def line(label, s):
    return (f"{label:<30} {s['w']}-{s['l']}  score "
            f"{s['score']:.3f}  med margin {s['median_margin']:+,.0f}  "
            f"(n={s['n']})")


def main(argv):
    workers = 6
    args = []
    it = iter(argv)
    for a in it:
        if a == "--workers":
            workers = int(next(it))
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        else:
            args.append(a)
    cands = args[:2]
    if len(cands) < 2:
        raise SystemExit(__doc__.splitlines()[3].strip())
    seeds = [int(x) for x in (args[2:] or ["101", "102", "103", "104"])]
    jobs, meta = [], []
    for c in cands:
        for o in OPPS:
            for s in seeds:
                for seat in (0, 1):
                    jobs.append((c, o, s, seat))
                    meta.append((c, o, s, seat))
    cells = {c: {} for c in cands}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (c, o, s, seat), (own, opp) in zip(meta, ex.map(play, jobs,
                                                            chunksize=1)):
            cells[c][(o, s, seat)] = (own, opp)

    for c in cands:
        cs = cells[c]
        print(line(c.split('/')[-1], summarise(cs)))
        for sp in ("selection", "holdout"):
            sub = {k: v for k, v in cs.items() if SPLIT[k[0]] == sp}
            if sub:
                print("  " + line(sp.upper(), summarise(sub))
                      + ("   <- DECISION HALF" if sp == "holdout" else ""))

    a, b = cands
    shared = [k for k in cells[a] if k in cells[b]]
    # Legacy line, unchanged in shape: raw discordant CELL counts.
    disc_a = sum(1 for k in shared
                 if cells[a][k][0] > cells[a][k][1]
                 and not cells[b][k][0] > cells[b][k][1])
    disc_b = sum(1 for k in shared
                 if cells[b][k][0] > cells[b][k][1]
                 and not cells[a][k][0] > cells[a][k][1])
    print(f"discordant cells: {a.split('/')[-1]} wins {disc_a}, "
          f"{b.split('/')[-1]} wins {disc_b}")

    verdicts = {}
    for sp in ("all", "selection", "holdout"):
        ks = [k for k in shared if sp == "all" or SPLIT[k[0]] == sp]
        if not ks:
            continue
        raw = len(ks)
        ks = dedupe_mirrored(ks, cells[a], cells[b])
        t = compare(cells[a], cells[b], ks)
        if t["better_a"] > t["better_b"] + 2:
            v = "IMPROVEMENT"
        elif t["better_b"] > t["better_a"] + 2:
            v = "REGRESSION"
        else:
            v = "TIE"
        verdicts[sp] = v
        print(f"  {sp:<9} n={t['n_pairs']}/{raw} worlds  winners "
              f"{t['better_a']}b/{t['better_b']}w p={t['p_value']:.4f} | "
              f"margin {t['margin_better_a']}b/{t['margin_better_b']}w "
              f"p={t['margin_sign_p']:.4f} med "
              f"{t['median_margin_delta']:+,.0f} | moved-not-flipped "
              f"{t['margin_moved_winner_same']}, identical {t['identical']} "
              f"-> {v}")
    print(f"OVERALL record (selection+held-out, NOT the decision): "
          f"{verdicts.get('all', 'TIE')}")
    # The gate's answer is the HELD-OUT half. Selecting and scoring on the same
    # opponents is what made v44_single win its panel 77-11 and then lose 84 of
    # 86 discordant held-out cells.
    print(f"VERDICT: {verdicts.get('holdout', verdicts.get('all', 'TIE'))}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main(sys.argv[1:])
