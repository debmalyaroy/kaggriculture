"""The recalibrated crown gate: paired McNemar PER BAND + a ladder-weighted
aggregate, in the currency the ladder pays (WINS, never margin).

Workstream C (crown de-saturation, 2026-09-18). The frozen gate --
`refresh_cycle.CROWN_GATE = 10` percentage points of win rate over the
incumbent on a loss-tape roster -- is SATURATED: once incumbent and challenger
both sweep the referees to 100%, no challenger can ever clear a +10pp win-rate
bar, so the gate is pinned at HOLD independent of quality (memory:
crown-gate-saturated). The referees were also built from ONE agent's losses, so
they measure "is that bug absent", not "is this route stronger".

The fix is a BANDED referee panel (`crown_panel.py`) plus this gate, which
replaces the single win-rate threshold with a PAIRED test per rating band:

  * every band is a McNemar exact paired test (`win_metric.paired_test`) of the
    challenger vs the incumbent over the SAME cells (opponent, seed, seat), so
    saturation in one band cannot hide a regression in another;
  * the bands are combined into ONE ladder-weighted aggregate score
    difference, weighting higher bands more because a win against a stronger
    field buys more rating than a win against a weaker one;
  * SHIP RULE: NO band significantly regressed (McNemar p < 0.05 with a
    negative score difference) AND the ladder-weighted aggregate improvement is
    positive. "Non-regression in every band" is reported both strictly (every
    band's diff >= 0) and sign-tested (no band a SIGNIFICANT regression); the
    ship flag uses the sign-tested reading, because with a handful of cells per
    band a tiny non-significant negative diff is noise, not a regression, and
    this repo gates on evidence sign, never on a dollar or point threshold.

Everything here is PURE (no engine, no files, no network): it takes the paired
per-cell SCORES already measured and returns a verdict, so `tests/
test_crown_gate.py` checks the arithmetic in milliseconds instead of a
20-minute tournament. The caller (`refresh_cycle.stage_crown_panel`) produces
the cells on the serve substrate.
"""
from __future__ import annotations

import kaggriculture.measure.win_metric as WM

# Ladder weight per band: the rating centroid, normalised in the aggregate.
# A win in a higher band is worth more rating, so it carries more weight in the
# single number the ship rule reads. Bands with no cells drop out entirely
# (they contribute neither weight nor evidence), so a panel that is thin at the
# top degrades to weighting the bands it actually measured -- it never invents
# evidence for a band it could not play.
BAND_WEIGHT = {
    "<2100": 2000.0,
    "2100-2300": 2200.0,
    "2300-2500": 2400.0,
    "2500-2700": 2600.0,
    "2700+": 2800.0,
}

ALPHA = 0.05


def band_verdict(best_scores, inc_scores):
    """Paired McNemar of the challenger vs the incumbent for ONE band.

    `best_scores`/`inc_scores` are positionally aligned per-cell win scores
    (1.0 win / 0.5 draw / 0.0 loss), same (opponent, seed, seat) order for
    both. Returns the `win_metric.paired_test` dict plus the two flags the
    ship rule reads.
    """
    t = WM.paired_test(list(best_scores), list(inc_scores))
    diff = t["score_diff"]
    sig = t["significant"]
    t["regressed"] = bool(diff < 0 and sig)          # a SIGNIFICANT loss
    t["improved"] = bool(diff > 0 and sig)
    t["non_regression"] = bool(diff >= 0)            # strict, weak sense
    return t


def aggregate(per_band_verdicts, weights=BAND_WEIGHT):
    """Ladder-weighted mean paired score difference over the bands present.

    Weights are per-band ladder centroids; a band absent from
    `per_band_verdicts` (no cells) contributes nothing. Returns
    (aggregate_diff, total_weight, {band: weight}).
    """
    num = den = 0.0
    used = {}
    for band, v in per_band_verdicts.items():
        if not v or not v.get("n_pairs"):
            continue
        w = float(weights.get(band, 2000.0))
        used[band] = w
        num += w * v["score_diff"]
        den += w
    return (num / den if den else 0.0), den, used


def crown_gate_banded(per_band_cells, weights=BAND_WEIGHT, alpha=ALPHA):
    """Decide the crown from paired per-band cells.

    `per_band_cells` = {band: (best_scores, inc_scores)} -- positionally
    aligned per-cell win scores from `stage_crown_panel`. Returns a verdict
    dict:

      ship            the recommendation (bool)
      reason          one-line human string
      aggregate       ladder-weighted mean paired score diff
      bands           {band: paired_test dict + flags}
      regressed_bands [band, ...] that SIGNIFICANTLY regressed (p < alpha)
      strict_non_regression  every band's diff >= 0 (reported, not required)
    """
    bands = {}
    for band, cells in per_band_cells.items():
        best, inc = cells
        if not best or not inc:
            continue
        bands[band] = band_verdict(best, inc)
    if not bands:
        return {"ship": False, "reason": "no banded cells measured",
                "aggregate": 0.0, "bands": {}, "regressed_bands": [],
                "strict_non_regression": False, "weights": {}}

    regressed = sorted(b for b, v in bands.items() if v["regressed"])
    strict_nr = all(v["non_regression"] for v in bands.values())
    agg, tot_w, used_w = aggregate(bands, weights)
    improved = sorted(b for b, v in bands.items() if v["improved"])

    ship = (not regressed) and agg > 0
    if regressed:
        reason = (f"HOLD: {len(regressed)} band(s) significantly regressed "
                  f"(p<{alpha}): {regressed}")
    elif agg <= 0:
        reason = (f"HOLD: no band regressed but the ladder-weighted aggregate "
                  f"is {agg:+.3f} (not an upgrade)")
    else:
        reason = (f"SHIP: no band regressed, ladder-weighted aggregate "
                  f"{agg:+.3f}"
                  + (f", improved {improved}" if improved else ""))
    return {"ship": bool(ship), "reason": reason, "aggregate": agg,
            "bands": bands, "regressed_bands": regressed,
            "improved_bands": improved,
            "strict_non_regression": bool(strict_nr),
            "weights": used_w}


def holdout_alarm(selection_ship, holdout_per_band, weights=BAND_WEIGHT,
                  alpha=ALPHA):
    """Overfit alarm on the RESERVED (holdout) referees.

    The selection referees chose the crown; the holdout referees never touched
    selection, so if selection says SHIP while a held-out band shows a
    significant regression, the panel is being fit rather than beaten. This is
    REPORT-ONLY (like `stage_holdout`): it returns the divergence, the crown
    decision stands, and the divergence is the alarm.

    Returns {diverges, reason, aggregate, regressed_bands, bands}.
    """
    v = crown_gate_banded(holdout_per_band, weights, alpha)
    diverges = bool(selection_ship and (v["regressed_bands"]
                                        or v["aggregate"] <= 0))
    if not v["bands"]:
        reason = "no held-out referees to check"
        diverges = False
    elif diverges and v["regressed_bands"]:
        reason = (f"held-out panel DIVERGES: selection shipped but "
                  f"{v['regressed_bands']} regressed on reserved referees")
    elif diverges:
        reason = (f"held-out panel DIVERGES: selection shipped but held-out "
                  f"aggregate is {v['aggregate']:+.3f}")
    else:
        reason = (f"held-out agrees (aggregate {v['aggregate']:+.3f}, "
                  f"no reserved-band regression)")
    return {"diverges": diverges, "reason": reason,
            "aggregate": v["aggregate"],
            "regressed_bands": v["regressed_bands"], "bands": v["bands"]}
