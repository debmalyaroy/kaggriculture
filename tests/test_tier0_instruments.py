"""Contract tests for the Tier-0 instruments (band_panel, leak_audit).

Both tools decide things: `band_panel` prints the PASS/FAIL bars the plan
gates ships on, and `leak_audit` returns a non-zero exit code that a build
chain can trip on. Every number they print comes out of a handful of pure
functions -- regime tagging, band assignment, leak extraction -- and those are
what this suite pins. NOTHING here plays an episode, so it is a `--fast` suite.

    python tests/test_tier0_instruments.py
"""
import os
import sys

import kaggriculture.measure.band_panel as BP  # noqa: E402
import kaggriculture.measure.leak_audit as LA  # noqa: E402


# ------------------------------------------------------------ regime tagging

def _rows(day_price):
    """[(day, {product: absolute price})] from {day: ratio} applied to all."""
    return [(d, {p: r * BP.BASE[p] for p in BP.STAT_PRODUCTS})
            for d, r in day_price.items()]


def test_regime_ratios_windows():
    rows = _rows({2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0, 9: 0.5, 10: 0.5,
                  11: 0.5, 12: 0.5})
    r35 = BP.regime_ratios(rows, BP.WINDOWS["d3-5"])
    r912 = BP.regime_ratios(rows, BP.WINDOWS["d9-12"])
    assert set(r35) == set(BP.STAT_PRODUCTS), r35
    assert all(abs(v - 1.0) < 1e-9 for v in r35.values()), r35
    assert all(abs(v - 0.5) < 1e-9 for v in r912.values()), r912
    # day 2 is outside every window and must not leak in
    assert BP.regime_ratios(rows, (20, 25)) == {}
    print("regime_ratios: window-selected, normalised by base price")


def test_cell_stat_needs_all_products():
    full = {p: 1.0 for p in BP.STAT_PRODUCTS}
    assert abs(BP.cell_stat(full) - 1.0) < 1e-9
    partial = dict(full)
    partial.pop(BP.STAT_PRODUCTS[-1])
    assert BP.cell_stat(partial) is None, \
        "a partial mean sits on a different scale than the calibration"
    assert BP.cell_stat({}) is None
    # the statistic is the MEAN of the four, not the min: a single cheap
    # product must not drag a rich world into LOW on its own
    mixed = dict(full)
    mixed[BP.STAT_PRODUCTS[0]] = 0.2
    assert BP.cell_stat(mixed) > 0.5
    print("cell_stat: mean over all four products, None when any is missing")


def test_regime_tag_and_tag_cell():
    lo = {p: 0.5 for p in BP.STAT_PRODUCTS}
    hi = {p: 1.5 for p in BP.STAT_PRODUCTS}
    assert BP.regime_tag(lo, 1.0) == "LOW"
    assert BP.regime_tag(hi, 1.0) == "HIGH"
    assert BP.regime_tag(hi, None) is None, "no threshold => no tag, not HIGH"
    assert BP.regime_tag({}, 1.0) is None
    # exactly at the threshold is HIGH -- the rule is strict `<`
    assert BP.regime_tag({p: 1.0 for p in BP.STAT_PRODUCTS}, 1.0) == "HIGH"
    result = {"ratios": {w: (lo if w == BP.PRIMARY else hi) for w in BP.WINDOWS}}
    tags = BP.tag_cell(result, {w: 1.0 for w in BP.WINDOWS})
    assert tags[BP.PRIMARY] == "LOW"
    assert all(v == "HIGH" for k, v in tags.items() if k != BP.PRIMARY), tags
    print("regime_tag: strict '<', per-window, None-safe")


def test_calibrate_threshold_is_tie_stable():
    xs = [i / 100.0 for i in range(100)]
    thr, got = BP.calibrate_threshold(xs, share=0.35)
    assert abs(got - 0.35) <= 0.01, (thr, got)
    assert 0.34 <= thr <= 0.36, thr
    # a fully degenerate statistic has no cut at all -- the d3-5 min(milk,
    # straw) failure mode; it must return None rather than a fake threshold
    assert BP.calibrate_threshold([1.0] * 50) == (None, None)
    # ties must not produce a cut that splits equal values
    ties = [1.0] * 60 + [2.0] * 40
    thr2, got2 = BP.calibrate_threshold(ties, share=0.35)
    assert got2 in (0.0, 0.6), got2
    assert thr2 is None or not any(abs(x - thr2) < 1e-12 for x in ties)
    print(f"calibrate_threshold: {thr:.3f} -> {got:.2f} share; degenerate "
          "input returns None")


def test_relative_tags_split_and_report_ties():
    stats = [float(i) for i in range(20)]
    tags, thr, got = BP.relative_tags(stats)
    assert tags.count("LOW") == 10 and tags.count("HIGH") == 10, tags
    assert thr == 9.5 and got == 0.5, (thr, got)
    # even when every cell is far below the ladder threshold (the panel-skew
    # case the absolute tag cannot resolve), the relative tag still splits
    skewed = [0.10 + 0.001 * i for i in range(20)]
    t2, _, g2 = BP.relative_tags(skewed)
    assert t2.count("LOW") == 10 and g2 == 0.5, (t2, g2)
    # heavily tied statistic (the real panel: 46 distinct values over 652
    # cells): no half exists, and the achieved share must say so instead of
    # a raw median silently dumping every tied cell on one side
    tied = [1.0] * 12 + [2.0] * 8
    t3, thr3, g3 = BP.relative_tags(tied)
    assert g3 == 0.6 and abs(g3 - 0.5) > 0.05, (thr3, g3)
    assert t3.count("LOW") == 12 and all(
        not (abs(x - thr3) < 1e-12) for x in tied), (t3, thr3)
    assert BP.relative_tags([1.0, 2.0])[0] == [None, None], "too few cells"
    assert BP.relative_tags([1.0] * 20)[1] is None, "no cut at all"
    print(f"relative_tags: 0.50 share on a spread stat, {g3:.2f} on a tied "
          "one, reported not hidden")


def test_auc_and_cohen_d():
    # perfect separation
    assert BP.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert BP.auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    # all ties -> chance
    assert BP.auc([1, 1, 1, 1], [False, True, False, True]) == 0.5
    assert BP.auc([1, 2], [True, True]) is None, "one class => no AUC"
    d = BP.cohen_d([0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
    assert d is not None and d > 5, d
    assert BP.cohen_d([1.0], [2.0]) is None
    assert BP.cohen_d([1.0, 1.0], [1.0, 1.0]) is None, "zero SD => None"
    print(f"auc/cohen_d: exact with ties; d={d:.1f} on a clean separation")


def test_discrimination_reports_the_split():
    # statistic and world bank move together: LOW cells are the poor worlds
    xs = [0.5 + 0.01 * i for i in range(100)]
    worlds = [40000.0 + 900.0 * i for i in range(100)]
    d = BP.discrimination(xs, worlds, thr=1.0)
    assert d["n"] == 100
    assert d["auc_sub90k"] > 0.9, d
    assert d["median_bank_low"] < d["median_bank_high"], d
    assert d["cohen_d"] > 1.0, d
    assert BP.discrimination(xs[:5], worlds[:5], 1.0) == {}, "undersized"
    assert BP.discrimination(xs, worlds, None) == {}, "no threshold"
    print(f"discrimination: AUC {d['auc_sub90k']}, d {d['cohen_d']}, "
          f"bank {d['median_bank_low']}/{d['median_bank_high']}")


def test_regime_from_stats_shape():
    reg = BP.regime_from_stats(
        {w: [0.5 + 0.01 * i for i in range(100)] for w in BP.WINDOWS},
        "unit test", worlds=[40000.0 + 900.0 * i for i in range(100)])
    assert reg["primary"] == BP.PRIMARY and reg["early"] == BP.EARLY
    assert set(reg["windows"]) == set(BP.WINDOWS)
    for w, d in reg["windows"].items():
        assert d["threshold"] is not None, w
        assert d["discrimination"], w
        assert d["quantiles"]["p10"] < d["quantiles"]["p90"], w
    # without worlds there is no discrimination block, and that is honest
    bare = BP.regime_from_stats({w: [0.1 * i for i in range(50)]
                                 for w in BP.WINDOWS}, "unit test")
    assert all(not d["discrimination"] for d in bare["windows"].values())
    print("regime_from_stats: every window carries threshold + discrimination")


# ----------------------------------------------------------- band assignment

def test_spread_is_even_and_capped():
    items = list(range(100))
    picked = BP.spread(items, 10)
    assert len(picked) == 10 and picked[0] == 0 and picked[-1] == 99
    assert picked == sorted(picked)
    assert BP.spread([1, 2, 3], 10) == [1, 2, 3], "never pads above supply"
    print(f"spread: {len(picked)} evenly spaced picks, endpoints kept")


def test_select_band_filters():
    lb = [("alpha", 2400.0, 10), ("beta", 1950.0, 700),
          ("gamma", 1800.0, 900), ("delta", 1750.0, 1000),
          ("Debmalya", 1919.0, 800), ("nogames", 1850.0, 950),
          ("norender", 1820.0, 960), ("toohigh", 2600.0, 3)]
    games = {"alpha": 50, "beta": 50, "gamma": 25, "delta": 5,
             "debmalya": 500, "nogames": 50, "norender": 50, "toohigh": 50}
    best = {k: ((999, 80000.0), f"ep_{k}_s0")
            for k in ("alpha", "beta", "gamma", "delta", "debmalya",
                      "nogames", "toohigh")}
    picked, n = BP.select_band(lb, games, best, 1700.0, 2000.0, min_games=20)
    names = [p["team"] for p in picked]
    assert "alpha" not in names and "toohigh" not in names, "rating window"
    assert "delta" not in names, "min_games=20 must drop a 5-game team"
    assert "norender" not in names, "no renderable 1.32.7 win => not a panel"
    assert "Debmalya" not in names, "never play ourselves"
    assert names == ["beta", "nogames", "gamma"], names   # rating-sorted
    assert n == 3, n
    assert picked[0]["rid"] == "ep_beta_s0" and picked[0]["lb_rank"] == 700
    print(f"select_band: {names} (window, min_games, renderable, self-exclude)")


def test_bars_are_the_plan_bars():
    assert BP.BARS["sub2000"] == 1.0, "any sub-2000 loss is a FAIL"
    assert BP.BARS["mid"] == 0.90 and BP.BARS["top100"] == 0.70
    assert set(BP.BAND_ORDER) == set(BP.BANDS)
    print("bars: sub2000 1.00 / mid 0.90 / top100 0.70")


# ---------------------------------------------------------- leak extraction

def _fixture_steps(seat=0, n_hands=3):
    """A synthetic 720-step replay with KNOWN leak metrics for `seat`.

    Shape matches what kaggle_environments hands back: steps[k][s] carries
    "observation" (public farms + that seat's private block) and "action".
    """
    steps = []
    for k in range(720):
        day = k // LA.TURNS_PER_DAY
        tiles = [[{"kind": "PLANT", "crop": "WHEAT" if i < 5 else None,
                   "consecutive_unwatered": 0} for i in range(5)],
                 [{"kind": "PASTURE",
                   "animal": "COW" if i < 4 else None,
                   "consecutive_unfed": 2 if (day == 10 and i < 3) else 0}
                  for i in range(5)]]
        farm = {"money": 1000.0 + k, "hands": ["h"] * n_hands, "tiles": tiles}
        other = {"money": 500.0, "hands": [], "tiles": []}
        pub = {"step": k, "farms": [farm, other] if seat == 0
               else [other, farm]}
        priv = {"shed": {"MILK": 12.0 if 8 <= day <= 12 else 0.0,
                         "WHEAT": 24.0 if k >= 700 else 0.0},
                "seeds": {"WHEAT": 20.0, "MELON": 0.0}}
        # farmer PASSes every turn; one hand works on even turns
        hands = [["PASS"] for _ in range(n_hands)]
        if k % 2 == 0:
            hands[0] = ["FEED", 1, 1]
        action = {"farmer": ["PASS"], "hands": hands, "market": []}
        if k == 100:
            action["market"] = [["BUY_PRODUCT", "WHEAT", 400],
                                ["BUY_PRODUCT", "FERTILIZER", 250],
                                ["BUY_ANIMAL", "COW", 9],
                                ["BUY_ANIMAL", "SHEEP", 4],
                                ["BUY_SEED", "WHEAT", 187]]
        st = [{"observation": dict(pub), "action": None},
              {"observation": dict(pub), "action": None}]
        st[seat]["observation"]["private"] = priv
        st[seat]["action"] = action
        st[1 - seat]["action"] = {"farmer": ["PASS"], "hands": [], "market": []}
        steps.append(st)
    return steps


def test_metrics_from_steps():
    m = LA.metrics_from_steps(_fixture_steps(seat=0), seat=0)
    assert m["feed_wheat"] == 400 and m["fert_bought"] == 250, m
    assert m["cows"] == 9 and m["sheep"] == 4, m
    assert m["seeds_bought"] == {"WHEAT": 187.0}, m["seeds_bought"]
    # step 0 carries no action; of the 719 acted steps hand 0 FEEDs on the
    # 359 even ones and everything else PASSes
    assert m["feed"] == 359, m["feed"]
    assert m["pass"] == 719 * (1 + 3) - 359, m["pass"]
    assert m["milk_held"] == 12.0, m["milk_held"]
    assert m["starving_d10"] == 3, m["starving_d10"]
    assert m["tiles_d28"] == 5, m["tiles_d28"]
    assert m["seed_held"] == 20.0, m["seed_held"]
    assert m["shed_700"] == 24.0 and m["shed_719"] == 24.0, m
    assert m["bank"] == 1000.0 + 719, m["bank"]
    assert m["idle_units"] > 0
    print(f"metrics_from_steps: wheat {m['feed_wheat']:.0f}, fert "
          f"{m['fert_bought']:.0f}, PASS {m['pass']}, starving d10 "
          f"{m['starving_d10']}, tiles d28 {m['tiles_d28']}")


def test_metrics_read_the_right_seat():
    """Seat 1's private block lives in steps[k][1] -- reading steps[k][0]
    would be the train/serve skew bug the trace capture already paid for."""
    m0 = LA.metrics_from_steps(_fixture_steps(seat=0), seat=0)
    m1 = LA.metrics_from_steps(_fixture_steps(seat=1), seat=1)
    for k in ("feed_wheat", "fert_bought", "milk_held", "starving_d10",
              "tiles_d28", "bank", "pass"):
        assert m0[k] == m1[k], (k, m0[k], m1[k])
    # and the OTHER seat in the same replay must not inherit our numbers
    other = LA.metrics_from_steps(_fixture_steps(seat=0), seat=1)
    assert other["feed_wheat"] == 0 and other["milk_held"] == 0, other
    print("metrics_from_steps: seat-correct, no cross-seat bleed")


def test_ops_normalises_missing_actions():
    assert LA._ops(None) == [["PASS"]]
    assert LA._ops({"farmer": None, "hands": [None, []]}) == [["PASS"]] * 3
    assert LA._ops({"farmer": ["DIG", 1, 1], "hands": [["WATER", 0, 0]]}) == \
        [["DIG", 1, 1], ["WATER", 0, 0]]
    print("_ops: absent farmer/hand entries count as PASS, never vanish")


def test_regression_direction_and_floor():
    # "high" metrics leak when they exceed the reference
    assert LA.regression(414, 183, "high") and not LA.regression(183, 183, "high")
    # "low" metrics leak when they fall short
    assert LA.regression(200, 326, "low") and not LA.regression(400, 326, "low")
    # a zero reference still flags a real leak, but not a rounding wobble
    assert LA.regression(2, 0, "high"), "2 starving animals is a leak"
    assert not LA.regression(1, 0, "high"), "the floor is 1 unit, not 0"
    # within tolerance in the costly direction is not a regression
    assert not LA.regression(220, 183, "high"), "25% tolerance"
    assert LA.regression(240, 183, "high")
    print("regression: direction-aware, 25% tolerance with a 1-unit floor")


def test_summarise_flags_the_median():
    good = {k: (ref if worse == "high" else ref * 2)
            for k, (ref, worse, _) in LA.REFERENCE.items()}
    bad = dict(good, feed_wheat=414, pass_=0)
    bad["pass"] = 926
    rows, flag = LA.summarise([{"metrics": good}] * 3)
    assert not flag, [r["metric"] for r in rows if r["regression"]]
    rows2, flag2 = LA.summarise([{"metrics": bad}] * 3)
    flagged = {r["metric"] for r in rows2 if r["regression"]}
    assert flag2 and flagged == {"feed_wheat", "pass"}, flagged
    # the verdict is on the MEDIAN, so one bad game out of three is not one
    rows3, flag3 = LA.summarise([{"metrics": good}, {"metrics": bad},
                                 {"metrics": good}])
    assert not flag3, [r["metric"] for r in rows3 if r["regression"]]
    print(f"summarise: median-based; flagged {sorted(flagged)} on a bad build")


def test_reference_matches_the_plan():
    """The top-10 medians are the audit's numbers; a silent edit here would
    move every future build's pass/fail bar."""
    expect = {"feed_wheat": 183, "fert_bought": 62, "pass": 513, "feed": 326,
              "care": 333, "collect": 338, "pastures": 17, "cows": 9,
              "sheep": 5, "seed_held": 3, "milk_held": 0, "starving_d10": 0,
              "tiles_d28": 27, "shed_700": 24}
    got = {k: v[0] for k, v in LA.REFERENCE.items()}
    assert got == expect, got
    assert LA.TOLERANCE == 0.25
    print("REFERENCE: 14 top-10 medians, unchanged from the 2026-09-03 audit")


if __name__ == "__main__":
    for fn in (test_regime_ratios_windows, test_cell_stat_needs_all_products,
               test_regime_tag_and_tag_cell, test_calibrate_threshold_is_tie_stable,
               test_relative_tags_split_and_report_ties, test_auc_and_cohen_d,
               test_discrimination_reports_the_split, test_regime_from_stats_shape,
               test_spread_is_even_and_capped, test_select_band_filters,
               test_bars_are_the_plan_bars, test_metrics_from_steps,
               test_metrics_read_the_right_seat, test_ops_normalises_missing_actions,
               test_regression_direction_and_floor, test_summarise_flags_the_median,
               test_reference_matches_the_plan):
        fn()
    print("\nall tier-0 instrument checks passed")
