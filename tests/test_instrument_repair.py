"""Contract tests for the 2026-09-03 instrument repair (band_panel/serve_gate).

The panels anti-predicted (`v44_single` won its selection panel 77-11 and then
lost 84 of 86 discordant HELD-OUT cells) and then went blind (the whole v22
chassis moved 4 of 652 cells). The repair added three things, and all three are
pure arithmetic that must never quietly change:

  * MARGIN alongside cells, so a saturated win column still resolves.
  * A recorded SELECTION / HELD-OUT split, so scoring cannot read back its own
    selection.
  * A world-distribution readout against the ladder, so a panel that plays a
    different game says so.

Plus the seat-mirror dedupe: both seats of one world reproduce to the dollar
with deterministic agents, so counting both doubled every sample and halved
every p-value. NOTHING here plays an episode -- this is a `--fast` suite.

    python tests/test_instrument_repair.py
"""
from kaggriculture.paths import ROOT
import os
import sys

import kaggriculture.measure.band_panel as BP  # noqa: E402


def _cells(pairs, **kw):
    return [dict({"own": a, "opp": b}, **kw) for a, b in pairs]


# --------------------------------------------------------------- margin block

def test_summarise_cells_counts_and_margins():
    cs = _cells([(100.0, 50.0), (10.0, 90.0), (70.0, 70.0), (200.0, 100.0)])
    s = BP.summarise_cells(cs)
    assert s["n"] == 4 and s["cells"] == [2, 1, 1], s
    assert abs(s["ratio"] - 0.625) < 1e-9, s
    # margins are own-opp per cell: +50, -80, 0, +100 -> median 25
    assert s["median_margin"] == 25.0, s
    assert abs(s["mean_margin"] - 17.5) < 1e-9, s
    assert s["median_own"] == 85.0 and s["median_opp"] == 80.0, s
    assert BP.summarise_cells([])["n"] == 0
    print(f"summarise_cells: {s['cells']} ratio {s['ratio']:.3f} "
          f"median margin {s['median_margin']:+.0f}")


def test_margin_resolves_where_cells_saturate():
    """The whole point of the repair: two agents that read IDENTICAL in the
    win column can be thousands of dollars apart per game."""
    a = _cells([(100000.0 + 1000 * i, 50000.0) for i in range(20)])
    b = _cells([(80000.0 + 1000 * i, 50000.0) for i in range(20)])
    sa, sb = BP.summarise_cells(a), BP.summarise_cells(b)
    assert sa["cells"] == sb["cells"] == [20, 0, 0], (sa, sb)
    assert sa["ratio"] == sb["ratio"] == 1.0
    assert sa["median_margin"] - sb["median_margin"] == 20000.0
    cmp_ = BP.compare_cells(a, b)
    assert cmp_["discordant"] == 0 and cmp_["p_value"] == 1.0, cmp_
    assert cmp_["margin_moved_winner_same"] == 20, cmp_
    assert cmp_["winner_moved"] == 0 and cmp_["identical_cells"] == 0
    assert cmp_["median_margin_delta"] == 20000.0
    assert cmp_["margin_better_a"] == 20 and cmp_["margin_better_b"] == 0
    assert cmp_["margin_sign_p"] < 1e-5, cmp_["margin_sign_p"]
    print("compare_cells: 20/20 cells identical in WINNERS, "
          f"+{cmp_['median_margin_delta']:,.0f} in dollars, sign p="
          f"{cmp_['margin_sign_p']:.2e}")


def test_dead_opponent_split_is_reported_separately():
    """A cell where the opponent never banked past its 3,000 start is a free
    win; the score must be readable without it, margin included."""
    cs = _cells([(120000.0, 3000.0), (120000.0, 0.0), (60000.0, 55000.0),
                 (40000.0, 70000.0)])
    s = BP.summarise_cells(cs)
    assert s["dead_opponent_cells"] == 2, s
    assert s["clean_cells"] == [1, 1] and s["clean_ratio"] == 0.5, s
    assert s["ratio"] == 0.75, s
    # the dead cells drag the raw median margin far above the clean one
    assert s["median_margin"] > s["clean_median_margin"], s
    # draws count as NOT-a-win in the clean record, exactly as before
    s2 = BP.summarise_cells(_cells([(50000.0, 50000.0), (60000.0, 40000.0)]))
    assert s2["clean_cells"] == [1, 1], s2
    print(f"dead-opponent guard: raw {s['ratio']:.3f} vs clean "
          f"{s['clean_ratio']:.3f}, margin {s['median_margin']:+,.0f} vs "
          f"{s['clean_median_margin']:+,.0f}")


def test_sign_test_is_exact_and_two_sided():
    assert BP.sign_test(0, 0) == 1.0
    assert BP.sign_test(5, 5) == 1.0
    assert abs(BP.sign_test(5, 0) - 2 * (1 / 32)) < 1e-12
    assert abs(BP.sign_test(0, 5) - 2 * (1 / 32)) < 1e-12, "symmetric"
    assert BP.sign_test(10, 0) < 0.005
    assert BP.sign_test(1, 1) == 1.0
    print("sign_test: exact binomial, two-sided, symmetric")


# -------------------------------------------------------------- holdout split

def test_assign_splits_is_stratified_and_deterministic():
    teams = [{"rid": f"r{i}", "lb_score": 2500.0 - 10.0 * i} for i in range(40)]
    BP.assign_splits(teams)
    n_hold = sum(1 for t in teams if t["split"] == "holdout")
    assert n_hold == 20, n_hold
    assert all(t["split"] in BP.SPLITS for t in teams)
    # STRATIFIED: both halves must span the band, not be "top half selects".
    hold = [t["lb_score"] for t in teams if t["split"] == "holdout"]
    sel = [t["lb_score"] for t in teams if t["split"] == "selection"]
    assert max(hold) > min(sel) and max(sel) > min(hold), (hold, sel)
    assert abs(sum(hold) / len(hold) - sum(sel) / len(sel)) < 20.0, \
        "the two halves must have near-equal mean rating"
    # DETERMINISTIC: re-running, and re-running on a shuffled list, agree.
    again = [{"rid": f"r{i}", "lb_score": 2500.0 - 10.0 * i}
             for i in range(39, -1, -1)]
    BP.assign_splits(again)
    m = {t["rid"]: t["split"] for t in again}
    assert all(m[t["rid"]] == t["split"] for t in teams), "order-independent"
    print(f"assign_splits: {len(sel)}/{n_hold}, mean rating "
          f"{sum(sel) / len(sel):.0f} vs {sum(hold) / len(hold):.0f}")


def test_assign_splits_handles_missing_ratings_and_odd_fractions():
    """The reactive panel has no leaderboard ratings at all."""
    teams = [{"rid": f"agent_{i}", "lb_score": None} for i in range(13)]
    BP.assign_splits(teams)
    n_hold = sum(1 for t in teams if t["split"] == "holdout")
    assert n_hold == 6 and len(teams) - n_hold == 7, n_hold
    quarter = [{"rid": f"r{i}", "lb_score": float(i)} for i in range(20)]
    BP.assign_splits(quarter, frac=0.25)
    assert sum(1 for t in quarter if t["split"] == "holdout") == 5
    print("assign_splits: rating-free rosters and non-half fractions both work")


def test_ensure_splits_backfills_only_when_needed():
    teams = [{"rid": "a", "lb_score": 1.0, "split": "selection"},
             {"rid": "b", "lb_score": 2.0, "split": "holdout"}]
    assert BP.ensure_splits(teams) is False, "already split -- must not reshuffle"
    assert teams[0]["split"] == "selection"
    bare = [{"rid": "a", "lb_score": 1.0}, {"rid": "b", "lb_score": 2.0}]
    assert BP.ensure_splits(bare) is True
    assert all(t["split"] in BP.SPLITS for t in bare)
    half = [{"rid": "a", "lb_score": 1.0, "split": "selection"},
            {"rid": "b", "lb_score": 2.0}]
    assert BP.ensure_splits(half) is True, "a partly-split manifest is not split"
    print("ensure_splits: back-fills a legacy manifest, never reshuffles a "
          "recorded one")


def test_seed_splits_and_cell_split():
    assert BP.seed_splits([501]) == {501: "selection"}, \
        "a single seed cannot be split -- the team axis carries the holdout"
    ss = BP.seed_splits([501, 502, 503, 504])
    assert sorted(ss) == [501, 502, 503, 504]
    assert sum(1 for v in ss.values() if v == "holdout") == 2, ss
    # default axis is the team; the seed axis is ignored
    assert BP.cell_split("holdout", "selection") == "holdout"
    assert BP.cell_split("selection", "holdout") == "selection"
    # with the seed axis on, the two must AGREE or the cell is neither
    assert BP.cell_split("holdout", "holdout", True) == "holdout"
    assert BP.cell_split("selection", "selection", True) == "selection"
    assert BP.cell_split("holdout", "selection", True) == "mixed"
    print("seed_splits/cell_split: team axis by default, strict AND with "
          "--seed-holdout")


def test_a_selection_winner_can_lose_the_holdout():
    """The v44_single failure mode, in miniature: an agent tuned to the cells
    it was selected on must be able to FAIL the held-out half, or the split is
    decoration."""
    teams = [{"rid": f"r{i}", "lb_score": 100.0 - i} for i in range(8)]
    BP.assign_splits(teams)
    cells = []
    for t in teams:
        win = t["split"] == "selection"       # only wins where it was selected
        cells.append({"own": 100.0 if win else 10.0,
                      "opp": 10.0 if win else 100.0,
                      "split": t["split"]})
    overall = BP.summarise_cells(cells)
    sel = BP.summarise_cells([c for c in cells if c["split"] == "selection"])
    hold = BP.summarise_cells([c for c in cells if c["split"] == "holdout"])
    assert overall["ratio"] == 0.5 and sel["ratio"] == 1.0
    assert hold["ratio"] == 0.0, hold
    assert hold["median_margin"] < 0 < sel["median_margin"]
    print(f"holdout discipline: selection {sel['ratio']:.3f}, held-out "
          f"{hold['ratio']:.3f} -- the split can and must disagree")


# ------------------------------------------------------ world distribution

def test_world_stats_against_the_ladder_reference():
    # a LADDER-LIKE panel: median ~85k, ~58% below 90k
    ws = [70000.0] * 58 + [110000.0] * 42
    s = BP.world_stats(_cells([(w, w) for w in ws]))
    assert s["n"] == 100 and abs(s["sub90k_share"] - 0.58) < 1e-9, s
    assert s["ladder_like"], s
    # the panel we actually had: 97% sub-90k at a ~50k median. NOT ladder-like.
    bad = [50000.0] * 97 + [200000.0] * 3
    sb = BP.world_stats(_cells([(w, w) for w in bad]))
    assert abs(sb["sub90k_share"] - 0.97) < 1e-9
    assert not sb["ladder_like"], sb
    assert sb["median_gap"] < -30000, sb
    assert sb["sub90k_gap"] > 0.3, sb
    assert BP.world_stats([])["n"] == 0
    print(f"world_stats: good panel ladder_like={s['ladder_like']}, "
          f"tape panel gap {sb['median_gap']:+,} / "
          f"{100 * sb['sub90k_gap']:+.0f}pp -> {sb['ladder_like']}")


def test_shop_stats_is_the_mechanism_check():
    """A tape never unlocks a shop, which is WHY its worlds are poor. The
    stat has to separate 'nothing ever opened' from 'the economy fired'."""
    dead = [{"own": 0.0, "opp": 0.0, "shops": []} for _ in range(10)]
    alive = [{"own": 0.0, "opp": 0.0,
              "shops": [(72, "PIZZA"), (145, "ICE_CREAM"), (200, "YARN")]}
             for _ in range(10)]
    sd, sa = BP.shop_stats(dead), BP.shop_stats(alive)
    assert sd["share_no_shop"] == 1.0 and sd["median_shops"] == 0, sd
    assert sd["median_first_unlock_step"] is None, sd
    assert sa["share_no_shop"] == 0.0 and sa["mean_shops"] == 3.0, sa
    assert sa["median_first_unlock_step"] == 72, sa
    assert BP.shop_stats([])["n"] == 0
    # a cell with no `shops` key counts as zero, never as missing data
    assert BP.shop_stats([{"own": 1.0, "opp": 1.0}])["share_no_shop"] == 1.0
    print(f"shop_stats: {sd['share_no_shop']:.0%} vs {sa['share_no_shop']:.0%} "
          "of cells with no shop unlocked")


def test_panel_tapes_halves_are_disjoint_and_cover():
    for band in BP.BAND_ORDER:
        if not os.path.exists(BP.band_manifest_path(band)):
            continue
        sel = BP.panel_tapes(band, "selection", existing_only=False)
        hold = BP.panel_tapes(band, "holdout", existing_only=False)
        allt = BP.panel_tapes(band, None, existing_only=False)
        assert not (set(sel) & set(hold)), band
        assert len(sel) + len(hold) == len(allt), band
        assert sel and hold, f"{band} has an empty half"
        print(f"panel_tapes {band}: {len(sel)} selection / {len(hold)} "
              "held-out, disjoint")


def test_world_bank_is_the_shared_bank():
    """The world's wealth is (own+opp)/2 -- a blowout win is not a rich
    world, and reading only our own bank would call it one."""
    s = BP.world_stats(_cells([(160000.0, 0.0)]))
    assert s["median_world_bank"] == 80000, s
    print("world_stats: shared bank, not our own")


# -------------------------------------------------------- seat-mirror dedupe

def test_dedupe_mirrored_halves_a_deterministic_panel():
    keys, a, b = [], {}, {}
    for ti in range(5):
        for seat in (0, 1):
            k = ("mid", ti, 501, seat)
            keys.append(k)
            a[k] = {"own": 100.0 + ti, "opp": 50.0}
            b[k] = {"own": 90.0 + ti, "opp": 50.0}
    keep = BP.dedupe_mirrored(sorted(keys), a, b)
    assert len(keep) == 5, keep
    assert all(k[-1] == 0 for k in keep), keep
    # a world where the seats genuinely DIFFER keeps both -- that is real
    # extra evidence, not a mirror
    k1 = ("mid", 9, 501, 0)
    k2 = ("mid", 9, 501, 1)
    a[k1] = {"own": 100.0, "opp": 50.0}
    a[k2] = {"own": 40.0, "opp": 90.0}
    b[k1] = b[k2] = {"own": 10.0, "opp": 10.0}
    keep2 = BP.dedupe_mirrored([k1, k2], a, b)
    assert len(keep2) == 2, keep2
    print("dedupe_mirrored: 10 cells -> 5 worlds; genuinely asymmetric seats "
          "are kept")


def test_dedupe_changes_the_p_value_it_should():
    """The bug this fixes: 24 mirrored cells reported p=0.023 where the 12
    real worlds give p=0.146."""
    a = _cells([(100.0 + i, 50.0) for i in range(6)] * 2)
    b = _cells([(200.0 + i, 50.0) for i in range(6)] * 2)
    doubled = BP.compare_cells(a, b)
    single = BP.compare_cells(a[:6], b[:6])
    assert doubled["margin_sign_p"] < single["margin_sign_p"], \
        (doubled["margin_sign_p"], single["margin_sign_p"])
    print(f"mirror inflation: p {doubled['margin_sign_p']:.4f} on doubled "
          f"cells vs {single['margin_sign_p']:.4f} on real worlds")


# ------------------------------------------------------------ rank agreement

def test_spearman_ranks_and_guards():
    assert abs(BP.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(BP.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
    assert BP.spearman([1, 2], [1, 2]) is None, "n<3 has no rank correlation"
    assert BP.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None, "no variance"
    r = BP.spearman([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
    assert 0.5 < r < 1.0, r
    # ties are averaged, not broken arbitrarily
    assert abs(BP.spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-9
    print(f"spearman: exact with ties, guarded below n=3; sample rho={r:.2f}")


# ------------------------------------------------------------ panel contract

def test_reactive_band_is_registered_and_barred():
    assert "reactive" in BP.BANDS and "reactive" in BP.BAND_ORDER
    assert BP.BARS["reactive"] == 0.90
    assert set(BP.BAND_ORDER) == set(BP.BANDS)
    assert set(BP.TAPE_BANDS) < set(BP.BANDS)
    assert "reactive" not in BP.TAPE_BANDS
    # the roster must be real files, or the "fix" is opponents that never play
    missing = [p for p in BP.REACTIVE_OPPS
               if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, missing
    assert len(BP.REACTIVE_OPPS) >= 10, BP.REACTIVE_OPPS
    print(f"reactive band: {len(BP.REACTIVE_OPPS)} live opponents, bar 0.90")


def test_ladder_reference_is_the_measured_one():
    """These two numbers are what every panel is judged against; a silent
    edit would move the definition of 'ladder-like'."""
    assert BP.LADDER_WORLD["median_world_bank"] == 85000.0
    assert BP.LADDER_WORLD["sub90k_share"] == 0.58
    assert BP.POOR_WORLD == 90000.0
    assert BP.SPLIT_FRAC == 0.5 and BP.SPLITS == ("selection", "holdout")
    print("ladder reference: 85k median / 58% sub-90k, from 8,000 traces")


if __name__ == "__main__":
    for fn in (test_summarise_cells_counts_and_margins,
               test_margin_resolves_where_cells_saturate,
               test_dead_opponent_split_is_reported_separately,
               test_sign_test_is_exact_and_two_sided,
               test_assign_splits_is_stratified_and_deterministic,
               test_assign_splits_handles_missing_ratings_and_odd_fractions,
               test_ensure_splits_backfills_only_when_needed,
               test_seed_splits_and_cell_split,
               test_a_selection_winner_can_lose_the_holdout,
               test_world_stats_against_the_ladder_reference,
               test_shop_stats_is_the_mechanism_check,
               test_panel_tapes_halves_are_disjoint_and_cover,
               test_world_bank_is_the_shared_bank,
               test_dedupe_mirrored_halves_a_deterministic_panel,
               test_dedupe_changes_the_p_value_it_should,
               test_spearman_ranks_and_guards,
               test_reactive_band_is_registered_and_barred,
               test_ladder_reference_is_the_measured_one):
        fn()
    print("\nall instrument-repair checks passed")
