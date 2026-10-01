"""Arithmetic of the banded crown gate (Workstream C). Pure, ~0ms."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kaggriculture.measure.crown_gate as CG


def _cells(best_wins, inc_wins, n):
    """n paired cells: first `best_wins` are best-only wins, next `inc_wins`
    are incumbent-only wins, the rest are ties (both win)."""
    best, inc = [], []
    for i in range(n):
        if i < best_wins:
            best.append(1.0); inc.append(0.0)
        elif i < best_wins + inc_wins:
            best.append(0.0); inc.append(1.0)
        else:
            best.append(1.0); inc.append(1.0)
    return best, inc


def test_clear_upgrade_ships():
    # Every band strongly better -> ship.
    per_band = {b: _cells(10, 0, 16) for b in CG.BAND_WEIGHT}
    v = CG.crown_gate_banded(per_band)
    assert v["ship"], v
    assert v["aggregate"] > 0
    assert not v["regressed_bands"]


def test_one_band_significant_regression_holds():
    per_band = {b: _cells(8, 0, 16) for b in CG.BAND_WEIGHT}
    per_band["2700+"] = _cells(0, 10, 16)          # top band clearly worse
    v = CG.crown_gate_banded(per_band)
    assert not v["ship"], v
    assert "2700+" in v["regressed_bands"], v


def test_flat_no_upgrade_holds():
    # No discordant pairs anywhere -> aggregate 0 -> not an upgrade.
    per_band = {b: _cells(0, 0, 12) for b in CG.BAND_WEIGHT}
    v = CG.crown_gate_banded(per_band)
    assert not v["ship"], v
    assert v["aggregate"] == 0.0


def test_ladder_weighting_favours_top_band():
    # Same per-band diff, but the aggregate must weight 2700+ above <2100.
    low = {"<2100": _cells(6, 0, 12)}
    high = {"2700+": _cells(6, 0, 12)}
    assert CG.aggregate({b: CG.band_verdict(*low["<2100"])
                         for b in ["<2100"]})[0] > 0
    a_low = CG.crown_gate_banded(low)["aggregate"]
    a_high = CG.crown_gate_banded(high)["aggregate"]
    # identical per-band score diff, so aggregates equal; weighting shows when
    # bands are mixed:
    mixed_topheavy = {"<2100": _cells(2, 0, 12), "2700+": _cells(10, 0, 12)}
    mixed_lowheavy = {"<2100": _cells(10, 0, 12), "2700+": _cells(2, 0, 12)}
    assert (CG.crown_gate_banded(mixed_topheavy)["aggregate"]
            > CG.crown_gate_banded(mixed_lowheavy)["aggregate"]), \
        "top-heavy improvement must aggregate higher than low-heavy"
    assert abs(a_low - a_high) < 1e-9


def test_holdout_alarm_flags_divergence():
    # Selection shipped, but a reserved band regresses -> diverge.
    holdout = {"2500-2700": _cells(0, 9, 16)}
    al = CG.holdout_alarm(True, holdout)
    assert al["diverges"], al
    # Selection held -> never a divergence, whatever holdout does.
    assert not CG.holdout_alarm(False, holdout)["diverges"]


def test_empty_is_hold():
    v = CG.crown_gate_banded({})
    assert not v["ship"]
    assert v["bands"] == {}


if __name__ == "__main__":
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
