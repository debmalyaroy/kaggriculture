"""Contract tests for src/win_metric.py -- now load-bearing for every gate.

The crown gate, the collapse alarm, the surrogate target and the ablation
drivers all read this module, so a silent error here mis-decides releases.

    python tests/test_win_metric.py
"""
import os
import sys

import kaggriculture.measure.win_metric as WM  # noqa: E402


def test_score():
    assert WM.score(100, 50) == 1.0
    assert WM.score(50, 100) == 0.0
    assert WM.score(70, 70) == 0.5, "a draw is half a point, not a loss"
    # $1 and $10,000 must be paid identically -- the whole point
    assert WM.score(1001, 1000) == WM.score(11000, 1000) == 1.0
    print("score: win/draw/loss correct; $1 == $10,000")


def test_expected_score_and_summarise():
    pairs = [(10, 5), (5, 10), (7, 7), (9, 1)]
    w, d, l, sc = WM.summarise(pairs)
    assert (w, d, l) == (2, 1, 1), (w, d, l)
    assert abs(sc - 0.625) < 1e-9, sc
    assert abs(WM.expected_score([1, 0, 0.5, 1]) - 0.625) < 1e-9
    print(f"summarise: W-D-L 2-1-1 -> score {sc:.3f}")


def test_flips_is_exact():
    # three games lost by 100, 5,000 and 50,000; a +1,000 effect flips ONE
    base = [-100.0, -5000.0, -50000.0]
    gained, lost, net = WM.flips(base, [1000.0] * 3)
    assert (gained, lost) == (1, 0), (gained, lost)
    assert abs(net - 1 / 3) < 1e-9, net
    # a mean-margin view would call +1,000/game the same in all three; flips
    # correctly prices only the one that crosses zero
    gained2, lost2, _ = WM.flips(base, [60000.0, 0.0, 0.0])
    assert (gained2, lost2) == (1, 0)
    # and an effect can LOSE games
    g3, l3, net3 = WM.flips([100.0, 200.0], [-500.0, -500.0])
    assert (g3, l3) == (0, 2), (g3, l3)
    assert net3 < 0
    print("flips: exact at the zero crossing, and can price losses")


def test_dollars_to_wins_monotone():
    m = [-100.0, -900.0, -9000.0, 500.0, 20000.0]
    a = WM.dollars_to_wins(m, 200)
    b = WM.dollars_to_wins(m, 1000)
    c = WM.dollars_to_wins(m, 10000)
    assert 0 <= a <= b <= c <= 1, (a, b, c)
    assert abs(a - 0.2) < 1e-9, a          # only the -100 game flips
    print(f"dollars_to_wins: monotone {a:.2f} <= {b:.2f} <= {c:.2f}")


def test_paired_test_detects_one_directional():
    # B better on 6 pairs, A never better, rest tied -> significant
    a = [0.0] * 6 + [1.0] * 15
    b = [1.0] * 6 + [1.0] * 15
    r = WM.paired_test(a, b)
    assert r["better_b"] == 6 and r["better_a"] == 0, r
    assert r["p_value"] < 0.05, r
    assert r["score_diff"] < 0
    # identical builds -> not significant
    r2 = WM.paired_test(a, a)
    assert r2["discordant"] == 0 and r2["p_value"] == 1.0
    assert not r2["significant"]
    print(f"paired_test: 6-0 discordant -> p={r['p_value']:.4f}; "
          f"identical -> p=1.0")


def test_parse_eval_strict():
    good = ("noise line\n"
            "RESULT\topp_a.py\t0.7500\t3\t0\t1\t4\t100.0\t90.0\t10.0\n"
            "RESULT\topp_b.py\t0.5000\t1\t0\t1\t2\t50.0\t60.0\t-10.0\n")
    rows = WM.parse_eval(good)
    assert len(rows) == 2, rows
    assert rows[0]["opponent"] == "opp_a.py" and rows[0]["wins"] == 3
    sc, wins, games = WM.totals(rows)
    assert (wins, games) == (4, 6), (wins, games)
    assert abs(sc - 4 / 6) < 1e-9
    try:
        WM.parse_eval("nothing here")
    except ValueError:
        pass
    else:
        raise AssertionError("strict parse must RAISE, not return 0/0 -- a "
                             "silent zero is how the crown gate got garbage")
    assert WM.parse_eval("nothing", strict=False) == []
    print("parse_eval: reads RESULT lines; raises rather than returning 0/0")


def test_min_detectable_shrinks():
    small = WM.min_detectable(28)
    big = WM.min_detectable(1000)
    assert big < small, (small, big)
    # observed low discordance must make it MORE powerful than the prior
    assert WM.min_detectable(84, discordance=0.1) < WM.min_detectable(84)
    print(f"min_detectable: {100 * small:.1f}pp @28 -> "
          f"{100 * big:.1f}pp @1000")


if __name__ == "__main__":
    for fn in (test_score, test_expected_score_and_summarise,
               test_flips_is_exact, test_dollars_to_wins_monotone,
               test_paired_test_detects_one_directional,
               test_parse_eval_strict, test_min_detectable_shrinks):
        fn()
    print("\nall win_metric checks passed")
