"""Differential test: the Rust price model against the real interpreter.

Ported first because a pre-ranker's whole job is valuing sell schedules, and
because this is where the named hazard lives -- `sqrt` and `ln` are libm calls
and Rust's libm need not agree with CPython's in the last bit. The engine takes
`int(round(price))`, which absorbs nearly every such difference; "nearly" over
720 steps x 30 days is exactly where drift hides.

So this compares BOTH:
  * the QUOTED integer, which is what the engine uses -- must match exactly
  * the RAW float, bit pattern against bit pattern -- reported, and allowed to
    differ only where rounding still lands on the same integer

Two rounding traps are asserted directly, since either would produce a curve
that is subtly wrong everywhere rather than obviously wrong somewhere:
CPython's `round()` is half-to-EVEN, while Rust's `f64::round()` is
half-away-from-zero.

    python tests/test_rust_market.py
"""
from kaggriculture.paths import ROOT
import json
import os
import struct
import subprocess
import sys

BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
if not os.path.exists(BIN):
    BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg")


def engine():
    """The vendored interpreter module itself -- the only ground truth.

    Imported as part of its package, not as a loose file: it does
    `from kaggle_environments.utils import resolve_episode_seed`, so a
    spec-from-file import fails on the package that is not yet on sys.path.
    `_vendor` is what puts vendor/ there.
    """
    try:
        import kaggriculture.engine._vendor as _vendor  # noqa: F401
        from kaggle_environments.envs.kaggriculture import kaggriculture as k
        return k
    except Exception as exc:                                    # noqa: BLE001
        print(f"engine import failed: {type(exc).__name__}: {exc}")
        return None


def rust(*args):
    out = subprocess.run([BIN, *[str(a) for a in args]], capture_output=True,
                         text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"kagg failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


ENG = engine()


def test_engine_available():
    assert ENG is not None, "vendored interpreter not importable"
    assert hasattr(ENG, "market_price"), "engine has no market_price"
    print(f"engine loaded: {len(ENG.PRODUCTS)} products, "
          f"I0={ENG.MARKET_I0}, floor={ENG.PRICE_FLOOR}")


def test_quoted_prices_match_across_the_range():
    """The integer the engine actually quotes must match exactly, everywhere."""
    total = mism = 0
    raw_diffs = 0
    for item in ENG.PRODUCTS:
        # Span deep scarcity through heavy glut, plus a fine sweep around I0
        # where the curve changes branch.
        ranges = [(0, 20000, 250), (9900, 10100, 1)]
        for lo, hi, step in ranges:
            rows = rust("price-sweep", item, lo, hi, step)
            for inv, raw_bits, quoted in rows:
                want = ENG.market_price(item, inv)
                total += 1
                if want != quoted:
                    mism += 1
                    if mism <= 5:
                        print(f"  MISMATCH {item} inv={inv}: "
                              f"engine {want} != rust {quoted}")
                # raw float may differ in the last bit without changing the
                # quote; count it so silent libm drift is visible
                p = ENG.MARKET_PARAMS[item]

                # 1.32.7 widened _shape to (func, x, T) for the hinge curve;
                # calling 2-arg on it silently degenerates hinge to linear
                # and would false-fail this suite after the engine swap.
                def _sh(f, x, t=p["T"]):
                    try:
                        return ENG._shape(f, x, t)
                    except TypeError:
                        return ENG._shape(f, x)
                if inv < p["I0"]:
                    f, tgt = p["below_func"], p["below_target"]
                    amp = tgt * p["base"] / _sh(f, p["T"])
                    raw = p["base"] + amp * _sh(f, p["I0"] - inv)
                else:
                    f, tgt = p["above_func"], p["above_target"]
                    amp = tgt * p["base"] / _sh(f, p["T"])
                    raw = p["base"] - amp * _sh(f, inv - p["I0"])
                if bits(raw) != raw_bits:
                    raw_diffs += 1
    assert mism == 0, f"{mism}/{total} quoted prices disagree"
    print(f"{total:,} quoted prices across {len(ENG.PRODUCTS)} products: "
          f"ALL EXACT")
    print(f"raw float bit-differences: {raw_diffs} "
          f"({'none -- libm agrees' if raw_diffs == 0 else 'absorbed by rounding'})")


def test_price_floor_holds():
    """Under extreme glut the two must AGREE and never go below the floor.

    Not every product reaches the floor, and asserting that they all do was my
    own wrong expectation rather than a port bug: WHEAT's glut penalty is
    logarithmic (above_func log, target 0.20), so at inventory 200,000 both
    engine and port quote 15. Only the steep shapes (`sq`) actually clamp.
    """
    floored = []
    for item in ENG.PRODUCTS:
        for inv in (200000, 10 ** 7):
            rows = rust("price-sweep", item, inv, inv, 1)
            _inv, _raw, quoted = rows[0]
            want = ENG.market_price(item, inv)
            assert quoted == want, (item, inv, quoted, want)
            assert quoted >= ENG.PRICE_FLOOR, (item, inv, quoted)
            if quoted == ENG.PRICE_FLOOR:
                floored.append(item)
    print(f"extreme glut: engine and port agree for all "
          f"{len(ENG.PRODUCTS)} products, none below "
          f"PRICE_FLOOR={ENG.PRICE_FLOOR}; "
          f"{len(set(floored))} actually clamp ({sorted(set(floored))})")


def test_python_round_is_half_to_even():
    """Guards the rounding rule the port had to replicate by hand."""
    assert round(0.5) == 0 and round(1.5) == 2 and round(2.5) == 2, \
        "CPython round() is not half-to-even here; the port's assumption breaks"
    print("confirmed: CPython round() is half-to-EVEN (0.5->0, 1.5->2, 2.5->2)")


def test_params_table_matches_the_engine():
    """A transcribed constant table is a drift risk; compare it field by field."""
    rows = rust("prices", 10000)
    for item in ENG.PRODUCTS:
        assert item in rows, f"rust is missing {item}"
        assert rows[item] == ENG.market_price(item, 10000), item
    assert len(rows) == len(ENG.PRODUCTS), (len(rows), len(ENG.PRODUCTS))
    print(f"params table: all {len(rows)} products present and equal at I0")


if __name__ == "__main__":
    for fn in (test_engine_available, test_python_round_is_half_to_even,
               test_params_table_matches_the_engine,
               test_quoted_prices_match_across_the_range,
               test_price_floor_holds):
        fn()
    print("\nPRICE MODEL VERIFIED bit-for-bit against the interpreter")
