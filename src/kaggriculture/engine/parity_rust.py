from kaggriculture.paths import ROOT
# LEGACY (2026-08-16 audit): pre-dates the full rustengine port; references
# rust/step_engine/*.dll which no longer exists, so it cannot run. Its
# embedded 1.32.6 MARKET_PARAMS are deliberately NOT hinge-patched -- use
# tests/test_rust_market.py for price parity instead.
"""Parity gate for the Rust market kernel vs the verified Python curve.

The Rust engine is unusable until this passes, and must re-pass after every
upstream engine bump. Level 1 (this file): the price curve, exhaustively --
every product x every integer inventory 0..30,000 -- against the Python
`_quote` that was itself verified against the interpreter at zero error.
Level 2 (with the field-phase port): full-episode state parity.

    python -m kaggriculture.engine.parity_rust
"""
import math
import os
import shutil
import sys

DLL = os.path.join(ROOT, "rust", "step_engine", "target", "release",
                   "step_engine.dll")
PYD = os.path.join(ROOT, "rust", "step_engine", "step_engine.pyd")

MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}


def shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    return math.log1p(value)


def py_quote(item, inventory):
    base, eq, scale, bf, bt, af, at = MARKET_PARAMS[item]
    if inventory < eq:
        amp = bt * base / shape(bf, scale)
        price = base + amp * shape(bf, eq - inventory)
    else:
        amp = at * base / shape(af, scale)
        price = base - amp * shape(af, inventory - eq)
    return max(1, int(round(price)))


def main():
    if not os.path.exists(PYD) or (os.path.getmtime(DLL) > os.path.getmtime(PYD)
                                   if os.path.exists(PYD) else True):
        shutil.copy2(DLL, PYD)
    sys.path.insert(0, os.path.dirname(PYD))
    import step_engine

    worst = None
    mismatches = 0
    checked = 0
    for item in MARKET_PARAMS:
        for inv in range(0, 30001):
            a = py_quote(item, float(inv))
            b = step_engine.quote(item, float(inv))
            checked += 1
            if float(a) != b:
                mismatches += 1
                if worst is None:
                    worst = (item, inv, a, b)
    if mismatches:
        print(f"PARITY FAIL: {mismatches}/{checked} mismatches; first at "
              f"{worst[0]} inv={worst[1]}: py {worst[2]} vs rust {worst[3]}")
        return 1
    print(f"PARITY OK: {checked:,} price points, zero mismatches")

    import time
    t0 = time.time()
    n = 200_000
    for i in range(n):
        step_engine.quote("MELON", float(i % 30000))
    rust_rate = n / (time.time() - t0)
    t0 = time.time()
    for i in range(n):
        py_quote("MELON", float(i % 30000))
    py_rate = n / (time.time() - t0)
    print(f"throughput: rust {rust_rate:,.0f}/s vs python {py_rate:,.0f}/s "
          f"({rust_rate / py_rate:.1f}x, includes FFI overhead; batch APIs "
          f"will amortize it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
