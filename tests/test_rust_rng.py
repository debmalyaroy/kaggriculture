"""D1: does the Rust RNG match CPython's `random.Random` bit-for-bit?

This is the gate on the whole engine port. If the stream matches, a
differential harness can assert identical engine state at every step and the
port is a mechanical exercise. If it does not, a Rust engine can only ever be
an approximate pre-ranker and never a gate.

Only two consumers exist in the interpreter, both on an RNG re-seeded per day:

    rng = random.Random((seed * 1_000_003) ^ day)
      rng.random() < weed_chance     # ~100 empty tiles x 2 players per day
      rng.choice(sorted(SHOPS))      # once per shop-unlock day

Floats are compared as EXACT bit patterns, not approximately: an engine that
agrees to 1e-9 is an engine that will diverge somewhere over 720 steps x 30
days, and the whole point of the harness is to detect that before it costs a
measurement.

    python tests/test_rust_rng.py
"""
from kaggriculture.paths import ROOT
import json
import os
import random
import struct
import subprocess
import sys

BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
if not os.path.exists(BIN):
    BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg")


def engine_src():
    """The official interpreter source: the optional vendor/ copy, else the pip-installed kaggle-environments."""
    v = os.path.join(ROOT, "vendor", "kaggle_environments", "envs", "kaggriculture", "kaggriculture.py")
    if os.path.exists(v):
        return v
    import kaggle_environments
    return os.path.join(os.path.dirname(kaggle_environments.__file__), "envs", "kaggriculture", "kaggriculture.py")

def _engine_shops():
    """PARSE sorted(SHOPS) out of the interpreter rather than hardcoding it.

    The first version of this test hardcoded six names taken from one replay's
    observed `unlocked_shops`, but the engine defines EIGHT. Because the same
    wrong list was used on the Python and Rust sides, `choice()` agreed with
    itself and disagreed with the engine -- `randbelow(6)` consumes a different
    number of draws than `randbelow(8)`. Read the source; never restate it.
    """
    import re
    eng = engine_src()
    txt = open(eng, encoding="utf-8", errors="replace").read()
    m = re.search(r"^SHOPS\s*=\s*\{(.*?)^\}", txt, re.S | re.M)
    if not m:
        raise AssertionError("could not parse SHOPS from the engine source")
    return sorted(re.findall(r'"([A-Z_]+)"\s*:', m.group(1)))


SHOPS = _engine_shops()
SEEDS = (1117071212, 1381306337, 1, 0, 987654321987)
DAYS = (0, 1, 2, 3, 7, 29)


def bits(x):
    """The float's IEEE-754 bit pattern as an int, so equality is bit equality.

    The Rust side emits `f64::to_bits()` for the same reason: a decimal
    rendering has to be chosen carefully to round-trip, and the first version
    of this test used Rust's `{:.17}` (17 digits after the point, not 17
    significant) which silently truncated small values and looked like an RNG
    divergence.
    """
    if isinstance(x, int):
        return x
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def unbits(n):
    """Bit pattern back to the float it denotes."""
    return struct.unpack("<d", struct.pack("<Q", int(n)))[0]


def rust(*args):
    out = subprocess.run([BIN, *[str(a) for a in args]], capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"kagg failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def test_shops_list_matches_the_engine():
    """Length is load-bearing: choice() -> randbelow(len(SHOPS))."""
    assert len(SHOPS) == 8, (len(SHOPS), SHOPS)
    print(f"engine SHOPS: {len(SHOPS)} entries {SHOPS}")


def test_binary_exists():
    assert os.path.exists(BIN), (
        f"{BIN} missing -- run: cd rustengine && cargo build --release")
    print(f"binary: {os.path.relpath(BIN, ROOT)}")


def test_rng_matches_cpython():
    checked = 0
    for seed in SEEDS:
        for day in DAYS:
            got = rust("rng-probe", seed, day)

            r = random.Random((seed * 1_000_003) ^ day)
            want_rand = [r.random() for _ in range(8)]
            r2 = random.Random((seed * 1_000_003) ^ day)
            want_bits = [r2.getrandbits(32) for _ in range(4)]
            r3 = random.Random((seed * 1_000_003) ^ day)
            want_choice = [r3.choice(SHOPS) for _ in range(4)]

            for i, (a, b) in enumerate(zip(want_rand, got["first8_random"])):
                assert bits(a) == bits(b), (
                    f"seed {seed} day {day} random()[{i}]: "
                    f"python {a!r} ({bits(a)}) != rust {b!r} ({bits(b)})")
            assert want_bits == got["getrandbits32_first4"], (
                f"seed {seed} day {day} getrandbits: "
                f"{want_bits} != {got['getrandbits32_first4']}")
            assert want_choice == got["choice_first4"], (
                f"seed {seed} day {day} choice: "
                f"{want_choice} != {got['choice_first4']}")
            checked += 1
    print(f"{checked} (seed, day) pairs: random() bit-exact, getrandbits "
          f"exact, choice() exact")


def test_weed_sequence_matches():
    """The actual consumer: a long run of random() for one day's weed rolls."""
    for seed in (1117071212, 1381306337):
        for day in (0, 6, 17):
            n = 240                       # ~100 tiles x 2 players, with slack
            got = rust("weeds", seed, day, n)
            r = random.Random((seed * 1_000_003) ^ day)
            want = [r.random() for _ in range(n)]
            assert [bits(x) for x in want] == [bits(x) for x in got], (
                f"weed sequence diverged for seed {seed} day {day}")
            # and the DERIVED DECISION at the engine's real threshold -- the
            # only thing the engine actually uses the draw for. `got` carries
            # bit patterns, so convert before comparing against a float.
            chance = 0.005
            assert ([x < chance for x in want]
                    == [unbits(b) < chance for b in got])
    print("weed draw sequences bit-exact over 240 draws x 6 (seed, day) pairs")


def test_large_key_seeding():
    """The per-day key exceeds 2^32, so init_by_array over 32-bit words is the
    only correct seeding path; a truncated init_genrand would pass small seeds
    and fail here."""
    seed = 987654321987
    key = (seed * 1_000_003) ^ 29
    assert key > 2 ** 32, key
    got = rust("rng-probe", seed, 29)
    r = random.Random(key)
    assert bits(r.random()) == bits(got["first8_random"][0])
    print(f"large key {key} (> 2^32) seeds identically")


def test_scaffold_constants_match_the_engine():
    """The Rust scaffold's constants must equal the interpreter's.

    Same reasoning as src/engine_check.py, which exists because a Kaggle image
    once shipped startingMoney 2000 against the ladder's 3000 and nothing
    raised. A port that drifts on a constant answers a different question
    confidently, so pin them from the vendored source rather than by eye.
    """
    got = rust("selftest")
    src = engine_src()
    if not os.path.exists(src):
        print("SKIP: vendored engine not present")
        return
    text = open(src, encoding="utf-8", errors="replace").read()
    # Defaults live in the interpreter's `get(cfg, "<key>", <default>)` calls.
    import re
    def default_for(key):
        m = re.search(r'get\(cfg,\s*"%s",\s*([0-9]+)\)' % key, text)
        return int(m.group(1)) if m else None
    want = {"board": default_for("boardSize"),
            "shed_cap": default_for("shedCapacity"),
            "max_orders": default_for("maxMarketOrdersPerTurn")}
    for k, v in want.items():
        if v is None:
            continue
        assert got[k] == v, f"{k}: rust {got[k]} != engine {v}"
    assert got["starting_money"] == 3000, got["starting_money"]
    assert got["steps"] == 720, got["steps"]
    shown = {k: v for k, v in want.items() if v is not None}
    print(f"scaffold constants match the vendored engine: {shown} "
          f"+ startingMoney 3000, episodeSteps 720")


if __name__ == "__main__":
    for fn in (test_shops_list_matches_the_engine,
               test_binary_exists, test_rng_matches_cpython,
               test_weed_sequence_matches, test_large_key_seeding,
               test_scaffold_constants_match_the_engine):
        fn()
    print("\nD1 PASS -- the random stream is reproducible in Rust; "
          "bit-exact differential testing of the engine is available")
