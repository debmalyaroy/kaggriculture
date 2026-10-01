"""Differential harness: the Rust engine vs the official interpreter.

Bit-identical FULL STATE at every step, or it fails -- final rewards are not
enough, because two engines can agree on the bank while disagreeing about a
tile, and that bug then surfaces on a seed nobody tested. A competitor
validated their port on 4,000 episodes; this harness is the machinery for the
same bar, run here over scripted chaos episodes plus real ladder replays.

Two tape sources, deliberately different:

* SCRIPTED chaos -- a seeded generator emitting every op the engine knows,
  legal or not, plus malformed market orders. Random soup exercises the
  silent-no-op paths that real agents avoid (illegal ops are the engine's
  hardest contract: bugs look like laziness).
* REAL replays -- recorded ladder episodes driven through both engines under
  the recorded seed. Realistic distributions: heavy selling, dense farms,
  hands, the paths that random soup starves.

The digest format is defined ONCE on each side (state.rs digest() and
`digest()` below) and includes every field of every tile, both private
blocks, money as IEEE-754 bit patterns, and the market. Any divergence prints
the step and both digests.

    python tests/test_rust_engine.py            # 3 scripted + up to 3 replays
    python tests/test_rust_engine.py --episodes 20 --replays 10
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import random
import struct
import subprocess
import sys


KAGG = os.path.join(ROOT, "rustengine", "target", "release",
                    "kagg.exe" if os.name == "nt" else "kagg")
TAPE_DIR = os.path.join(ROOT, ".local", "rustdiff")

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
            "MILK", "WOOL", "FERTILIZER"]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def fmt_sorted(d):
    return ",".join(f"{k}={int(v)}" for k, v in sorted(d.items()))


def digest(state, step):
    """EXACT mirror of state.rs State::digest()."""
    obs0 = state[0]["observation"]
    parts = [f"t{step}"]
    for i, farm in enumerate(obs0["farms"]):
        hands = " ".join(f"{p[0]},{p[1]}" for p in farm["hands"])
        s = (f"f{i}:m{bits(farm['money'])};"
             f"p{farm['farmer'][0]},{farm['farmer'][1]};"
             f"h{hands};r{farm['hires_today']};"
             f"q{','.join(farm['unlocked_quadrants'])};")
        cells = []
        for row in farm["tiles"]:
            for t in row:
                if t is None:
                    cells.append(".")
                elif t == "LOCKED":
                    cells.append("L")
                elif t.get("kind") == "WEED":
                    cells.append("W")
                elif t.get("kind") == "PLANT":
                    cells.append(
                        f"[P:{t['crop']},{t['planted_day']},"
                        f"{int(t['watered_today'])},"
                        f"{t['consecutive_unwatered']},{t['yield_units']},"
                        f"{t['max_lifespan_step']},"
                        f"{t.get('fertilized_until_day', -1)}]")
                elif "animal" in t:
                    cells.append(
                        f"[A:{t['kind']},{t['animal']},{t['placed_day']},"
                        f"{t['yield_units']},{t['consecutive_unfed']},"
                        f"{int(t['fed_today'])},{int(t['cared_today'])},"
                        f"{int(t['fertilizer_available'])},"
                        f"{t.get('pending_care_bonus', 0)}]")
                else:
                    cells.append(f"[S:{t['kind']}]")
        parts.append(s + "".join(cells))
    for i in range(2):
        priv = state[i]["observation"]["private"]
        invs = "/".join(fmt_sorted(v) for v in priv["inventories"])
        parts.append(f"s{i}:{fmt_sorted(priv['shed'])};"
                     f"{fmt_sorted(priv['seeds'])};{invs}")
    mkt = obs0["market"]
    parts.append(f"mk:{fmt_sorted(mkt['inventory'])};"
                 f"{fmt_sorted(mkt['prices'])}")
    parts.append(f"tw:{','.join(obs0['town']['unlocked_shops'])}")
    return "|".join(parts)


# ------------------------------------------------------------------- tapes --

def scripted_actions(rng):
    """One player's turn of seeded chaos: every op, legal or not."""
    def unit():
        r = rng.random()
        if r < 0.22:
            return [rng.choice(["NORTH", "SOUTH", "EAST", "WEST"])]
        if r < 0.36:
            return ["WATER"]
        if r < 0.46:
            return ["HARVEST"]
        if r < 0.56:
            return ["PLANT", rng.choice(CROPS)]
        if r < 0.63:
            return ["DROP"]
        if r < 0.70:
            return ["PICKUP", rng.choice(PRODUCTS + ANIMALS),
                    rng.randint(1, 5)]
        if r < 0.75:
            return (["PLACE", rng.choice(ANIMALS)] if rng.random() < 0.5
                    else ["PLACE", rng.choice(PRODUCTS), rng.randint(1, 4)])
        if r < 0.80:
            return ["FEED"]
        if r < 0.84:
            return ["CARE"]
        if r < 0.87:
            return ["COLLECT_FERTILIZER"]
        if r < 0.90:
            return ["FERTILIZE"]
        if r < 0.93:
            return ["DIG"]
        if r < 0.95:
            return ["BUILD_COOP"]
        if r < 0.97:
            return ["BUILD_PASTURE"]
        return ["PASS"]

    market = []
    for _ in range(rng.randint(0, 3)):
        r = rng.random()
        if r < 0.25:
            market.append(["BUY_SEED", rng.choice(CROPS), rng.randint(1, 6)])
        elif r < 0.55:
            market.append(["SELL", rng.choice(PRODUCTS), rng.randint(1, 30)])
        elif r < 0.70:
            market.append(["BUY_PRODUCT", rng.choice(["WHEAT", "FERTILIZER"]),
                           rng.randint(1, 4)])
        elif r < 0.80:
            market.append(["BUY_ANIMAL", rng.choice(ANIMALS), 1])
        elif r < 0.90:
            market.append(["HIRE"])
        elif r < 0.95:
            market.append(["BUY_LAND"])
        else:
            # Malformed on purpose: both engines must ignore it identically.
            market.append(rng.choice([["SELL", "GOOSE", 3], ["SELL"],
                                      ["BUY_PRODUCT", "MILK", 2],
                                      ["NONSENSE", "X", 1]]))
    return {"farmer": unit(),
            "hands": [unit() for _ in range(rng.randint(0, 3))],
            "market": market}


def norm_order(o):
    """Normalise an order so BOTH engines see identical input: the third
    element becomes a plain int when numeric (Python would int() a float
    where Rust's i64 parse would refuse -- same data, different verdicts)."""
    if not isinstance(o, list):
        return None
    out = []
    for v in o:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            out.append(str(int(v)))
        elif isinstance(v, str) and " " not in v and "\t" not in v:
            out.append(v)
        else:
            return None
    return out


def norm_unit(a):
    if not isinstance(a, list) or not a:
        return ["PASS"]
    return norm_order(a) or ["PASS"]


def tape_line(action):
    farmer = " ".join(norm_unit(action.get("farmer")))
    hands = ";".join(" ".join(norm_unit(h))
                     for h in (action.get("hands") or []))
    market = ";".join(" ".join(o) for o in
                      filter(None, (norm_order(m)
                                    for m in (action.get("market") or []))))
    return f"{farmer}\t{hands}\t{market}"


def denorm(action):
    """The dict the PYTHON engine gets -- rebuilt FROM the tape encoding so
    both engines consume byte-equivalent input."""
    def parse_unit(tokens):
        return [int(t) if t.lstrip("-").isdigit() else t for t in tokens]
    farmer, hands, market = tape_line(action).split("\t")
    return {
        "farmer": parse_unit(farmer.split(" ")),
        "hands": [parse_unit(h.split(" ")) for h in hands.split(";") if h],
        "market": [parse_unit(o.split(" ")) for o in market.split(";") if o],
    }


# --------------------------------------------------------------- episodes --

def run_python(seed, action_pairs):
    """Drive the OFFICIAL interpreter step by step; return per-step digests."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": seed},
               info={"seed": seed})
    env.reset(2)
    digests = []
    for a0, a1 in action_pairs:
        if env.done:
            break
        env.step([denorm(a0), denorm(a1)])
        state = [{"observation": dict(s.observation)} for s in env.state]
        digests.append(digest(state, env.state[0].observation.step))
    banks = [float(env.state[i].observation.farms[i]["money"])
             for i in range(2)]
    return digests, banks


def run_rust(seed, action_pairs, name):
    os.makedirs(TAPE_DIR, exist_ok=True)
    tape = os.path.join(TAPE_DIR, f"{name}.tape")
    with open(tape, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {seed}\n")
        for a0, a1 in action_pairs:
            fh.write(tape_line(a0) + "\n")
            fh.write(tape_line(a1) + "\n")
    out = subprocess.run([KAGG, "episode", tape], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[:400]
    digests, final = [], None
    for line in out.stdout.splitlines():
        if line.startswith("FINAL "):
            _, b0, b1 = line.split()
            final = [struct.unpack("<d", struct.pack("<Q", int(b0)))[0],
                     struct.unpack("<d", struct.pack("<Q", int(b1)))[0]]
        else:
            digests.append(line.split(" ", 1)[1])
    return digests, final, tape


def compare(name, seed, action_pairs):
    py, py_banks = run_python(seed, action_pairs)
    rs, rs_banks, tape = run_rust(seed, action_pairs[:len(py)], name)
    n = min(len(py), len(rs))
    for k in range(n):
        if py[k] != rs[k]:
            # Show the first differing field, not two 4KB strings.
            pp, rr = py[k].split("|"), rs[k].split("|")
            for a, b in zip(pp, rr):
                if a != b:
                    print(f"\nFAIL {name}: step {k + 1} diverges")
                    print(f"  python: {a[:220]}")
                    print(f"  rust:   {b[:220]}")
                    print(f"  tape: {tape}")
                    return False
            print(f"\nFAIL {name}: step {k + 1} diverges (length)")
            return False
    if len(py) != len(rs):
        print(f"\nFAIL {name}: python ran {len(py)} steps, rust {len(rs)}")
        return False
    if py_banks != rs_banks:
        print(f"\nFAIL {name}: final banks {py_banks} vs {rs_banks}")
        return False
    print(f"  {name}: {n} steps bit-identical, final banks "
          f"{py_banks[0]:,.0f} / {py_banks[1]:,.0f}")
    return True


def replay_pairs(path):
    rep = json.load(open(path, encoding="utf-8"))
    seed = (rep.get("info") or {}).get("seed")
    steps = rep.get("steps") or []
    if seed is None or len(steps) < 100:
        return None, None
    pairs = []
    for st in steps[1:]:
        acts = []
        for i in (0, 1):
            a = st[i].get("action") if i < len(st) else None
            acts.append(a if isinstance(a, dict) else {})
        pairs.append(tuple(acts))
    return int(seed), pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=3,
                    help="scripted chaos episodes")
    ap.add_argument("--replays", type=int, default=3,
                    help="real ladder replays (if staged locally)")
    ap.add_argument("--steps", type=int, default=719)
    args = ap.parse_args()

    if not os.path.exists(KAGG):
        print(f"rust binary missing -- build it first:\n"
              f"  cd rustengine && cargo build --release\n({KAGG})")
        return 1

    ok = bad = 0
    print(f"scripted chaos: {args.episodes} episode(s), "
          f"{args.steps} steps each")
    for ep in range(args.episodes):
        seed = 1_000_003 * (ep + 1) + 7
        rng = random.Random(9000 + ep)
        pairs = [(scripted_actions(rng), scripted_actions(rng))
                 for _ in range(args.steps)]
        if compare(f"chaos_{ep}", seed, pairs):
            ok += 1
        else:
            bad += 1

    # Only replays recorded on the ENGINE VERSION we vendor may enter the
    # parity comparison: a version-mixed set false-alarms after every ladder
    # rebalance (a 1.32.6 replay can never reproduce on 1.32.7 -- the market
    # rules differ -- and that is a fact about the ladder's history, not a
    # bug in either engine). The vendored version is the source of truth.
    # models/engine_version.json is the single source of truth for which
    # engine we are running (the vendored package reports "dev" -- it is not
    # pip-installed). Written only by scripts/engine_swap_1327.py.
    vendor_ver = "1.32.6"
    try:
        with open(os.path.join(ROOT, "models", "engine_version.json"),
                  encoding="utf-8") as fh:
            vendor_ver = json.load(fh).get("engine", vendor_ver)
    except (OSError, ValueError):
        pass
    candidates = sorted(glob.glob(os.path.join(
        ROOT, "data", "sameday", "_stage", "*", "*.json")))
    replays = []
    for path in candidates:
        if len(replays) >= args.replays:
            break
        if vendor_ver:
            try:
                head = open(path, encoding="utf-8",
                            errors="ignore").read(4000)
                if f'"module_version": "{vendor_ver}"' not in head \
                        and f'"module_version":"{vendor_ver}"' not in head:
                    continue
            except OSError:
                continue
        replays.append(path)
    if replays:
        print(f"real replays: {len(replays)} (engine {vendor_ver})")
    elif candidates and vendor_ver:
        print(f"real replays: none on vendored engine {vendor_ver} yet "
              f"(ladder transition window) -- scripted chaos only")
    for path in replays:
        seed, pairs = replay_pairs(path)
        if seed is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if compare(f"replay_{name}", seed, pairs):
            ok += 1
        else:
            bad += 1

    print(f"\n{ok} episode(s) bit-identical, {bad} diverged")
    if bad == 0 and ok > 0:
        print("all rust-engine differential checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
