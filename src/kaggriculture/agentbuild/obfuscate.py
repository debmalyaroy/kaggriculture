"""Signature-obfuscation analysis and acceptance gate, measured not asserted.

The premise (src/self_identify.py) is that we are 100% identifiable and a
competitor -- the public first-48-turn detector (Revanth Tambisetty,
2026-08-09) is a live example -- can key a counter to us. This tool measures
how much of that identity a runtime layer could actually remove, and at what
cost, before anything ships.

Three identity channels, in order of how portable they are across detectors:

  1. FARM-EVENT opening   what we HIRE / BUY_ANIMAL / BUY_SEED / BUILD in the
                          first 48 turns. This is the channel Revanth's public
                          detector uses and the dominant one in our own
                          identifier. It cannot change without farming
                          differently -- so obfuscating it is not free.
  2. SELL-MIX             the final basket of products we sell. A fixed crop
                          plan produces a fixed basket, so this too is a
                          consequence of how we farm.
  3. SELL-CURVE TIMING    WHEN within the season we sell. The one channel a
                          post-hoc order reshuffle can touch -- but the
                          identifier samples the curve every 30 turns and keys
                          mostly on farm events, so timing jitter barely moves
                          it, and provably cannot move channels 1-2 at all.

For each candidate transform we recompute our own identifier's confidence on
the perturbed route (offline identity axis) and, for survivors that actually
reduce confidence, price the transform with a real engine self-play battery
(cost axis). A layer ships only if it drives identity toward chance AND the
paired battery shows zero cost.

    python src/obfuscate.py                 # full report, no engine
    python src/obfuscate.py --cost          # + engine cost battery on survivors
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.data.features as F  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402

IDW_PATH = os.path.join(ROOT, "models", "v22", "identifier", "weights.json")
BASE_AGENT = os.path.join(ROOT, "agents", "v19_2_route.py")
PROBE_T = 192               # identification is effectively closed by here
PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
            "WHEAT", "FERTILIZER")


def load_base_route():
    spec = importlib.util.spec_from_file_location("_base", BASE_AGENT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return [copy.deepcopy(t) for t in m._ROUTE]


def identity(route, idw):
    """(class, confidence) our own identifier assigns to a route's prefix."""
    vec = F.prefix_features(route, t=PROBE_T)
    p = TI.stdlib_predict(idw, vec)
    cls = max(range(len(p)), key=lambda i: p[i])
    return cls, p[cls], p


# --------------------------------------------------------------- transforms

def t_sell_jitter(route, delay):
    """Channel 3: hold every SELL `delay` turns (crossing 30-turn sample bins
    when delay>=30). Revenue-neutral in units; only timing moves."""
    out = [copy.deepcopy(t) for t in route]
    carried = [[] for _ in range(len(out) + delay + 1)]
    for t in range(len(out)):
        keep = []
        for o in (out[t].get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and t < 640):
                carried[min(t + delay, len(carried) - 1)].append(list(o))
            else:
                keep.append(o)
        out[t]["market"] = carried[t] + keep
    return out


def _open_edit(route, edits):
    """Apply count edits to opening BUY orders (turns < 48). `edits` maps a
    (op, item) pair to a signed delta; negatives remove/reduce, positives add
    at turn 0."""
    out = [copy.deepcopy(t) for t in route]
    remaining = dict(edits)
    for t in range(48):
        market = out[t].get("market") or []
        for o in market:
            if not (isinstance(o, list) and len(o) >= 2):
                continue
            key = (o[0], o[1] if len(o) > 1 else None)
            if key in remaining and remaining[key] < 0 and len(o) >= 3:
                take = min(int(o[2] or 0), -remaining[key])
                o[2] = int(o[2]) - take
                remaining[key] += take
    add = [list(k) + [d] for k, d in remaining.items() if d > 0]
    if add:
        out[0]["market"] = add + (out[0].get("market") or [])
    return out


def t_animal_swap(route):
    """Channel 1: shift the herd 1 cow -> 1 sheep (stay sheep-first basin,
    change the herd fingerprint)."""
    return _open_edit(route, {("BUY_ANIMAL", "COW"): -1,
                              ("BUY_ANIMAL", "SHEEP"): +1})


def t_seed_shuffle(route):
    """Channel 1: move 2 melon seeds into wheat (same basin, different seed
    fingerprint from Revanth's 5/5 centroid)."""
    return _open_edit(route, {("BUY_SEED", "MELON"): -2,
                              ("BUY_SEED", "WHEAT"): +2})


def t_sellmix_shift(route):
    """Channel 2: reassign 20% of WHEAT sell units to FERTILIZER across the
    season (breaks the final-basket signature; needs the shed to hold it, so
    almost certainly costs -- that is the point of measuring)."""
    out = [copy.deepcopy(t) for t in route]
    for t in range(len(out)):
        for o in (out[t].get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] == "WHEAT"):
                q = int(o[2] or 0)
                move = q // 5
                if move:
                    o[2] = q - move
                    out[t]["market"].append(["SELL", "FERTILIZER", move])
    return out


TRANSFORMS = [
    ("sell_jitter+2", lambda r: t_sell_jitter(r, 2)),      # the shipped jitter
    ("sell_jitter+30", lambda r: t_sell_jitter(r, 30)),    # bin-crossing
    ("sell_jitter+60", lambda r: t_sell_jitter(r, 60)),
    ("animal_swap", t_animal_swap),
    ("seed_shuffle", t_seed_shuffle),
    ("sellmix_shift", t_sellmix_shift),
]


# ------------------------------------------------------------------- checks

def jitter_invariance(route):
    """Data-level proof that SELL-timing jitter cannot touch the SELL-MIX
    (channel 2): the summed sold units per product are byte-identical before
    and after. If this holds, no timing layer can ever move self_identify,
    which reads exactly those totals."""
    def totals(rt):
        s = {p: 0 for p in PRODUCTS}
        for t in rt:
            for o in (t.get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in s):
                    s[o[1]] += max(0, int(o[2] or 0))
        return s
    return totals(route) == totals(t_sell_jitter(route, 30))


# --------------------------------------------------------------------- cost

def cost_battery(base_route, cand_route, seeds, opponents):
    """Engine self-play cost: mean (candidate - base) final bank over a fixed
    panel, both seats. Requires the vendored engine."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    scratch = os.path.join(ROOT, ".local", "obf")
    os.makedirs(scratch, exist_ok=True)
    base_f = os.path.join(scratch, "_base.py")
    cand_f = os.path.join(scratch, "_cand.py")
    _emit_agent(base_route, base_f)
    _emit_agent(cand_route, cand_f)

    def play(a, b, seed):
        env = make("kaggriculture", configuration={
            "episodeSteps": 720, "seed": seed, "actTimeout": 60,
            "runTimeout": 100000})
        env.run([a, b])
        f = env.steps[-1]
        return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)

    dbank = []
    for opp in opponents:
        for s in seeds:
            b0, _ = play(base_f, opp, s)
            c0, _ = play(cand_f, opp, s)
            _, b1 = play(opp, base_f, s)
            _, c1 = play(opp, cand_f, s)
            dbank.append(((c0 - b0) + (c1 - b1)) / 2)
    return sum(dbank) / len(dbank), dbank


def _emit_agent(route, path):
    """Minimal open-loop route agent with positional hand alignment, enough
    to price a route in self-play without importing the full runtime."""
    import base64
    import zlib
    payload = base64.b85encode(zlib.compress(
        json.dumps(route, separators=(",", ":")).encode())).decode()
    src = '''import base64, copy, json, zlib
_R = json.loads(zlib.decompress(base64.b85decode("%s".encode())))
def _get(o, k, d=None):
    v = o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
    return d if v is None else v
def agent(obs, config=None):
    step = int(_get(obs, "step", 0) or 0)
    a = copy.deepcopy(_R[min(max(0, step), len(_R) - 1)])
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    hands = _get(farms[me], "hands", []) if me < len(farms) else []
    ah = a.get("hands") or []
    a["hands"] = [ah[i] if i < len(ah) else ["PASS"] for i in range(len(hands))]
    return a
''' % payload
    open(path, "w", encoding="utf-8").write(src)


# ------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cost", action="store_true",
                    help="price identity-reducing survivors with the engine")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    idw = json.load(open(IDW_PATH, encoding="utf-8"))
    base = load_base_route()
    b_cls, b_conf, _ = identity(base, idw)
    n_cls = len(json.load(open(IDW_PATH))["b"])
    chance = 1.0 / n_cls

    # channel-1 exposure straight out of the broad dataset
    fp = os.path.join(ROOT, "data", "fingerprints", "summary.json")
    sat = None
    if os.path.exists(fp):
        w = json.load(open(fp, encoding="utf-8"))["where_we_sit"]
        sat = w.get("v19_2_route", {})

    print("=" * 68)
    print("OBFUSCATION EXPOSURE")
    print("=" * 68)
    print(f"identifier: base route -> class {b_cls} @ {b_conf:.3f} "
          f"confidence (chance = {chance:.3f}, {n_cls} classes)")
    if sat:
        print(f"opening cluster: {sat.get('public_label')} -- "
              f"{sat.get('saturation_note')}")
    print(f"sell-mix (self_identify): 100% identifiable (see self_identify.py)")
    inv = jitter_invariance(base)
    print(f"jitter-invariance proof: SELL-timing jitter preserves the sell "
          f"basket -> {inv}  (=> timing can NEVER move self_identify)")

    print("\n" + "=" * 68)
    print("TRANSFORM SEARCH  (identity axis, offline)")
    print("=" * 68)
    print(f"{'transform':<16} {'class':>5} {'conf':>7} {'drop':>7}  channel")
    survivors = []
    chan = {"sell_jitter+2": 3, "sell_jitter+30": 3, "sell_jitter+60": 3,
            "animal_swap": 1, "seed_shuffle": 1, "sellmix_shift": 2}
    results = []
    for name, fn in TRANSFORMS:
        cand = fn(base)
        c_cls, c_conf, _ = identity(cand, idw)
        drop = b_conf - c_conf if c_cls == b_cls else b_conf  # class flip = big
        results.append({"transform": name, "class": c_cls,
                        "conf": round(c_conf, 3), "drop": round(drop, 3),
                        "channel": chan[name], "route": cand})
        flag = "  <- reduces identity" if drop > 0.2 else ""
        print(f"{name:<16} {c_cls:>5} {c_conf:>7.3f} {drop:>7.3f}  "
              f"ch{chan[name]}{flag}")
        if drop > 0.2:
            survivors.append(results[-1])

    verdict = {"base_class": b_cls, "base_conf": round(b_conf, 3),
               "chance": round(chance, 3),
               "jitter_preserves_mix": inv,
               "opening_saturation": sat,
               "transforms": [{k: v for k, v in r.items() if k != "route"}
                              for r in results]}

    if args.cost:
        print("\n" + "=" * 68)
        print("COST BATTERY  (engine self-play, every transform)")
        print("=" * 68)
        print(f"{'transform':<16} {'bank delta':>13} {'identity':>9}  verdict")
        seeds = list(range(1000, 1000 + args.seeds))
        panel = [os.path.join(ROOT, "agents", "v18_route.py"), "pass"]
        cost = {}
        for r in results:
            mean_d, _ = cost_battery(base, r["route"], seeds, panel)
            free = abs(mean_d) < 3000
            reduces = r["drop"] > 0.2
            v = ("SHIP" if free and reduces else
                 "no-benefit" if free else "COSTS")
            print(f"{r['transform']:<16} {mean_d:>+13,.0f} "
                  f"{'-'+format(r['drop'],'.2f') if r['drop']>0 else '  0':>9}  {v}")
            cost[r["transform"]] = round(mean_d)
        verdict["cost"] = cost

    out = os.path.join(ROOT, "models", "v22", "obfuscation_report.json")
    json.dump(verdict, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
