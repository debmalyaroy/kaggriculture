"""Broad behavioral features for a route: the signature clustering runs on.

A program's identity is more than its sell curve -- it is what it builds,
when it hires, when it buys land, and how it ends the season. Blocks are
individually normalized so no channel dominates by scale.

    import kaggriculture.data.features as features
    vec, blocks = features.route_features(actions)
    pre = features.prefix_features(actions, t=240)   # identifier training
"""
import sys

PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
            "WHEAT", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("COW", "SHEEP", "GOOSE")
SAMPLE = 30                     # curve sample stride, turns
_LABOR_SAMPLES = 24
WINDOWS = ((0, 144), (144, 288), (288, 432), (432, 576), (576, 720))


def _orders(turn):
    for o in (turn.get("market") or []):
        if isinstance(o, list) and len(o) >= 2:
            yield o


def _unit_ops(turn):
    for op in [turn.get("farmer")] + list(turn.get("hands") or []):
        if isinstance(op, list) and op:
            yield op


def route_features(actions, upto=720):
    """(flat_vector, named_blocks) over the first `upto` turns."""
    n = min(upto, len(actions), 720)
    sell_cum = {p: 0 for p in PRODUCTS}
    buy_cum = {p: 0 for p in PRODUCTS}
    sell_curve = {p: [] for p in PRODUCTS}
    hire_curve = []
    hires = 0
    animals = {a: 0 for a in ANIMALS}
    plants = {c: [0] * len(WINDOWS) for c in CROPS}
    builds = {"BUILD_PASTURE": 0, "BUILD_COOP": 0}
    land_turns = []
    endgame_sold = 0
    total_sold = 0

    for t in range(n):
        turn = actions[t]
        for o in _orders(turn):
            op = o[0]
            if op == "SELL" and len(o) >= 3 and o[1] in PRODUCTS:
                q = max(0, int(o[2] or 0))
                sell_cum[o[1]] += q
                total_sold += q
                if t >= 648:
                    endgame_sold += q
            elif op in ("BUY_PRODUCT", "BUY_SEED") and len(o) >= 3 and o[1] in PRODUCTS:
                buy_cum[o[1]] += max(0, int(o[2] or 0))
            elif op == "BUY_ANIMAL" and len(o) >= 2 and o[1] in animals:
                animals[o[1]] += int(o[2]) if len(o) >= 3 else 1
            elif op == "HIRE":
                hires += 1
            elif op == "BUY_LAND":
                land_turns.append(t)
        for u in _unit_ops(turn):
            if u[0] == "PLANT" and len(u) >= 2 and u[1] in plants:
                for wi, (lo, hi) in enumerate(WINDOWS):
                    if lo <= t < hi:
                        plants[u[1]][wi] += 1
                        break
            elif u[0] in builds:
                builds[u[0]] += 1
        if t % SAMPLE == 0:
            for p in PRODUCTS:
                sell_curve[p].append(sell_cum[p])
            hire_curve.append(hires)

    def norm(vals):
        top = max(max(vals), 1)
        return [v / top for v in vals]

    blocks = {
        "sell_curves": norm([v for p in PRODUCTS for v in sell_curve[p]]),
        "buy_totals": norm([buy_cum[p] for p in PRODUCTS]),
        "plantings": norm([plants[c][w] for c in CROPS
                           for w in range(len(WINDOWS))]),
        "herd": norm([animals[a] for a in ANIMALS]),
        "labor": norm(hire_curve),
        "builds_land": norm([builds["BUILD_PASTURE"], builds["BUILD_COOP"],
                             len(land_turns),
                             (land_turns[0] if land_turns else 720),
                             (land_turns[-1] if land_turns else 720)]),
        "endgame": [endgame_sold / max(1, total_sold)],
    }
    flat = [v for b in ("sell_curves", "buy_totals", "plantings", "herd",
                        "labor", "builds_land", "endgame")
            for v in blocks[b]]
    return flat, blocks


def prefix_features(actions, t):
    """What an in-game observer could have accumulated by turn t.

    Fixed-dimensional regardless of t: the 24 canonical curve samples are
    zero-padded beyond the observation point (the future is unknown, not
    carried forward), so rows from different turns stack into one matrix.
    Used to manufacture identifier training rows; callers may add dropout
    noise to mimic floored-sale invisibility."""
    _, blocks = route_features(actions, upto=t)
    per = 24
    n_seen = min(per, t // SAMPLE + 1)
    curves = blocks["sell_curves"]
    seen_per = len(curves) // len(PRODUCTS) if curves else 0
    out = []
    for p in range(len(PRODUCTS)):
        seg = curves[p * seen_per:(p + 1) * seen_per][:n_seen]
        out.extend(seg + [0.0] * (per - len(seg)))
    # Strictly in-game-observable blocks only: an opponent's BUYs are not
    # reconstructible from public inventory (they conflate with consumption),
    # so the identifier must never train on them. Farm-EVENT blocks are
    # included (v3): plants, herd adds, hires and land are all observable
    # live as opponent farm-state diffs, and -- unlike sale curves -- cannot
    # be obfuscated without actually farming differently.
    labor = blocks["labor"][:_LABOR_SAMPLES]
    labor = labor + [0.0] * (_LABOR_SAMPLES - len(labor))
    return (out + blocks["endgame"] + blocks["plantings"] + blocks["herd"]
            + labor + blocks["builds_land"])


def distance(a, b):
    """L1 over the common prefix of two flat vectors."""
    m = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(m)) / max(1, m)


if __name__ == "__main__":
    import kaggriculture.data.routes as R
    rid = sys.argv[1] if len(sys.argv) > 1 else "91468035_s1"
    flat, blocks = route_features(R.load_route(rid))
    print(f"{rid}: {len(flat)} dims")
    for k, v in blocks.items():
        print(f"  {k:<12} {len(v):>4} dims  head {[round(x, 2) for x in v[:5]]}")
