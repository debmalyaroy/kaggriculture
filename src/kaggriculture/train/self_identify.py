"""How identifiable are WE? The acceptance gate for signature obfuscation.

Runs our own identifier against our own live games (our seat's sales, as an
opponent would reconstruct them) and reports how often we are confidently
classified. Today's number is the baseline; any obfuscation layer ships
only if it drives this to chance while a paired battery shows zero cost.

    python -m kaggriculture.train.self_identify --submission 55396717
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

import kaggriculture.train.train_identifier as TI  # noqa: E402

IDW = os.path.join(ROOT, "models", "v22", "identifier", "weights.json")
PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
            "CARROT", "WHEAT", "FERTILIZER")


def game_vector(g):
    """Final-totals approximation of the prefix features at t=720: the
    curve samples are unknown from the index, so fill the last sample only
    -- a lower bound on identifiability (curves make us MORE identifiable,
    not less)."""
    sold = {p: int((g.get("sold") or {}).get(p, 0) or 0) for p in PRODUCTS}
    top = max(max(sold.values()), 1)
    vec = []
    for p in PRODUCTS:
        vec.extend([0.0] * 23 + [sold[p] / top])
    vec.append(0.0)                       # endgame share unknown from index
    vec.append(719 / 720.0)
    return vec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", default=None,
                    help="default: every submission in the ourgames index")
    args = ap.parse_args()

    idw = json.load(open(IDW, encoding="utf-8"))
    games = json.load(open(os.path.join(ROOT, "data", "ourgames",
                                        "index.json"), encoding="utf-8"))["games"]
    from collections import Counter, defaultdict
    per_sub = defaultdict(Counter)
    conf = defaultdict(list)
    for g in games.values():
        sub = str(g.get("submission"))
        if args.submission and sub != args.submission:
            continue
        if sum(int((g.get("sold") or {}).get(p, 0) or 0) for p in PRODUCTS) < 50:
            continue
        p = TI.stdlib_predict(idw, game_vector(g))
        cls = max(range(len(p)), key=lambda i: p[i])
        per_sub[sub][cls] += 1
        conf[sub].append(p[cls])
    for sub in sorted(per_sub):
        c = per_sub[sub]
        total = sum(c.values())
        top_cls, top_n = c.most_common(1)[0]
        mean_conf = sum(conf[sub]) / len(conf[sub])
        print(f"submission {sub}: {total} games -> class {top_cls} in "
              f"{100 * top_n / total:.0f}% of them (mean confidence "
              f"{mean_conf:.2f}) -- {'IDENTIFIABLE' if top_n / total > 0.5 else 'mixed'}")
    print("\nObfuscation acceptance: the dominant-class share must fall to "
          "chance with zero paired-battery cost before any layer ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
