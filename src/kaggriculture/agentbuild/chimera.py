"""Compose a prefix-mimicry chimera: another family's opening, our endgame.

Identity is read from the opening (identifiers commit around day 8-10), so
a chimera that plays family X's first K turns and then diverges baits any
opponent whose counter is keyed to X -- a false commit costs the committer
real margin (measured from our own side: -$1.4k..-2.9k each).

This composes the raw route; the suffix MUST then be fine-tuned against the
prefix's actual farm state (that is what dodges the splicing trap):

    python -m kaggriculture.agentbuild.chimera --donor <route_id> --base <route_id> --k 192 \
        --out models/v22/chimera/route.json
    python -m kaggriculture.train.train_arms --arm chimera_fix \
        --base models/v22/chimera/route.json --targets <tapes...>

Ship gate: the referee tournament prices the mimicry cost; deploy only when
the bait value (opponents measurably reacting to identity) exceeds it.
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donor", required=True,
                    help="route id whose opening we impersonate")
    ap.add_argument("--base", required=True,
                    help="route id whose mid/endgame we keep")
    ap.add_argument("--k", type=int, default=192,
                    help="prefix length in turns (identification closes ~192)")
    ap.add_argument("--out", default=os.path.join("models", "v22", "chimera",
                                                  "route.json"))
    args = ap.parse_args()

    donor = R.load_route(args.donor)
    base = R.load_route(args.base)
    k = max(1, min(args.k, 719))
    composite = ([copy.deepcopy(t) for t in donor[:k]]
                 + [copy.deepcopy(t) for t in base[k:720]])

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(composite, open(out, "w", encoding="utf-8"),
              separators=(",", ":"))
    print(f"chimera: {args.donor}[0:{k}] + {args.base}[{k}:720] -> "
          f"{os.path.relpath(out, ROOT)}")
    print("REQUIRED next step (the suffix assumes the wrong farm until "
          "fine-tuned):")
    print(f"  python -m kaggriculture.train.train_arms --arm chimera_fix --base "
          f"{os.path.relpath(out, ROOT)} --targets <referee tapes> "
          f"--gens 10 --pop 12")
    print("Then verify identity capture: the identifier must classify the "
          "chimera's first 192 turns as the donor's class "
          "(src/kaggriculture/train/self_identify.py logic on the composite).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
