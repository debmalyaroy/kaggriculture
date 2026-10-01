"""Inject the LEARNED day-6 dispatcher into a built agent (2026-09-07).

Post-build transform: embeds (1) the pure-python feature extractor, (2) the
trained decision tree, (3) the candidate day-6 continuations (identical to
those the tree was trained on), and injects a day-6 (step 144) block that —
when the `tree_dispatch` flag is ON and no owned route has already committed
— extracts features, walks the tree, and splices the selected continuation
(prefix-compatible: route[:144] + cont). This is the winning-meta lever:
LEARNED selection over the portfolio from REAL adaptive outcomes, robust
where sim_search (fixed-opponent rollout) was fragile.

    python src/trackp/harness/build_tree_dispatch.py --in v48_bandit.py \
        --out v49_bandit.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import ast
import json
import os
import re
import sys

HARN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARN)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import dispatcher_train_data as DTD                            # noqa: E402


def features_src():
    """The dispatcher_features module body, renamed for inlining (no import)."""
    src = open(os.path.join(HARN, "dispatcher_features.py"),
               encoding="utf-8").read()
    # strip the module docstring + __future__; keep constants + functions,
    # prefix names with _DSP_ to avoid clashes.
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.DOTALL)
    src = src.replace("from __future__ import annotations", "")
    for name in ("ITEMS", "SHOPS", "SHOP_DEMAND", "ANIMAL_PROD", "FEATN",
                 "_board", "features"):
        src = re.sub(r"\b" + name + r"\b", "_DSP_" + name, src)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default=os.path.join(
        ROOT, "models", "trackp", "dispatcher", "tree_144.json"))
    a = ap.parse_args()
    src = open(a.inp, encoding="utf-8").read()
    tree = json.load(open(a.tree, encoding="utf-8"))["tree"]
    base = DTD.base_route()
    cands = DTD.candidates(base)            # [(name, cont[144:719]), ...]
    conts = [c[1] for c in cands]
    names = [c[0] for c in cands]

    import base64
    import zlib
    blob = base64.b85encode(zlib.compress(
        json.dumps(conts, separators=(",", ":")).encode(), 9)).decode("ascii")
    block = (
        "\n\n# ===== LEARNED DAY-6 DISPATCHER (2026-09-07) =====\n"
        + features_src()
        + "\n_DSP_TREE = " + json.dumps(tree) + "\n"
        + "_DSP_CONTS = json.loads(zlib.decompress(base64.b85decode("
        + repr(blob) + ")).decode('utf-8'))\n"
        + "_DSP_NAMES = " + json.dumps(names) + "\n"
        + "def _dsp_pick(_x):\n"
        "    _n = _DSP_TREE\n"
        "    while _n[0] != 'leaf':\n"
        "        _f, _thr, _l, _r = _n\n"
        "        _n = _l if _x[_f] <= _thr else _r\n"
        "    return int(_n[1])\n"
    )

    # insert the block right before the agent() function
    anchor = "def agent(obs"
    assert src.count(anchor) >= 1, "no agent() in source"
    src = src.replace(anchor, block + "\n" + anchor, 1)

    # inject the day-6 selection into the route-swap area: fire at step 144,
    # only if tree_dispatch flag ON and nothing else committed the route.
    swap_anchor = '        _route = state.get("route") or _ROUTE'
    assert src.count(swap_anchor) == 1, "route anchor not found/unique"
    inject = (
        '        if (globals().get("_FLAGS", {}).get("tree_dispatch", False)\n'
        '                and not state.get("dsp_done") and step == 144\n'
        '                and not (state.get("d3branch") or '
        'state.get("counter_on") or state.get("wtail"))):\n'
        '            state["dsp_done"] = True\n'
        '            try:\n'
        '                _xi = _DSP_features(obs)\n'
        '                _ci = _dsp_pick(_xi)\n'
        '                if 0 <= _ci < len(_DSP_CONTS):\n'
        '                    state["route"] = _ROUTE[:144] + _DSP_CONTS[_ci]\n'
        '                    state["dspbranch"] = True\n'
        '            except Exception:\n'
        '                pass\n'
    )
    src = src.replace(swap_anchor, inject + swap_anchor, 1)
    ast.parse(src)
    open(a.out, "w", encoding="utf-8").write(src)
    print(f"built {a.out} ({len(src):,} B) — tree_dispatch layer "
          f"({len(names)} candidates: {names})")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
