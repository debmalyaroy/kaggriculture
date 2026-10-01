"""Pluggable feature flags (2026-09-07).

Every strategic layer in the built agent is individually toggleable so its
contribution can be MEASURED (ablation) and disabled if it "takes us down".
The injector rewrites each `action = _LAYER(...)` pipeline call into a
flag-gated form and embeds a `_FLAGS` dict; the ablation harness flips one
flag at a time and reads the closed-loop delta.

Flags default ON (current shipped behavior) unless a config overrides. A
disabled flag makes the layer a NO-OP (action passes through unchanged),
never removed from the source — so a flip is behavior-exact reversible.

Usage:
    from feature_flags import inject
    src = inject(src, flags_dict)          # returns transformed source

The pipeline layers (bandit chassis) and their flags:
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import json
import os
import re

CONFIG = os.path.join(ROOT, "models", "trackp", "feature_flags.json")

# layer function name -> flag key. Order matches the agent() pipeline.
PIPELINE_LAYERS = {
    "_rank_sell_slots": "rank_sell_slots",
    "_terminal_flush": "terminal_flush",
    "_policy_head_retime": "policy_head_retime",
    "_tomato_read": "tomato_read",
    "_weed_repair": "weed_repair",
    "_feed_pull": "feed_pull",
    "_relay": "relay",
    "_pull_sells": "pull_sells",
    "_premium_lead": "premium_lead",
    "_adaptive_market": "adaptive_market",
    "_market_timing": "market_timing",
    "_sell_first": "sell_first",
    "_impact_slots": "impact_slots",
    "_threat_first": "threat_first",
}

# router / owned-route layers gated by state keys (handled specially in the
# swap block); recorded here for the ablation harness to know about them.
ROUTER_FLAGS = ["d3_fork", "family_counter", "d6_fork", "world_tail", "arms"]

# closed-loop layers (trackp / added this cycle)
CLOSED_LOOP_FLAGS = ["sim_search", "opp_model", "dispatcher", "tree_dispatch"]

ALL_FLAGS = (list(PIPELINE_LAYERS.values()) + ROUTER_FLAGS
             + CLOSED_LOOP_FLAGS)


def default_flags():
    return {k: True for k in ALL_FLAGS}


def load_flags(path=CONFIG):
    f = default_flags()
    if os.path.exists(path):
        try:
            f.update(json.load(open(path, encoding="utf-8")))
        except Exception:                                      # noqa: BLE001
            pass
    return f


def inject(src, flags):
    """Wrap each pipeline layer call in a flag gate and embed _FLAGS.

    Transforms lines of the exact form
        action = _LAYER(<args>)
    into
        action = _LAYER(<args>) if _FLAGS.get("<flag>", True) else action
    Idempotent: a line already carrying `if _FLAGS.get` is left alone.
    """
    # embed the _FLAGS dict once, right after the first import block.
    # repr() -> a Python dict literal (True/False), NOT json.dumps (true/false)
    # which would NameError when the agent source is exec'd.
    flag_line = "_FLAGS = " + repr({str(k): bool(v)
                                    for k, v in flags.items()}) + "\n"
    if "_FLAGS = " not in src:
        # insert after the module docstring / first import math
        m = re.search(r"^import math.*$", src, re.MULTILINE)
        if m:
            src = src[:m.end()] + "\n" + flag_line + src[m.end():]
        else:
            src = flag_line + src
    n = 0
    for fn, flag in PIPELINE_LAYERS.items():
        # match: <indent>action = _LAYER(...)  (single line, not already gated)
        pat = re.compile(
            r"^(?P<i>[ \t]*)action = (?P<call>" + re.escape(fn)
            + r"\([^\n]*\))[ \t]*$", re.MULTILINE)

        def repl(mo, flag=flag):
            nonlocal n
            if "_FLAGS.get" in mo.group(0):
                return mo.group(0)
            n += 1
            return (mo.group("i") + "action = " + mo.group("call")
                    + ' if _FLAGS.get("' + flag + '", True) else action')
        src = pat.sub(repl, src)
    return src, n


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--init":
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        json.dump(default_flags(), open(CONFIG, "w"), indent=1)
        print(f"wrote {CONFIG} with {len(ALL_FLAGS)} flags")
    else:
        print("flags:", ALL_FLAGS)
