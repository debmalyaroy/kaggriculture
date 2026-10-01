"""Truncated Python references of the v61.1 agent (task P2.4).

The WHOLE agent file is exec'd (so every late-bound helper and every later monkey-patch --
e.g. `_shadow_terminal` at 1470, `_v219_request` at 6095 -- is the final version the real agent
runs), then the callable captured as a wrapper's parent is returned: the chain up to and
including the layer just before that wrapper. A line cut would be wrong: `_V219` calls
`_r53_labor_assignment` etc., defined thousands of lines later.

Each `load_cut` call returns a fresh module (fresh per-player state).
"""
import collections, contextlib, io, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "agents", "v61.1_bandit.py")

# cut name -> the global holding the chain THROUGH that layer (the next wrapper's parent).
CUTS = {
    "chassis": "_PRE_TERMINAL_AGENT",   # make_agent + _SHOP terminal rescue
    "terminal": "_PRE_ROOM_AGENT",      # + 7-turn terminal closure planner (712-718)
    "room": "_V28_CORE",                # + hour-23 room guard
    "v28": "_V219_PARENT",              # + v28 entry guard
    "v219": "_EXPERIMENT_PARENT",       # + V219 tomato investment
    "experiment": "_ORDER_PARENT",      # + APPLY_TIMING wrapper (off)
    "order": "_V31_CORE",               # + v224 sales-first ordering
    "v31": "_V231_PARENT",              # + v31 entry guard
    "v231": "_R36_SALE_PARENT",         # + V231 cattle substitution
    "r36": "_R37_PARENT",               # + R36 sale reserve (RACEGATE) + sales-first
    "r37": "_RELEASE_PARENT",           # + R37 horizons / R44 probe / quote reorder
    "release": "_V233_PARENT",          # + RELEASE export guard
    "v233": "_R46_SHEEP_AGENT",         # + V233 six-sheep expansion (VE/SL/VT final)
    "r46": "_R51_INPUT_PARENT",         # + R46 sheep-overlay guard
    "r51_input": "_R51_WAREHOUSE_PARENT",  # + R51 fertilizer tour planner
    "r51_warehouse": "_R53_LABOR_PARENT",  # + R51 hour-23 warehouse close
    "r53": "_R70_PARENT",               # + R53 telemetry wrapper
    "r70": "_R85_PARENT",               # + R70 telemetry wrapper
    "r85": "_R95_PARENT",               # + R85 feed skip / fert sale / 2nd warehouse
    "r95": "_R97_PARENT",               # + R95 wheat-buy trim
    "r97": "_V9_COURIER_PARENT",        # + R97 wheat-supply guard
    "courier": "_V9_CARROT_PARENT",
    "carrot": "_V9_HERD_PARENT",
    "herd": "_V9_FERT_PARENT",
    "fert": "_V9_OPENING_PARENT",
    "opening": "_V9_RACE_PARENT",
    "race": "_V9_RACEPX_PARENT",
    "racepx": "_V9_RACEGATE_PARENT",
    "racegate": "_P_ctrtable_336053",
    "ctrtable": "_P_overflow_338343",
    "overflow": "_CA_PARENT",
    "ca": "_OR2_PARENT",
    "or2": "_CH_PARENT",
    "ch": "_SR_PARENT",
    "sr": "_HD2_PARENT",
    "hd2": "_CS_PARENT",
    "cs": "_RACE_PARENT",
    "race": "_R127_PARENT",
    "r127": "_PG_HOST",
    "pg": "_V44Y_HOST",
    "v44y": "_Y_HOST",
    "y": "_E334_BASE",
    "e335": "_V11_ENTRY",
    "v11": "_V13V_PARENT",
    "v13v": "_E343_WL_BASE",
    "e343_wl": "_ADV_PARENT",
    "adv": "_T62A_PARENT",
    "t62a": "_PIPE_PARENT",
    "pipe": "_MA_INNER",
    "ma": "_WB3_INNER",
    "wb3": "_FX_PARENT",
    "fx": "_DP_PARENT",
    "dp": "_MP_PARENT",
    "mp": "_BD_PARENT",
    "bd": "_MPX_PARENT",
    "mpx": "_SM_PARENT",
    "sm": "_CXD_HOST",
    "cxd": "_E410_PARENT",
    "e410": "_E402_PARENT",
    "e402": "_MG_PARENT",
    "mg": "_IG_PARENT",
    "ig": "_RSA_PARENT",
    "rsa": "_kaggle_entry",
    "full": "agent",
}

_CODE = {}


def load_entry(name, src=SRC):
    if src not in _CODE:
        with open(src, encoding="utf-8") as fh:
            _CODE[src] = compile(fh.read(), src, "exec")
    ns = {"__name__": "v611_ref"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_CODE[src], ns)
    # Telemetry dicts are initialised by OUTER wrappers (e.g. _R70_REPORT by _r70_before). A
    # truncated chain never runs them, so a `+= 1` on a missing key raises and an entry guard
    # returns PASS -- behaviour the full agent never has. Default every report to 0.
    for k, v in list(ns.items()):
        if isinstance(v, dict) and not isinstance(v, collections.defaultdict) and (k.endswith("_REPORT") or k.endswith("_STATS")
                                                                                  or k.endswith("_report") or k.endswith("_stats")):
            ns[k] = collections.defaultdict(int, v)
    return ns[name]


def load_cut(name):
    return load_entry(CUTS.get(name, name))
