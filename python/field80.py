"""The 80-agent gate field (operator 29 Sep): the downloaded public agents + lineage variants of them (a few strategy
constants perturbed, so each is a NEW but plausible member of its lineage) + our own previous agents.

    python python/field80.py [--variants 26] [--seed 29] [--out .local/field80]

Copies (never edits) the root repo's data/winplan/{field,agents} into OUT (workspace rule: copy from root, never
share). Variant k of agent A: A's main.py with 3 of its top-level numeric strategy constants (V9_*, _V92_P_*,
CROP_MIN_PRICE, _V231_CAP, herd / carrot / fertilizer thresholds; never LAST_ACT_STEP / MAX_ORDERS / engine
constants) moved by +-15..30% (ints rounded, kept >= 1 when they were >= 1). Writes OUT/field/*.py wrappers
(one per agent) and OUT/manifest.json {name: {path, base, tweaks}}.
"""
import argparse
import json
import os
import random
import re
import shutil

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = RL  # one repo since the 2026-10-01 merge
SRC_FIELD = os.path.join(ROOT, "data", "winplan", "field")
SRC_AGENTS = os.path.join(ROOT, "data", "winplan", "agents")
CFG = os.path.join(ROOT, "data", "winplan", "ladder_config.json")
CONST = re.compile(r"^(_?[A-Z][A-Z0-9_]{2,})( *= *)(-?[0-9][0-9.]*)(\s*(#.*)?)$", re.M)
TUNABLE = re.compile(r"^(V9_|_V92_P_|CROP_MIN_PRICE|_V231_CAP|_R37_(?!PRICE_FLOOR|HINGE)|RSA_|RACE_|CA_|_CA_|TSELL_|ADV_|OR2_)")
FORBID = {"LAST_ACT_STEP", "MAX_ORDERS", "_RELEASE_ERRORS", "_R51_INPUT_MAX_WORKERS", "_V92_P_EVERY"}

WRAP = """import sys, os, json, importlib.util
_d = {d!r}
if _d not in sys.path: sys.path.insert(0, _d)
_s = importlib.util.spec_from_file_location({mod!r}, os.path.join(_d, 'main.py'))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
_CFG = json.load(open({cfg!r}))
# Kaggle runs the module's LAST callable (insertion order), not `agent`
_inner = [v for v in vars(_m).values() if callable(v)][-1]
_argc = _inner.__code__.co_argcount if hasattr(_inner, '__code__') else 2
def agent(obs, config=None):
    return _inner(obs, config if config is not None else _CFG) if _argc > 1 else _inner(obs)
"""


def agent_dir(wrapper):
    m = re.search(r"_d = '([^']+)'", open(wrapper, encoding="utf-8").read())
    return m.group(1).replace("\\\\", "\\") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=26)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", default=os.path.join(RL, ".local", "field80"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "field"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "agents"), exist_ok=True)
    cfg = os.path.join(a.out, "ladder_config.json")
    shutil.copy(CFG, cfg)
    man = {}
    bases = []
    for w in sorted(os.listdir(SRC_FIELD)):
        if not w.endswith(".py"):
            continue
        name = w[:-3]
        d = agent_dir(os.path.join(SRC_FIELD, w))
        if not d or not os.path.isdir(d):
            print(f"[field80] skip {name}: no agent dir")
            continue
        dst = os.path.join(a.out, "agents", name)
        if not os.path.isdir(dst):
            shutil.copytree(d, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        open(os.path.join(a.out, "field", w), "w", encoding="utf-8").write(WRAP.format(d=dst, mod=f"f80_{name}", cfg=cfg))
        man[name] = {"path": os.path.join(a.out, "field", w), "base": name, "tweaks": {}}
        src = open(os.path.join(dst, "main.py"), encoding="utf-8", errors="replace").read()
        cands = [m for m in CONST.finditer(src) if TUNABLE.match(m.group(1)) and m.group(1) not in FORBID]
        if len(cands) >= 3:
            bases.append((name, dst, cands))
    rng.shuffle(bases)
    for name, dst, cands in bases[: a.variants]:
        vname = f"{name}__t{a.seed}"
        vdst = os.path.join(a.out, "agents", vname)
        if os.path.isdir(vdst):
            shutil.rmtree(vdst)
        shutil.copytree(dst, vdst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        src = open(os.path.join(vdst, "main.py"), encoding="utf-8", errors="replace").read()
        tweaks = {}
        for m in rng.sample(cands, 3):
            old = m.group(3)
            v = float(old)
            f = rng.choice([-1, 1]) * rng.uniform(0.15, 0.30)
            nv = v * (1 + f)
            if "." not in old:
                nv = int(round(nv))
                if int(old) >= 1:
                    nv = max(1, nv)
                if nv == int(old):
                    nv = int(old) + (1 if f > 0 else -1)
                new = str(nv)
            else:
                new = f"{nv:.4g}"
            line_old = m.group(0)
            line_new = f"{m.group(1)}{m.group(2)}{new}{m.group(4)}"
            src = src.replace(line_old, line_new, 1)
            tweaks[m.group(1)] = [old, new]
        open(os.path.join(vdst, "main.py"), "w", encoding="utf-8").write(src)
        open(os.path.join(a.out, "field", vname + ".py"), "w", encoding="utf-8").write(WRAP.format(d=vdst, mod=f"f80_{vname}", cfg=cfg))
        man[vname] = {"path": os.path.join(a.out, "field", vname + ".py"), "base": name, "tweaks": tweaks}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    nv = sum(1 for v in man.values() if v["tweaks"])
    print(f"[field80] {len(man) - nv} public agents + {nv} lineage variants -> {a.out}")
    for k, v in man.items():
        if v["tweaks"]:
            print(f"  {k}: {v['tweaks']}")


if __name__ == "__main__":
    main()
