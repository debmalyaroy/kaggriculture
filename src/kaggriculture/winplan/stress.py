"""Stress roster: perturbed clones of the strongest public opponent families.

The ladder's 2200-2800 band is full of TWEAKED copies of a handful of public
chassis agents -- a flag flipped, a horizon shifted, a sell offset nudged. The
gate field (data/winplan/field) only holds the originals, so a candidate can
be tuned to the exact published knobs and still lose to the clones. This
module builds K perturbed variants per family:

    python -m kaggriculture.winplan.stress build [--families a,b] [-k 4]
    python -m kaggriculture.winplan.stress smoke [--workers 2]
    python -m kaggriculture.winplan.stress spot  [--n 2]

* knobs: module-level UPPERCASE assignments of a bool/int/float literal (the
  LAST binding of each name -- chassis layers re-bind, the last one is live),
  read somewhere in the code, never used as a subscript, never re-bound from
  inside a function (`global`), not a game-structure constant (order cap, end
  step, board size...) and not an int >= 600 (an end-of-game step boundary).
  Knobs inside zlib+base64 `exec(compile(...))` payloads (pioneers) are
  decoded, perturbed and re-encoded.
* a variant flips N_FLIP booleans and jitters N_JITTER numerics (small ints
  +-1..3, larger ints / floats +-8..20%), from a fixed per-(family, k,
  attempt) RNG seed, and records every diff in the manifest.
* each variant is a full copy of the package dir under
  data/winplan/stress_agents/<family>__v<k>/ plus a shim in
  data/winplan/stress_field/<family>__v<k>.py (the harvest shim format:
  LAST-callable entry, ladder config to 2-arg agents). The shim also evicts
  the package's sibling modules from sys.modules around the load, so two
  variants of a multi-file package (god's mode: base_agent.py) never share a
  cached sibling inside one gate worker.
* smoke: one full Rust-serve game vs agents/v61_bandit.py per variant, each in
  a fresh subprocess; reject on crash, persistent None/malformed actions, bank
  < 20% of the original family's bank in the same world, or mean latency
  >= 50 ms, or an INERT variant (identical game to the original: every
  perturbed knob was dead in that world). Rejects are regenerated with the
  next attempt seed.

Gate the roster later with gate.roster(field_dir=stress.FIELD).
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

from kaggriculture.paths import ROOT
from kaggriculture.winplan import paths as P

AGENTS = os.path.join(P.DATA, "stress_agents")
FIELD = os.path.join(P.DATA, "stress_field")
MANIFEST = os.path.join(FIELD, "manifest.json")
OURS = os.path.join(ROOT, "agents", "v61_bandit.py")
GATE_FILE = os.path.join(P.GATES, "v61_kfaithful__f1f26242e39f.jsonl")

# weakest v61 matchups in the Kaggle-faithful gate + two historically strong
FAMILIES = [
    "kaggriculture_tetsutani_demand_preserving",
    "herd_safe_v3_experimental_risk_aware_feed",
    "the_shepherds_ledger_herd_safe_sovereign",
    "kaggriculture_v52_lean_flock_yarn_route",
    "kaggriculture_v53_opening_signature",
    "god_s_mode_hacked_stores",
    "kaggriculture_v15stack_nb",
    "pioneers_of_kaggle_town",
]
K = 4
N_FLIP = 2
N_JITTER = 4
MAX_ATTEMPTS = 12
BANK_FLOOR = 0.20          # of the original's bank in the same world
LATENCY_MS = 50.0

# game-structure / plumbing names: never perturbed
SKIP_NAME = re.compile(
    r"MAX_ORDERS|ORDERS_PER|LAST_ACT|LAST_STEP|FINAL_STEP|END_STEP|EPISODE|TURNS_PER|STEPS_PER|"
    r"HOURS_PER|DAYS?_TOTAL|N_DAYS|BOARD|GRID|SIZE|WIDTH|HEIGHT|SHED_CAP|CAPACITY|IDX|INDEX|SEAT|"
    r"PLAYER|VERSION|SEED|ERRORS|WORKERS|DEBUG|LOG|VERBOSE|TELEMETRY|TRACE|TIMEOUT|BUDGET|"
    r"DEADLINE|TIME_|_MS$|_SEC|PROFILE|ASSERT|STRICT|CACHE|EXPORT|PRINT|DUMP_FILE|PATH|FILE")
B64_EXEC = re.compile(r"b64decode\(\s*'([A-Za-z0-9+/=]{200,})'\s*\)")


# ---------------------------------------------------------------- sources
def family_dir(family):
    """Package dir of a field shim (its `_d = ...` line)."""
    text = open(os.path.join(P.FIELD, family + ".py"), encoding="utf-8").read()
    return ast.literal_eval(text.split("_d = ", 1)[1].split("\n", 1)[0].strip())


def families_from_gate(path=GATE_FILE, n=6):
    """Families sorted by wins minus losses of our agent (weakest first)."""
    from kaggriculture.winplan import gate as G
    fams = G.summarize(path)["families"]
    return sorted(fams, key=lambda o: fams[o]["w"] - fams[o]["l"])[:n]


def _units(pkg_dir):
    """[(relfile, payload_index|None, source)] -- every .py file plus each
    zlib+base64 exec payload inside it."""
    out = []
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), pkg_dir)
            src = open(os.path.join(root, f), encoding="utf-8").read()
            out.append((rel, None, src))
            for i, m in enumerate(B64_EXEC.finditer(src)):
                try:
                    out.append((rel, i, zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8")))
                except Exception:                                    # noqa: BLE001
                    pass
    return out


def _is_knob_name(nm):
    core = nm.lstrip("_")
    return bool(core) and core.upper() == core and any(c.isalpha() for c in core)


def scan_knobs(src):
    """{name: site} for the perturbable knobs of one source unit."""
    tree = ast.parse(src)
    last = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        nm = node.targets[0].id
        if not _is_knob_name(nm):
            continue
        v, neg = node.value, False
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
            v, neg = v.operand, True
        if isinstance(v, ast.Constant) and type(v.value) in (bool, int, float) and not (neg and type(v.value) is bool):
            val = -v.value if neg else v.value
            last[nm] = {"value": val, "type": type(val).__name__,
                        "pos": (node.value.lineno, node.value.col_offset,
                                node.value.end_lineno, node.value.end_col_offset)}
        else:
            last.pop(nm, None)          # re-bound to a non-literal: not a knob
    loads, subscripted, globaled = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.add(node.id)
        elif isinstance(node, ast.Subscript):
            for sub in ast.walk(node.slice):
                if isinstance(sub, ast.Name):
                    subscripted.add(sub.id)
        elif isinstance(node, ast.Global):
            globaled.update(node.names)
    out = {}
    for nm, site in last.items():
        if nm not in loads or nm in subscripted or nm in globaled or SKIP_NAME.search(nm):
            continue
        if site["type"] == "int" and abs(site["value"]) >= 600:
            continue
        if site["type"] == "float" and site["value"] == 0.0:
            continue
        out[nm] = site
    return out


def knob_pool(pkg_dir):
    """[(unit_key, name, site)] over every unit of a package."""
    pool = []
    for rel, idx, src in _units(pkg_dir):
        try:
            ks = scan_knobs(src)
        except SyntaxError:
            continue
        pool += [((rel, idx), nm, site) for nm, site in sorted(ks.items())]
    return pool


# ---------------------------------------------------------------- perturb
def _jitter(rng, v):
    if isinstance(v, float):
        new = v * (1 + rng.choice((-1, 1)) * rng.uniform(0.08, 0.20))
        new = float(f"{new:.4g}")
        return new if new != v else v * 1.1
    if v == 0:
        return rng.choice((1, 2))
    a = abs(v)
    if a <= 3:
        step = 1
    elif a <= 15:
        step = rng.randint(1, 3)
    else:
        step = max(1, round(a * rng.uniform(0.08, 0.20)))
    new = v + rng.choice((-1, 1)) * step
    if v > 0 and new < 1:
        new = v + step
    if v < 0 and new > -1:
        new = v - step
    return new


def plan_variant(pool, family, k, attempt, n_flip=N_FLIP, n_jitter=N_JITTER):
    rng = random.Random(f"stress:{family}:{k}:{attempt}")
    bools = [p for p in pool if p[2]["type"] == "bool"]
    nums = [p for p in pool if p[2]["type"] != "bool"]
    pick = rng.sample(bools, min(n_flip, len(bools))) + rng.sample(nums, min(n_jitter, len(nums)))
    diffs = []
    for (unit, nm, site) in pick:
        old = site["value"]
        new = (not old) if site["type"] == "bool" else _jitter(rng, old)
        diffs.append({"file": unit[0], "payload": unit[1], "knob": nm, "old": old, "new": new,
                      "line": site["pos"][0]})
    return diffs


def _splice(src, edits):
    """Replace AST value spans [(pos, text)] -- ast offsets are UTF-8 bytes."""
    b = src.encode("utf-8")
    starts = [0]
    for ln in b.split(b"\n")[:-1]:
        starts.append(starts[-1] + len(ln) + 1)
    spans = []
    for (l0, c0, l1, c1), text in edits:
        spans.append((starts[l0 - 1] + c0, starts[l1 - 1] + c1, text.encode("utf-8")))
    for s, e, t in sorted(spans, reverse=True):
        b = b[:s] + t + b[e:]
    return b.decode("utf-8")


def apply_variant(src_dir, dst_dir, diffs, pool):
    shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("__pycache__"))
    sites = {(u, nm): s for (u, nm, s) in pool}
    by_file = {}
    for d in diffs:
        by_file.setdefault(d["file"], []).append(d)
    for rel, ds in by_file.items():
        path = os.path.join(dst_dir, rel)
        src = open(path, encoding="utf-8").read()
        plain = [(sites[((rel, None), d["knob"])]["pos"], repr(d["new"])) for d in ds if d["payload"] is None]
        payload_edits = {}
        for d in ds:
            if d["payload"] is not None:
                payload_edits.setdefault(d["payload"], []).append(
                    (sites[((rel, d["payload"]), d["knob"])]["pos"], repr(d["new"])))
        if payload_edits:
            matches = list(B64_EXEC.finditer(src))
            repl = {}
            for i, eds in payload_edits.items():
                inner = zlib.decompress(base64.b64decode(matches[i].group(1))).decode("utf-8")
                inner = _splice(inner, eds)
                repl[i] = base64.b64encode(zlib.compress(inner.encode("utf-8"), 9)).decode("ascii")
            # payload literals never overlap the module-level knob lines, but
            # splice them by character offset from the end so both stay valid
            for i in sorted(repl, reverse=True):
                m = matches[i]
                src = src[:m.start(1)] + repl[i] + src[m.end(1):]
            src_tree_ok = ast.parse(src)                                # noqa: F841
            if plain:   # recompute plain positions on the re-encoded source
                ks = scan_knobs(src)
                plain = [(ks[d["knob"]]["pos"], repr(d["new"])) for d in ds if d["payload"] is None]
        if plain:
            src = _splice(src, plain)
        ast.parse(src)
        open(path, "w", encoding="utf-8").write(src)


def entry_name(pkg_dir):
    """Name of the callable Kaggle's loader would run from pkg_dir/main.py."""
    sib = _siblings(pkg_dir)
    saved = {m: sys.modules.pop(m) for m in sib if m in sys.modules}
    sys.path.insert(0, pkg_dir)
    ns = {"__name__": "stress_entry"}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(open(os.path.join(pkg_dir, "main.py"), encoding="utf-8").read(),
                         os.path.join(pkg_dir, "main.py"), "exec"), ns)
    finally:
        sys.path.remove(pkg_dir)
        for m in sib:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    return [v for v in ns.values() if callable(v)][-1].__name__


def _siblings(pkg_dir):
    return sorted(f[:-3] for f in os.listdir(pkg_dir) if f.endswith(".py") and f != "main.py")


def write_shim(name, pkg_dir):
    body = (
        "import sys, os, json, importlib.util\n"
        f"_d = {pkg_dir!r}\n"
        "# evict sibling modules of another copy of this package (stress variants)\n"
        f"_sib = {_siblings(pkg_dir)!r}\n"
        "for _n in _sib: sys.modules.pop(_n, None)\n"
        "if _d in sys.path: sys.path.remove(_d)\n"
        "sys.path.insert(0, _d)\n"
        f"_s = importlib.util.spec_from_file_location('ws_{name}', os.path.join(_d, 'main.py'))\n"
        "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
        "for _n in _sib: sys.modules.pop(_n, None)\n"
        "sys.path.remove(_d)\n"
        f"_CFG = json.load(open({P.LADDER_CONFIG!r}))\n"
        "# Kaggle runs the module's LAST callable (insertion order), not `agent`\n"
        "_inner = [v for v in vars(_m).values() if callable(v)][-1]\n"
        "_argc = _inner.__code__.co_argcount if hasattr(_inner, '__code__') else 2\n"
        "def agent(obs, config=None):\n"
        "    return _inner(obs, config if config is not None else _CFG) if _argc > 1 else _inner(obs)\n")
    path = os.path.join(FIELD, f"{name}.py")
    open(path, "w", encoding="utf-8").write(body)
    return path


# ---------------------------------------------------------------- manifest
def load_manifest():
    try:
        return json.load(open(MANIFEST, encoding="utf-8"))
    except (OSError, ValueError):
        return {"variants": {}, "originals": {}}


def save_manifest(man):
    os.makedirs(FIELD, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    json.dump(man, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, MANIFEST)


def build_variant(family, k, attempt, man, log=print):
    src_dir = family_dir(family)
    pool = knob_pool(src_dir)
    diffs = plan_variant(pool, family, k, attempt)
    name = f"{family}__v{k}"
    dst = os.path.join(AGENTS, name)
    apply_variant(src_dir, dst, diffs, pool)
    want, got = entry_name(src_dir), entry_name(dst)
    if want != got:
        raise RuntimeError(f"{name}: entry point {got!r} != original {want!r}")
    shim = write_shim(name, dst)
    man["variants"][name] = {"family": family, "k": k, "attempt": attempt, "pool_size": len(pool),
                             "entry": got, "dir": P.rel(dst), "shim": P.rel(shim),
                             "diffs": diffs, "smoke": None, "status": "built"}
    log(f"built {name} (attempt {attempt}, pool {len(pool)}): "
        + ", ".join(f"{d['knob']} {d['old']}->{d['new']}" for d in diffs))
    return name


def build(families=None, k=K, log=print):
    os.makedirs(AGENTS, exist_ok=True); os.makedirs(FIELD, exist_ok=True)
    man = load_manifest()
    for fam in families or FAMILIES:
        for i in range(1, k + 1):
            name = f"{fam}__v{i}"
            if name in man["variants"] and man["variants"][name].get("status") in ("built", "accepted"):
                continue
            build_variant(fam, i, man["variants"].get(name, {}).get("attempt", 0), man, log)
            save_manifest(man)
    return man


# ---------------------------------------------------------------- smoke
_SMOKE = r"""
import contextlib, io, json, sys, time
from kaggriculture.bandit.gate import loss_forensics as LF, harness as H
ours_path, opp_path = sys.argv[1], sys.argv[2]
w, seed = H.world_seeds(1)[0]
st = {"calls": 0, "none": 0, "bad": 0, "ms": 0.0, "worst_ms": 0.0}
_orig = LF.load_pyagent
def _wrapped(p):
    f = _orig(p)
    if p != opp_path:
        return f
    argc = f.__code__.co_argcount if hasattr(f, "__code__") else 2
    def inst(obs, config=None):
        t = time.perf_counter()
        a = f(obs, config) if argc > 1 else f(obs)
        ms = (time.perf_counter() - t) * 1000
        st["calls"] += 1; st["ms"] += ms; st["worst_ms"] = max(st["worst_ms"], ms)
        if a is None:
            st["none"] += 1
        elif not (isinstance(a, dict) and isinstance(a.get("farmer"), list)
                  and isinstance(a.get("hands", []), list)
                  and isinstance(a.get("market", []), list) and len(a.get("market", [])) <= 10):
            st["bad"] += 1
        return a
    return inst
LF.load_pyagent = _wrapped
res = {"world": w, "seed": seed}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, us, them = LF.capture_game(_orig(ours_path), opp_path, seed, 0)
    res.update(us=us, them=them)
except Exception as exc:
    res["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
n = max(1, st["calls"])
res.update(calls=st["calls"], none=st["none"], bad=st["bad"],
           mean_ms=round(st["ms"] / n, 2), worst_ms=round(st["worst_ms"], 1))
print("SMOKE " + json.dumps(res))
"""


def smoke_one(shim, ours=OURS):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", _SMOKE, ours, os.path.abspath(shim)],
                       capture_output=True, text=True, timeout=1800, cwd=ROOT, env=env)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("SMOKE ")]
    if not line:
        return {"error": f"rc={r.returncode}: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]}
    return json.loads(line[0][6:])


def verdict(res, base):
    """None if the variant is a realistic clone, else the rejection reason.
    `base` is the original family's smoke result in the same world."""
    base = base or {}
    base_bank = base.get("them")
    if res.get("error"):
        return "crash: " + res["error"]
    calls = res.get("calls", 0)
    if calls and (res["none"] + res["bad"]) > 0.05 * calls:
        return f"bad actions: {res['none']} None + {res['bad']} malformed of {calls}"
    if base_bank and res["them"] < BANK_FLOOR * base_bank:
        return f"degenerate: bank {res['them']:.0f} < {BANK_FLOOR:.0%} of original {base_bank:.0f}"
    if res.get("mean_ms", 0) >= LATENCY_MS:
        return f"slow: mean {res['mean_ms']} ms"
    if base_bank is not None and (res["us"], res["them"]) == (base.get("us"), base_bank):
        # every perturbed knob was dead in this world: a duplicate of the
        # original, not a clone -- it would only re-spend gate budget
        return "inert: identical game to the original in the smoke world"
    return None


def _demote(man, name, why, log):
    v = man["variants"][name]
    rej = v.get("rejected", []) + [{"attempt": v["attempt"], "diffs": v["diffs"], "why": why,
                                     "smoke": v.get("smoke")}]
    log(f"REJECT {name} (attempt {v['attempt']}): {why}")
    if v["attempt"] + 1 >= MAX_ATTEMPTS:
        v["status"] = "failed"; v["rejected"] = rej
        return
    build_variant(v["family"], v["k"], v["attempt"] + 1, man, log)
    man["variants"][name]["rejected"] = rej


def smoke(families=None, workers=2, log=print):
    """Smoke originals then every unaccepted variant; regenerate rejects."""
    man = load_manifest()
    fams = families or FAMILIES
    todo = [f for f in fams if f not in man["originals"]]
    with ThreadPoolExecutor(workers) as ex:
        for fam, res in zip(todo, ex.map(lambda f: smoke_one(os.path.join(P.FIELD, f + ".py")), todo)):
            man["originals"][fam] = res
            log(f"original {fam}: {res}")
            save_manifest(man)
    # re-apply the current verdict to earlier acceptances (criteria can tighten)
    for name, v in list(man["variants"].items()):
        if v["family"] in fams and v["status"] == "accepted":
            why = verdict(v["smoke"], man["originals"].get(v["family"]))
            if why:
                _demote(man, name, why, log); save_manifest(man)
    while True:
        todo = [n for n, v in man["variants"].items()
                if v["family"] in fams and v["status"] == "built"]
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            for name, res in zip(todo, ex.map(lambda n: smoke_one(os.path.join(ROOT, man["variants"][n]["shim"])), todo)):
                v = man["variants"][name]
                base = man["originals"].get(v["family"], {})
                v["smoke"] = res
                why = verdict(res, base)
                if why is None:
                    v["status"] = "accepted"
                    log(f"ACCEPT {name}: bank {res['them']:.0f} (orig {base.get('them', 0):.0f}), "
                        f"v61 {res['us']:.0f}, {res['mean_ms']} ms/turn")
                else:
                    _demote(man, name, why, log)
                save_manifest(man)
    return man


def spot(n=2, seeds=(476105496,), log=print):
    """Official-vs-Rust exact-bank check on n accepted variants (distinct families)."""
    from kaggriculture.winplan import gate as G
    man = load_manifest()
    picked, seen = [], set()
    for name, v in sorted(man["variants"].items()):
        if v["status"] == "accepted" and v["family"] not in seen:
            picked.append(name); seen.add(v["family"])
    # prefer one multi-file package and one embedded-payload package if present
    pref = [x for x in picked if x.startswith(("god_s_mode", "pioneers"))]
    picked = (pref + [x for x in picked if x not in pref])[:n]
    out = {}
    for name in picked:
        res = G.spot_check(OURS, os.path.join(ROOT, man["variants"][name]["shim"]), list(seeds))
        man["variants"][name]["spot_check"] = res
        out[name] = res
        log(f"spot {name}: {res}")
        save_manifest(man)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "smoke", "spot", "knobs"))
    ap.add_argument("--families", default="")
    ap.add_argument("-k", type=int, default=K)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args(argv)
    fams = [f for f in a.families.split(",") if f] or None
    if a.cmd == "build":
        build(fams, a.k)
    elif a.cmd == "smoke":
        smoke(fams, a.workers)
    elif a.cmd == "spot":
        spot(a.n)
    else:
        for fam in fams or FAMILIES:
            pool = knob_pool(family_dir(fam))
            print(fam, len(pool), [f"{nm}={s['value']}" for _, nm, s in pool])


if __name__ == "__main__":
    main()
