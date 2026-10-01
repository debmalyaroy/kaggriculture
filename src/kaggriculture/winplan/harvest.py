"""Public notebooks -> runnable, classified gate opponents.

1. `kaggriculture.data.notebooks.harvest` pulls new public notebooks.
2. Each notebook's PACKAGING cells run in its own sandbox dir (shell / pip /
   evaluation cells skipped) and the produced submission.tar.gz (or main.py)
   is unpacked IN FULL -- multi-file packages (router.py + actions.json,
   mirror_plan.py, ...) are common, and keeping only main.py breaks them.
3. Every package is classified from its code (lineage markers, binary check)
   and smoke-loaded on a real observation.
4. A loader shim per package goes into data/winplan/field/: it puts the
   package dir on sys.path and passes the LADDER configuration to agents that
   take a 2nd arg (the gate harness calls agent(obs) only, and some agents --
   fieldcraft -- require `config`).
Duplicates (identical package contents) are kept once.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
from concurrent.futures import ThreadPoolExecutor

from kaggriculture.paths import ROOT
from kaggriculture.winplan import paths as P

KDIR = os.path.join(ROOT, "data", "kernels")
SKIP = re.compile(r"\benv\.run\(|\bmake\(\s*['\"]kaggriculture|evaluate\(|\binstall\b|"
                  r"kaggle_environments\.make|run_match|tournament|for\s+seed\s+in")
WF = re.compile(r"^\s*%%writefile\s+(-a\s+)?(\S+)[^\n]*\n", re.M)
MARKERS = {
    "chassis": r"class Chassis\b|ROUTES_DATA", "metav4": r"(?i)metav4",
    "ahmed_vNN": r"Berat|ahmedberatozer", "yhay81_router": r"yhay81|Shop Router",
    "prvsiyan": r"prvsiyan", "tetsutani": r"tetsutani",
    "nathanjacob_pipe": r"(?i)nathanjacob|Pipe-\d+", "gluzdov": r"(?i)gluzdov",
    "herd_safe": r"(?i)herd.?safe", "race_clone_detector": r"def _race_clone\(",
    "rival_model": r"(?i)rival_lead|board_similarity|_race_lost",
    "subprocess/binary": r"subprocess|ctypes|os\.system",
}


def _script_for(nb):
    cells = json.load(open(nb, encoding="utf-8", errors="replace")).get("cells", [])
    parts = ["import os, sys\nsys.argv=['nb']\n"]
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        m = WF.match(src)
        if m:
            body = src[m.end():]
            mode = "a" if m.group(1) else "w"
            parts.append(
                f"import pathlib as _pl\n_p=_pl.Path({m.group(2)!r}.replace('/kaggle/working/','').lstrip('/') or 'main.py')\n"
                f"_p.parent.mkdir(parents=True, exist_ok=True)\nopen(_p,{mode!r},encoding='utf-8').write({body!r})\n")
            continue
        if SKIP.search(src):
            continue
        code = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        code = code.replace("/kaggle/working/", "./").replace("/kaggle/working", ".")
        parts.append(f"try:\n{textwrap.indent(code, '    ') or '    pass'}\n    pass\nexcept SystemExit:\n    pass\n"
                     f"except BaseException as _e:\n    print('CELL {i} ERR', type(_e).__name__, str(_e)[:120])\n")
    return "\n".join(parts)


def _package_from(wd):
    """{member: bytes} of the package a sandbox produced, root-relative to main.py."""
    for t in sorted(glob.glob(os.path.join(wd, "**", "*.tar.gz"), recursive=True)):
        try:
            with tarfile.open(t) as tf:
                mm = {m.name.lstrip("./"): tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        except Exception:                                            # noqa: BLE001
            continue
        mains = [k for k in mm if os.path.basename(k) == "main.py"]
        if mains:
            base = os.path.dirname(min(mains, key=len))
            return {os.path.relpath(k, base) if base else k: v for k, v in mm.items()}
    mp = glob.glob(os.path.join(wd, "**", "main.py"), recursive=True)
    return {"main.py": open(mp[0], "rb").read()} if mp else {}


def run_packaging(ref):
    slug = ref.replace("/", "__")[:80]
    nbs = glob.glob(os.path.join(KDIR, ref.replace("/", "_"), "*.ipynb"))
    if not nbs:
        return slug, {}, "no ipynb"
    wd = os.path.join(P.NBRUN, slug)
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    open(os.path.join(wd, "_run.py"), "w", encoding="utf-8").write(_script_for(nbs[0]))
    try:
        subprocess.run([sys.executable, "_run.py"], cwd=wd, capture_output=True, timeout=240,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        how = "ran"
    except subprocess.TimeoutExpired:
        how = "timeout"
    pkg = _package_from(wd)
    code = b"".join(v for k, v in pkg.items() if k.endswith(".py"))
    if not pkg or (b"def agent" not in code and b"agent =" not in code and b"agent=" not in code):
        return slug, {}, f"no agent ({how})"
    return slug, pkg, "ok"


def _binary(b):
    if b[:4] == b"\x7fELF" or b[:2] == b"MZ":
        return "EXECUTABLE"
    try:
        b.decode("utf-8"); return None
    except UnicodeDecodeError:
        return "non-utf8"


def _shim(slug, pkg_dir):
    short = re.sub(r"[^A-Za-z0-9_]", "_", slug.split("__", 1)[-1])[:48]
    body = (
        "import sys, os, json, importlib.util\n"
        f"_d = {pkg_dir!r}\n"
        "if _d not in sys.path: sys.path.insert(0, _d)\n"
        f"_s = importlib.util.spec_from_file_location('wp_{short}', os.path.join(_d, 'main.py'))\n"
        "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
        f"_CFG = json.load(open({P.LADDER_CONFIG!r}))\n"
        "# Kaggle runs the module's LAST callable (insertion order), not `agent`\n"
        "_inner = [v for v in vars(_m).values() if callable(v)][-1]\n"
        "_argc = _inner.__code__.co_argcount if hasattr(_inner, '__code__') else 2\n"
        "def agent(obs, config=None):\n"
        "    return _inner(obs, config if config is not None else _CFG) if _argc > 1 else _inner(obs)\n")
    path = os.path.join(P.FIELD, f"{short}.py")
    open(path, "w", encoding="utf-8").write(body)
    return path


def install(slug, pkg, catalog):
    """Write a package into data/winplan/agents/<slug>, classify, smoke, shim."""
    h = hashlib.sha256(b"".join(pkg[k] for k in sorted(pkg))).hexdigest()[:12]
    for other, rec in catalog.items():
        if rec.get("hash") == h and other != slug:
            catalog[slug] = {"dup_of": other, "hash": h}
            return catalog[slug]
    d = os.path.join(P.AGENTS, slug)
    shutil.rmtree(d, ignore_errors=True)
    for k, v in pkg.items():
        p = os.path.join(d, k); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(v)
    code = "\n".join(v.decode("utf-8", "replace") for k, v in pkg.items() if k.endswith(".py"))
    rec = {"hash": h, "members": {k: len(v) for k, v in pkg.items()},
           "binary_members": {k: _binary(v) for k, v in pkg.items() if _binary(v)},
           "py_bytes": len(code), "markers": [m for m, rx in MARKERS.items() if re.search(rx, code)]}
    if rec["py_bytes"] < 10000:
        rec["status"] = "skipped: toy (<10KB)"
    else:
        shim = _shim(slug, d)
        r = subprocess.run([sys.executable, "-c",
                            "import importlib.util as u,sys;s=u.spec_from_file_location('x',sys.argv[1]);"
                            "m=u.module_from_spec(s);s.loader.exec_module(m);assert callable(m.agent)", shim],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            rec["status"] = "ok"; rec["shim"] = P.rel(shim)
        else:
            os.remove(shim); rec["status"] = "load fail: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120]
    catalog[slug] = rec
    return rec


def catalog_path():
    return os.path.join(P.DATA, "catalog.json")


def load_catalog():
    try:
        return json.load(open(catalog_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ensure_ladder_config():
    if os.path.exists(P.LADDER_CONFIG):
        return
    reps = glob.glob(os.path.join(P.REPLAYS, "*.json")) + glob.glob(os.path.join(ROOT, ".local", "crawl", "replays", "*.json"))
    cfg = json.load(open(reps[0], encoding="utf-8"))["configuration"] if reps else {
        "actTimeout": 1, "boardSize": 10, "episodeSteps": 720, "farmHandCostMult": 1, "marketParams": {},
        "maxMarketOrdersPerTurn": 10, "runTimeout": 1200, "seed": None, "shedCapacity": 100,
        "startingMoney": 3000, "townCenterSellInterval": 24, "townShopSellInterval": 4,
        "townShopUnlockInterval": 3, "turnsPerDay": 24, "weedSpawnChance": 0.005}
    json.dump(cfg, open(P.LADDER_CONFIG, "w"))


def migrate_legacy():
    """One-off: adopt today's .local/newfield packages without re-running notebooks."""
    ensure_ladder_config()
    legacy = os.path.join(ROOT, ".local", "newfield")
    cat = load_catalog()
    for d in sorted(glob.glob(os.path.join(legacy, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in cat:
            continue
        pkg = {}
        for root, _, files in os.walk(d):
            if "__pycache__" in root:
                continue
            for f in files:
                p = os.path.join(root, f)
                pkg[os.path.relpath(p, d).replace("\\", "/")] = open(p, "rb").read()
        if "main.py" in pkg:
            install(os.path.basename(d), pkg, cat)
    json.dump(cat, open(catalog_path(), "w"), indent=1)
    return cat


def harvest(since, top=100, jobs=6, log=print):
    """Pull notebooks, package + install every one run since `since` not yet in the catalog."""
    import kaggriculture.data.notebooks as NB
    ensure_ladder_config()
    NB.harvest(top=top, verbose=False)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "notebooks.csv"), encoding="utf-8"))
            if r.get("last_run", "") >= since]
    cat = load_catalog()
    todo = [r["ref"] for r in rows if r["ref"].replace("/", "__")[:80] not in cat]
    log(f"{len(rows)} notebooks since {since}; {len(todo)} new to package")
    new = []
    with ThreadPoolExecutor(jobs) as ex:
        for slug, pkg, how in ex.map(run_packaging, todo):
            if pkg:
                rec = install(slug, pkg, cat)
                if rec.get("status") == "ok":
                    new.append(slug)
            else:
                cat[slug] = {"status": how}
    json.dump(cat, open(catalog_path(), "w"), indent=1)
    return {"notebooks": len(rows), "packaged": len(todo), "new_agents": new,
            "field_size": len([f for f in os.listdir(P.FIELD) if f.endswith(".py")])}
