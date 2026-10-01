"""Watch for the ladder's 1.32.7 flip and perform the atomic engine swap.

The 2026-08-15 rebalance (PR #1399: hinge scarcity pricing for CARROT/
TOMATO/EGG) was announced with "rolls out shortly". Local measurement must
follow the LADDER, not the announcement -- the 1.32.4 lesson. So:

  --check   scan the freshest staged replays' module_version; if >=1.32.7
            appears, run the swap; otherwise exit quietly. Hooked into the
            hourly scrape, so the swap happens within ~1h of the flip.
  --swap    force the swap now (operator use only).
  --status  print current state.

The atomic swap:
  1. back up vendored 1.32.6 engine files -> .local/_envpkg/vendor_1326_backup
  2. install the staged 1.32.7 files into vendor/
  3. flip ENGINE_1327 in rustengine/src/market.rs and _ENGINE_1327 in
     src/counter_agent.py; rebuild kagg.exe (release)
  4. write models/engine_version.json = 1.32.7 (trackp/common, leak_check,
     build_agent and the parity harness all key on it)
  5. rebuild agents/planner_v0.py (its projections adopt the hinge)
  6. deliberate re-baseline: engine_check.py --update-baseline
  7. cross-check: vendored python quotes == kagg prices over a sweep
  8. revoke stale funnel evidence (preranker_recall.json) -- pre-hinge
     open-loop rankings are measurements of a different game
  9. parity: test_rust_engine (chaos + any 1.32.7 replays staged so far)

Everything is logged loudly to data/logs/engine_swap.log.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys

VENDOR_KG = os.path.join(ROOT, "vendor", "kaggle_environments", "envs",
                         "kaggriculture")
STAGED_KG = os.path.join(ROOT, ".local", "_envpkg", "v1327",
                         "kaggle_environments", "envs", "kaggriculture")
BACKUP = os.path.join(ROOT, ".local", "_envpkg", "vendor_1326_backup")
STATE = os.path.join(ROOT, "models", "engine_version.json")
LOG = os.path.join(ROOT, "data", "logs", "engine_swap.log")
MARKET_RS = os.path.join(ROOT, "rustengine", "src", "market.rs")
COUNTER = os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "counter_agent.py")


def log(msg: str):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def state() -> str:
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh).get("engine", "1.32.6")
    except (OSError, ValueError):
        return "1.32.6"


def ladder_version(n: int = 20) -> str:
    """The MAJORITY module_version across the freshest staged replays.

    Was 'MAX ever seen' -- deliberately sticky so a single 1.32.7 canary
    replay tripped the swap. That one-way trigger is exactly why we stayed
    on 1.32.7 for four days after Kaggle rolled the canary back (2026-08-20):
    the rollback is invisible to a max, and there was no revert path at all.
    A majority over recent replays moves in BOTH directions with the ladder.
    """
    files = []
    for pat in ("data/sameday/_stage/*/*.json",
                "data/ourgames/_stage/*/*.json",
                "data/refresh/*/_stage/*/*.json"):
        files += glob.glob(os.path.join(ROOT, pat))
    try:
        files.sort(key=os.path.getmtime, reverse=True)
    except OSError:
        pass
    seen = []
    import re
    for f in files[: n * 4]:
        try:
            head = open(f, encoding="utf-8", errors="ignore").read(4000)
        except OSError:
            continue
        m = re.search(r'"module_version"\s*:\s*"([^"]+)"', head)
        if not m:
            continue
        seen.append(m.group(1))
        if len(seen) >= n:
            break
    if not seen:
        return ""
    from collections import Counter
    return Counter(seen).most_common(1)[0][0]


def _ver_tuple(v: str):
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _flip_flag(path: str, old: str, new: str):
    text = open(path, encoding="utf-8").read()
    if new in text:
        log(f"  {os.path.basename(path)}: already flipped")
        return
    if old not in text:
        raise SystemExit(f"flag line not found in {path}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.replace(old, new, 1))
    log(f"  flipped {os.path.basename(path)}")


def _run(cmd, timeout=1800, cwd=None) -> int:
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True,
                       timeout=timeout)
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-3:]
    for line in tail:
        log(f"    {line}")
    return r.returncode


def swap() -> int:
    if state() == "1.32.7":
        log("swap: already on 1.32.7 -- nothing to do")
        return 0
    if not os.path.exists(os.path.join(STAGED_KG, "kaggriculture.py")):
        log("swap ABORT: staged 1.32.7 files missing (.local/_envpkg/v1327)")
        return 1
    log("ENGINE SWAP 1.32.6 -> 1.32.7 BEGINS")

    os.makedirs(BACKUP, exist_ok=True)
    for f in ("kaggriculture.py", "kaggriculture.json"):
        shutil.copy(os.path.join(VENDOR_KG, f), os.path.join(BACKUP, f))
    log("  vendored 1.32.6 backed up")

    for f in ("kaggriculture.py", "kaggriculture.json"):
        shutil.copy(os.path.join(STAGED_KG, f), os.path.join(VENDOR_KG, f))
    shutil.rmtree(os.path.join(VENDOR_KG, "__pycache__"),
                  ignore_errors=True)
    log("  1.32.7 installed into vendor/")

    _flip_flag(MARKET_RS, "pub const ENGINE_1327: bool = false;",
               "pub const ENGINE_1327: bool = true;")
    # every python-side price-model template (found by the 2026-08-16 audit:
    # counter_agent, the v22 bandit template, the pool/tape templates, and
    # the v1 route source -- v2/route/bandit builds inherit on regeneration)
    for pyf in (COUNTER,
                os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "v22_agent.py"),
                os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "pool_agent.py"),
                os.path.join(ROOT, "src", "kaggriculture", "engine", "tape_runtime.py"),
                os.path.join(ROOT, "agents", "v1_heuristic.py")):
        _flip_flag(pyf, "_ENGINE_1327 = False", "_ENGINE_1327 = True")

    log("  rebuilding kagg.exe (release)")
    if _run(["cargo", "build", "--release"], timeout=900,
            cwd=os.path.join(ROOT, "rustengine")) != 0 \
            or _run_cargo_check() != 0:
        log("swap ABORT: cargo build failed -- flags left flipped, "
            "FIX BY HAND")
        return 1

    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump({"engine": "1.32.7",
                   "swapped": dt.datetime.now().isoformat(
                       timespec="seconds")}, fh, indent=1)
    log("  engine_version.json -> 1.32.7")

    log("  rebuilding planner_v0 (hinge projections)")
    args = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "trackp",
                                         "build_agent.py"),
            "--out", os.path.join(ROOT, "agents", "planner_v0.py")]
    pb = os.path.join(ROOT, "models", "trackp", "params_best.json")
    if os.path.exists(pb):
        args += ["--params", pb]
    _run(args, timeout=300)

    log("  re-baselining engine_check")
    _run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "engine", "engine_check.py"),
          "--update-baseline"], timeout=600)

    log("  cross-check: vendored python quotes vs kagg prices")
    code = _run([sys.executable, "-c", CROSS_CHECK], timeout=600)
    if code != 0:
        log("swap WARNING: price cross-check FAILED -- investigate before "
            "trusting any engine number")

    recall = os.path.join(ROOT, "models", "lab", "preranker_recall.json")
    if os.path.exists(recall):
        os.remove(recall)
        log("  revoked preranker recall evidence (pre-hinge rankings are a "
            "different game); the funnel stays closed until re-measured")

    log("  parity: test_rust_engine (chaos + any 1.32.7 replays)")
    _run([sys.executable, os.path.join(ROOT, "tests",
                                       "test_rust_engine.py"),
          "--episodes", "2", "--replays", "3"], timeout=1800)

    log("ENGINE SWAP COMPLETE -- all pre-1.32.7 scarcity-price conclusions "
        "for CARROT/TOMATO/EGG are void; next 04:30 cycle re-crowns on the "
        "new economics")
    return 0


def unswap() -> int:
    """Revert 1.32.7 -> 1.32.6 (2026-08-20). The ladder canaried 1.32.7 on
    2026-08-15/16, this watcher swapped us, and Kaggle then ROLLED BACK: the
    index shows 0 of 3,852 routes on 1.32.7 for 2026-08-18/19/20, all 1.32.6.
    We stayed on 1.32.7 for four days because --check was one-way (sticky
    state + max-ever ladder_version), crowning every agent on an engine the
    ladder does not run and baking hinge CARROT/TOMATO/EGG pricing into the
    shipped tapes -- a direct mispricing bug on a 1.32.6 ladder."""
    if state() == "1.32.6":
        log("unswap: already on 1.32.6 -- nothing to do")
        return 0
    if not os.path.exists(os.path.join(BACKUP, "kaggriculture.py")):
        log("unswap ABORT: 1.32.6 backup missing (.local/_envpkg/"
            "vendor_1326_backup) -- cannot restore vendored engine by hand")
        return 1
    log("ENGINE REVERT 1.32.7 -> 1.32.6 BEGINS")

    for f in ("kaggriculture.py", "kaggriculture.json"):
        shutil.copy(os.path.join(BACKUP, f), os.path.join(VENDOR_KG, f))
    shutil.rmtree(os.path.join(VENDOR_KG, "__pycache__"), ignore_errors=True)
    log("  vendored 1.32.6 restored from backup")

    _flip_flag(MARKET_RS, "pub const ENGINE_1327: bool = true;",
               "pub const ENGINE_1327: bool = false;")
    for pyf in (COUNTER,
                os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "v22_agent.py"),
                os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "pool_agent.py"),
                os.path.join(ROOT, "src", "kaggriculture", "engine", "tape_runtime.py"),
                os.path.join(ROOT, "agents", "v1_heuristic.py")):
        _flip_flag(pyf, "_ENGINE_1327 = True", "_ENGINE_1327 = False")

    log("  rebuilding kagg.exe (release)")
    if _run(["cargo", "build", "--release"], timeout=900,
            cwd=os.path.join(ROOT, "rustengine")) != 0:
        log("unswap ABORT: cargo build failed -- flags left flipped, FIX BY "
            "HAND")
        return 1
    if _cargo_check_1326() != 0:
        log("unswap WARNING: post-build price probe did not look like 1.32.6 "
            "-- investigate before trusting any engine number")

    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump({"engine": "1.32.6",
                   "swapped": dt.datetime.now().isoformat(timespec="seconds"),
                   "reverted_from": "1.32.7",
                   "reason": "ladder rolled back 1.32.7 canary; index shows "
                             "0/3852 routes on 1.32.7 for 08-18..08-20"},
                  fh, indent=1)
    log("  engine_version.json -> 1.32.6")

    log("  rebuilding planner_v0 (1.32.6 projections)")
    args = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "trackp",
                                         "build_agent.py"),
            "--out", os.path.join(ROOT, "agents", "planner_v0.py")]
    pb = os.path.join(ROOT, "models", "trackp", "params_best.json")
    if os.path.exists(pb):
        args += ["--params", pb]
    _run(args, timeout=300)

    log("  re-baselining engine_check")
    _run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "engine", "engine_check.py"),
          "--update-baseline"], timeout=600)

    # Every 1.32.7-era measurement is now a different game: the pre-ranker
    # recall and the serve-substrate equivalence audit must both be re-earned
    # on 1.32.6 before they gate anything.
    for stale in (os.path.join(ROOT, "models", "lab", "preranker_recall.json"),
                  os.path.join(ROOT, "models", "serve_equiv.json")):
        if os.path.exists(stale):
            os.remove(stale)
            log(f"  revoked stale 1.32.7 evidence: "
                f"{os.path.relpath(stale, ROOT)}")

    log("ENGINE REVERT COMPLETE -- back on the ladder's 1.32.6. All agents "
        "crowned/built since 2026-08-16 were measured on the wrong engine; "
        "re-crown and rebuild the live pair on 1.32.6.")
    return 0


def _cargo_check_1326() -> int:
    """After a 1.32.6 build the hinge prices must be GONE (CARROT != 531)."""
    kagg = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
    r = subprocess.run([kagg, "prices", "9000"], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        return 1
    d = json.loads(r.stdout)
    hinge = (d.get("CARROT") == 531 and d.get("TOMATO") == 3252
             and d.get("EGG") == 758)
    log(f"    kagg probe @9000: carrot {d.get('CARROT')} tomato "
        f"{d.get('TOMATO')} egg {d.get('EGG')} -> "
        f"{'STILL HINGE (bad)' if hinge else 'non-hinge 1.32.6 (ok)'}")
    return 1 if hinge else 0


def _run_cargo_check() -> int:
    kagg = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
    r = subprocess.run([kagg, "prices", "9000"], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        return 1
    d = json.loads(r.stdout)
    ok = d.get("CARROT") == 531 and d.get("TOMATO") == 3252 \
        and d.get("EGG") == 758
    log(f"    kagg hinge probe @9000: carrot {d.get('CARROT')} tomato "
        f"{d.get('TOMATO')} egg {d.get('EGG')} -> "
        f"{'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


CROSS_CHECK = r"""
import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.getcwd(), 'vendor'))
from kaggle_environments.envs.kaggriculture import kaggriculture as K
kagg = os.path.join('rustengine', 'target', 'release', 'kagg.exe')
bad = 0
for inv in range(2000, 20001, 613):
    rust = json.loads(subprocess.run([kagg, 'prices', str(inv)],
                      capture_output=True, text=True).stdout)
    for item, p in K.MARKET_PARAMS.items():
        py = K._market_price(p, float(inv)) if hasattr(K, '_market_price') \
            else None
        if py is None:
            base, I0, T = p['base'], p['I0'], p['T']
            if inv < I0:
                f = p['below_func']
                amp = p['below_target']*base/K._shape(f, T, T)
                raw = base + amp*K._shape(f, I0-inv, T)
            else:
                f = p['above_func']
                amp = p['above_target']*base/K._shape(f, T, T)
                raw = base - amp*K._shape(f, inv-I0, T)
            py = max(1, int(round(raw)))
        if py != rust.get(item):
            bad += 1
            print('MISMATCH', item, inv, py, rust.get(item))
print('cross-check:', 'PASS' if bad == 0 else f'{bad} mismatches')
sys.exit(1 if bad else 0)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--revert", "--unswap", action="store_true",
                    dest="revert", help="force 1.32.7 -> 1.32.6 (operator)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status or not (a.check or a.swap or a.revert):
        print(json.dumps({"state": state(),
                          "ladder_majority": ladder_version()}))
        return 0
    if a.swap:
        return swap()
    if a.revert:
        return unswap()
    # --check: TWO-WAY. Follow the ladder majority in either direction.
    lv = ladder_version()
    if not lv:
        return 0
    cur = state()
    on7 = _ver_tuple(lv) >= (1, 32, 7)
    if on7 and cur != "1.32.7":
        log(f"LADDER FLIP DETECTED (majority {lv}): swapping to 1.32.7")
        return swap()
    if not on7 and cur != "1.32.6":
        log(f"LADDER ROLLED BACK (majority {lv}): reverting to 1.32.6")
        return unswap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
