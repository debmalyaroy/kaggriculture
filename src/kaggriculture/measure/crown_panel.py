"""The BANDED referee panel for the crown gate (Workstream C, 2026-09-18).

A crown decided on one agent's loss tapes saturates (memory:
crown-gate-saturated). This assembles a panel keyed by LIVE rating band --

    <2100 | 2100-2300 | 2300-2500 | 2500-2700 | 2700+

-- so `crown_gate.crown_gate_banded` can paired-test a challenger against the
incumbent PER BAND and never be fooled by a swept low band while a high band
regresses. Referees come from three vetted sources:

  1. the reactive agents certified in .local/serve_equiv_reactive.json (our
     internal agents, the weak end of the panel);
  2. the public-frontier reactive agents in data/gauntlet, each attributed to
     its author's CURRENT leaderboard rating (exact team/username match);
  3. the two ~2500 reference tapes in .local/panel/ref2500 (explicitly rated).

Every candidate is VETTED before it enters the panel: it must load, export
agent(obs), play a full 720-step episode on the serve substrate WITHOUT
crashing, and REPRODUCE bank-for-bank on a repeated run (a non-deterministic
referee makes every paired test a lie). Vetted referees are FROZEN as copies
under .local/crown_panel/refs/<band>/ so another lane cannot change a referee
under a running comparison, and the banded manifest is written to
models/crown_panel.json.

Each band is split into a SELECTION half (the referees the gate scores on) and
a HELD-OUT reserve (never used in selection) for the overfit alarm --
`crown_gate.holdout_alarm`. The split is deterministic and stratified within
the band.

    python -m kaggriculture.measure.crown_panel --build
    python -m kaggriculture.measure.crown_panel --build --per-band 6
    python -m kaggriculture.measure.crown_panel --show
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PANEL_JSON = os.path.join(ROOT, "models", "crown_panel.json")
REF_DIR = os.path.join(ROOT, ".local", "crown_panel", "refs")
LB_DIR = os.path.join(ROOT, ".local", "crown_panel", "lb")
# LOWER-CONFIDENCE tape referees rendered from the top-100 replay corpus'
# high-rated WINNER seats (measure.opponents.build_tape). A lifted replay tape
# DESYNCs in a new world (memory: early-splice-2026-09-04, recording-solves-
# desync), so it does NOT faithfully represent that team's reactive strength --
# it is the ONLY runnable signal we hold for the >=2700 frontier, tagged
# `tape_desync_risk` so the gate can weight or skip it and NEVER placed in the
# holdout (it must not raise the overfit alarm on a known-approximate referee).
DESYNC_DIR = os.path.join(ROOT, ".local", "crown_panel", "desync_tapes")

# Band edges. `crown_gate.BAND_WEIGHT` weights these by rating centroid.
BANDS = [("<2100", -1e9, 2100.0), ("2100-2300", 2100.0, 2300.0),
         ("2300-2500", 2300.0, 2500.0), ("2500-2700", 2500.0, 2700.0),
         ("2700+", 2700.0, 1e9)]
BAND_NAMES = [b[0] for b in BANDS]
PER_BAND = 6
HOLDOUT_PER_BAND = 1          # reserved referees per band (when >= 3 in band)
VET_SEEDS = (77001, 77002)    # two worlds; each played twice for determinism

# Fixed VET opponent: a simple, deterministic reactive agent every candidate
# can play a full episode against.
VET_OPP = os.path.join(ROOT, "agents", "v0_baseline.py")


def band_of(rating):
    if rating is None:
        return None
    for name, lo, hi in BANDS:
        if lo <= rating < hi:
            return name
    return None


# --------------------------------------------------------------- rating map --

def _lb_csv():
    """Freshest leaderboard CSV: download, else the newest already on disk."""
    os.makedirs(LB_DIR, exist_ok=True)
    import subprocess
    import zipfile
    src = None
    try:
        subprocess.run(["kaggle", "competitions", "leaderboard", "kaggriculture",
                        "--download", "-p", LB_DIR], capture_output=True,
                       text=True, timeout=300, encoding="utf-8",
                       errors="replace")
        z = os.path.join(LB_DIR, "kaggriculture.zip")
        if os.path.exists(z):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".csv"):
                        safe = os.path.join(LB_DIR,
                                            os.path.basename(n).replace(":", "_"))
                        with zf.open(n) as fi, open(safe, "wb") as fo:
                            fo.write(fi.read())
                        src = safe
            os.remove(z)
    except Exception as exc:                                       # noqa: BLE001
        print(f"WARNING leaderboard download failed: {exc}")
    if src is None:
        cands = sorted(glob.glob(os.path.join(LB_DIR, "*.csv"))
                       + glob.glob(os.path.join(ROOT, ".local", "crown_panel_lb",
                                                "*.csv"))
                       + glob.glob(os.path.join(ROOT, ".local", "band_panel",
                                                "lb", "*.csv")),
                       key=os.path.getmtime)
        if not cands:
            return None, None
        src = cands[-1]
        print(f"WARNING using leaderboard on disk: {src}")
    return src, src


def rating_map():
    """{lowercased team name OR username: (score, rank)} from the live LB.

    Returns ({}, None) if no leaderboard is reachable -- the caller then falls
    back to explicit/nominal ratings only.
    """
    src, _ = _lb_csv()
    if not src:
        return {}, None
    out = {}
    with open(src, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                sc = float(r["Score"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                rk = int(r.get("Rank") or 0)
            except ValueError:
                rk = 0
            tn = (r.get("TeamName") or "").strip().lower()
            if tn:
                out.setdefault(tn, (sc, rk))
            for u in (r.get("TeamMemberUserNames") or "").split(","):
                u = u.strip().lower()
                if u:
                    out[u] = (sc, rk)
    return out, src


# ------------------------------------------------------------- the seed set --
#
# (relative source path, author token for an EXACT LB lookup, explicit rating).
# Explicit rating wins; else the author's current LB rating; else the agent is
# skipped (never guessed into a band). Internal agents carry a NOMINAL rating
# only to anchor the weak end -- flagged `internal` in the manifest.

def _gauntlet(name, token):
    return (f"data/gauntlet/{name}.py", token, None, "reactive-public")


SEED = [
    # 1. ~2500 reference tapes (explicitly rated).
    (".local/panel/ref2500/ref_v46_chassis_2517.py", None, 2517.0, "tape-ref"),
    (".local/panel/ref2500/ref_k0006_2494.py", None, 2494.0, "tape-ref"),
    # 2a. public-frontier reactive agents, author -> current LB rating. Missing
    #     files (the data/gauntlet roster is regenerable and changes) are
    #     skipped by _attribute, not an error -- keep the whole roster listed so
    #     it is picked up automatically when those files return.
    _gauntlet("lynnsakurai_farming_score_v4_a_better_shop", "lynnsakurai"),
    _gauntlet("aurax7_kaggriculture_shop_router_reactive_v5", "aurax7"),
    _gauntlet("pub_nathanjacob_anticlone", "nathanjacob"),
    _gauntlet("pub_tetsutani_mirror", "tetsutani"),
    # pub_hesoponyo_transformer (2485.8) EXCLUDED: a pure-python BC+PPO
    # entity-token Transformer forward pass takes seconds PER STEP -> minutes
    # per 720-step episode and ~2.4 GB RSS, far past the referee latency budget
    # and a re-OOM risk on this box. The 2300-2500 band is full without it.
    _gauntlet("kaito_v48", "kaitofukami"),
    _gauntlet("kaito_v43", "kaitofukami"),
    _gauntlet("y3uanm_kaggriculture_market_impact_router_v4", "y3uanm"),
    _gauntlet("pub_prvsiyan_yhay81_13tape", "yhay81"),
    _gauntlet("boatlee_v16_rc5_r5a_high_score_8c_4s_recovery", "boatlee"),
    _gauntlet("munib", "munibmemon"),
    # 2b. LB-rating-labelled rendered mid-band tapes (data/gauntlet/mid_<rating>_
    #     <episode>.py). The rating is in the FILENAME, so no author lookup is
    #     needed -- these fill the 2300-2700 span the current gauntlet roster is
    #     thin on. Rendered from real winning routes of teams in that band.
    ("data/gauntlet/mid_2350_107512923.py", None, 2350.0, "tape-mid"),
    ("data/gauntlet/mid_2387_109375510.py", None, 2387.0, "tape-mid"),
    ("data/gauntlet/mid_2414_106901648.py", None, 2414.0, "tape-mid"),
    ("data/gauntlet/mid_2454_106127596.py", None, 2454.0, "tape-mid"),
    ("data/gauntlet/mid_2490_106296373.py", None, 2490.0, "tape-mid"),
    ("data/gauntlet/mid_2518_107079921.py", None, 2518.0, "tape-mid"),
    # 2c. reactive PUBLIC kernels attributable to an author's current LB rating
    #     (HIGH-confidence reactive, not a lifted tape). alperen5252525's scored
    #     shop-router agent -- a distinct lineage from the pub_* agents already
    #     in 2500-2700. (lime0001's kernel is the same experiment byte-for-byte
    #     modulo comments, so it is NOT added -- identical behaviour under two
    #     ratings would be fake diversity the referee-power pruner should see as
    #     one agent.)
    ("data/kernels/alperen5252525_kaggriculture-metacounter-r1-scored-agent/"
     "kaggriculture-metacounter-r1-scored-agent.py", "alperen5252525", 2594.2,
     "reactive-kernel"),
    # 2d. DECODED reactive public kernels from the top-cohort scrape
    #     (`.local/scratch/gm/high_agents_index.csv`: author_ladder + decoded>0).
    #     These are real reactive code (HIGH confidence), rated by the AUTHOR's
    #     competition ladder rating (per operator 2026-09-18). Each was VETTED
    #     one-per-subprocess (load + full 720-step serve + deterministic); the
    #     ones EXCLUDED for being too slow to referee (>150s / a handful of
    #     episodes, like the excluded transformer) are:
    #       tokenjunkielabs/titan-frontier (2732) and bruceqdu/rl-mcts (2551).
    #     biohack44 (2719) decoded only a data blob (no runnable agent).
    #     ---- 2700+ (fills the previously-EMPTY frontier band) ----
    ("data/kernels/_agents/"
     "kaggriculture-adaptive-public-state-multi-route___MOON_PAYLOAD.py",
     "yamakawanin", 2837.0, "reactive-kernel"),
    ("data/kernels/_agents/"
     "kaggriculture-adaptive-public-state-multi-route___MUNIB_PAYLOAD.py",
     "yamakawanin", 2837.0, "reactive-kernel"),
    ("data/kernels/_agents/"
     "kaggriculture-adaptive-public-state-multi-route___MUTOY_PAYLOAD.py",
     "yamakawanin", 2837.0, "reactive-kernel"),
    ("data/kernels/_agents/"
     "kaggriculture-findings-from-zero-to-top-meta___AGENT_B64_PARTS.py",
     "raykkretzschmar", 2763.0, "reactive-kernel"),
    #     (rayk's ___C95_AGENT_B64_PARTS.py is the same agent to 6 lines -- not
    #     added, identical behaviour under one rating is not diversity.)
    #     ---- 2500-2700 (thicken) ----
    ("data/kernels/_agents/kaggriculture-baseline__AGENT_B85.py",
     "pavloivanin", 2597.0, "reactive-kernel"),
    ("data/kernels/_agents/best-market-agent-high-strategy__AGENT_B64.py",
     "reyhanksatria", 2530.0, "reactive-kernel"),
    ("data/kernels/_agents/kaggriculture-adaptive-shop-guard__BLOB.py",
     "reyhanksatria", 2530.0, "reactive-kernel"),
    #     ---- 2100-2300 (thicken the THIN band: was 1 referee) ----
    ("data/kernels/_agents/v29-r1-adaptive-market-hysteresis__PAYLOAD.py",
     "boatlee", 2230.0, "reactive-kernel"),
    ("data/kernels/_agents/v21-r1-public-state-route-portfolio__PAYLOAD.py",
     "boatlee", 2230.0, "reactive-kernel"),
    ("data/kernels/_agents/strong-statr-barnyard-economist___AGENT_B85_PARTS.py",
     "akhileshgodugu", 2172.0, "reactive-kernel"),
    # 3. certified reactive internal agents (weak-end anchors; NOMINAL rating).
    ("agents/agent_v4_optimal_20260805_014340.py", None, 1980.0, "internal"),
    ("agents/agent_vadapt_both_20260805_225549.py", None, 1960.0, "internal"),
    ("agents/agent_vadapt_risk_20260805_032413.py", None, 1940.0, "internal"),
    ("agents/agent_vadapt_both_20260805_032405.py", None, 1920.0, "internal"),
    ("agents/ml_ridge.py", None, 1700.0, "internal"),
    ("agents/v0_baseline.py", None, 1400.0, "internal"),
]


def _attribute(rmap):
    """Resolve every SEED entry to (path, name, rating, source). Drops entries
    whose file is missing or whose rating cannot be established."""
    out = []
    for rel, token, explicit, kind in SEED:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  skip (missing): {rel}")
            continue
        rating, src = explicit, kind
        if rating is None and token:
            hit = rmap.get(token.lower())
            if hit:
                rating = hit[0]
            else:
                print(f"  skip (no LB rating for {token!r}): {rel}")
                continue
        if rating is None:
            print(f"  skip (no rating): {rel}")
            continue
        name = os.path.splitext(os.path.basename(rel))[0]
        out.append({"path": p, "name": name, "rating": float(rating),
                    "source": src, "token": token, "desync": False})
    return out


def _desync_candidates(dir_=DESYNC_DIR):
    """Tape referees rendered from top-100 winner replays (LOWER confidence).

    Files are `tape27_<rating>_<episode>_s<seat>.py`; the rating is in the name
    (the replay payload carries none). Every one is flagged `desync=True` so it
    is tagged in the manifest, forced to the SELECTION split, and excluded from
    the holdout overfit alarm.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(dir_, "tape27_*.py"))):
        name = os.path.splitext(os.path.basename(p))[0]
        parts = name.split("_")
        try:
            rating = float(parts[1])
        except (IndexError, ValueError):
            continue
        out.append({"path": p, "name": name, "rating": rating,
                    "source": "tape-desync-risk", "token": None,
                    "desync": True})
    return out


# ----------------------------------------------------------------- vetting --

def vet(path, srv=None, opp=VET_OPP, seeds=VET_SEEDS):
    """(ok, reason). Loads, plays each seed twice on serve, requires no crash
    and bank-for-bank reproduction (determinism). Runs IN-PROCESS -- used by
    `--vet-one`, which is one candidate per interpreter so a heavy tape's
    memory is fully reclaimed on exit (the panel build spawns this per
    candidate; loading every tape in one process OOM'd the box, memory:
    c9-opening-nogo)."""
    import kaggriculture.engine.serve_match as SM
    own = srv is None
    if own:
        srv = SM.Serve()
    try:
        try:
            agent = SM.load_agent(path)
            oppfn = SM.load_agent(opp)
        except Exception as exc:                                   # noqa: BLE001
            return False, f"load failed: {type(exc).__name__}: {str(exc)[:80]}"
        for seed in seeds:
            try:
                r1 = SM.run_match(agent, oppfn, seed, srv)
                r2 = SM.run_match(SM.load_agent(path), oppfn, seed, srv)
            except Exception as exc:                               # noqa: BLE001
                return False, (f"crash seed {seed}: {type(exc).__name__}: "
                               f"{str(exc)[:80]}")
            if r1 != r2:
                return False, f"non-deterministic seed {seed}: {r1} != {r2}"
        return True, "ok"
    finally:
        if own:
            srv.close()


def vet_subprocess(path):
    """VET `path` in a FRESH interpreter and return (ok, reason).

    Memory discipline: a 350 KB tape exec'd in-process leaks ~150 MB that
    Python does not return to the OS, and loading the whole panel that way
    OOM-killed the box. One subprocess per candidate bounds peak memory to a
    single tape; the child dies and reclaims everything.
    """
    import subprocess
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.path.join(ROOT, "src"))
    try:
        p = subprocess.run([sys.executable, "-m",
                            "kaggriculture.measure.crown_panel",
                            "--vet-one", path],
                           capture_output=True, text=True, timeout=150,
                           encoding="utf-8", errors="replace", cwd=ROOT,
                           env=env)
    except subprocess.TimeoutExpired:
        # A referee that cannot play a handful of episodes in 150s is too slow
        # to be one (and, on this box, a memory risk) -- reject, don't hang.
        return False, "VET timed out (>150s) -- too slow for a referee"
    for line in (p.stdout or "").splitlines():
        if line.startswith("VET_RESULT\t"):
            _tag, ok, reason = (line.split("\t", 2) + ["", ""])[:3]
            return ok == "OK", reason
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    return False, f"no VET_RESULT (exit {p.returncode}): {tail[-1][:80] if tail else ''}"


# -------------------------------------------------------------- band splits --

def _assign_splits(members, holdout_n=HOLDOUT_PER_BAND):
    """Stamp `selection`/`holdout` within one band, deterministically.

    Reserve up to `holdout_n` referees (only when the band keeps >= 3, so
    selection never drops below 2), picking the ones nearest the band's rating
    median so the reserve spans the band rather than skimming an extreme.
    """
    members = sorted(members, key=lambda m: (m["rating"], m["name"]))
    for m in members:
        m["split"] = "selection"
    # A tape_desync_risk referee is a known-approximate stand-in for a frontier
    # agent, so it must NEVER sit in the holdout: a "regression" against it is
    # not an overfit signal, it is the tape drifting. Only holdout-eligible
    # (non-desync) members can be reserved.
    eligible = [i for i, m in enumerate(members)
                if not m.get("tape_desync_risk")]
    if len(eligible) < 3 or holdout_n <= 0:
        return members
    ratings = sorted(members[i]["rating"] for i in eligible)
    med = ratings[len(ratings) // 2]
    order = sorted(eligible,
                   key=lambda i: (abs(members[i]["rating"] - med),
                                  members[i]["name"]))
    for i in order[:min(holdout_n, len(eligible) - 2)]:
        members[i]["split"] = "holdout"
    return members


# ------------------------------------------------------------------- build --

def build(per_band=PER_BAND, holdout_n=HOLDOUT_PER_BAND, bands=None,
          merge=True):
    """Build (or extend) the banded panel. VET runs one candidate per
    subprocess so peak memory is a single tape.

    `bands` limits the build to a subset (so each can be a small foreground
    job); `merge` keeps already-built bands from the existing manifest, so
    `--bands 2500-2700` extends the panel instead of replacing it.
    """
    os.makedirs(os.path.dirname(PANEL_JSON), exist_ok=True)
    target_bands = bands or BAND_NAMES
    rmap, lb_src = rating_map()
    if not rmap:
        print("WARNING: no leaderboard -- only explicitly/nominally rated "
              "referees will be placed")
    cands = _attribute(rmap) + _desync_candidates()
    by_band = {b: [] for b in BAND_NAMES}
    # Prefer HIGH-confidence referees (real reactive code) over the lower-
    # confidence replay tapes WITHIN a band: a desync-risk tape is only used to
    # top a band up once the reactive/rendered candidates are exhausted (operator
    # 2026-09-18: fall back to replay tapes only if a band is still thin). So a
    # band sorts non-desync first (by rating), then desync (by rating).
    for c in sorted(cands, key=lambda c: (c.get("desync", False), -c["rating"])):
        b = band_of(c["rating"])
        if b:
            by_band[b].append(c)

    prev = load() if merge else None
    panel = dict((prev or {}).get("bands") or {}) if merge else {}
    for band in target_bands:
        kept = []
        for c in by_band[band]:
            if len(kept) >= per_band:
                break
            ok, why = vet_subprocess(c["path"])
            tag = "OK " if ok else "REJECT"
            print(f"  [{band:9}] {tag} {c['name'][:44]:44} "
                  f"r={c['rating']:.0f} {c['source']:16} {why}", flush=True)
            if not ok:
                continue
            bdir = os.path.join(REF_DIR,
                                band.replace("<", "lt").replace("+", "plus"))
            os.makedirs(bdir, exist_ok=True)
            frozen = os.path.join(bdir, f"{c['name']}.py")
            shutil.copyfile(c["path"], frozen)
            kept.append({"name": c["name"], "rating": c["rating"],
                         "source": c["source"], "token": c.get("token"),
                         "tape": os.path.relpath(frozen, ROOT),
                         "origin": os.path.relpath(c["path"], ROOT),
                         "tape_desync_risk": bool(c.get("desync")),
                         "vet": why})
        _assign_splits(kept, holdout_n)
        panel[band] = kept

    man = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "lb_source": (os.path.relpath(lb_src, ROOT) if lb_src else None),
           "per_band": per_band, "holdout_per_band": holdout_n,
           "vet": {"opponent": os.path.relpath(VET_OPP, ROOT),
                   "seeds": list(VET_SEEDS),
                   "rule": "loads + 720-step serve episode, no crash, "
                           "bank-for-bank reproducible"},
           "band_names": BAND_NAMES, "bands": panel}
    json.dump(man, open(PANEL_JSON, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"\ncrown panel -> {os.path.relpath(PANEL_JSON, ROOT)}")
    for band in BAND_NAMES:
        ms = panel[band]
        sel = sum(1 for m in ms if m["split"] == "selection")
        hold = sum(1 for m in ms if m["split"] == "holdout")
        print(f"  {band:10}: {len(ms):2} referee(s) "
              f"({sel} selection / {hold} held-out)"
              + ("  <-- THIN" if len(ms) < 3 else ""))
    return man


# ------------------------------------------------------------------- loader --

def load():
    if not os.path.exists(PANEL_JSON):
        return None
    try:
        return json.load(open(PANEL_JSON, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def referees(split="selection", bands=None, man=None, include_desync_risk=True):
    """{band: [absolute referee path, ...]} for the requested split.

    split=None returns every referee. Missing frozen files are skipped. Only
    bands with at least one referee appear. `include_desync_risk=False` drops
    the lower-confidence tape referees (the top-100 replay tapes that DESYNC in
    a new world), so a caller can score the crown on the HIGH-confidence
    (reactive) referees only and treat the desync band as advisory.
    """
    man = man or load()
    if not man:
        return {}
    out = {}
    for band in (bands or man.get("band_names", BAND_NAMES)):
        paths = []
        for m in (man.get("bands") or {}).get(band, []):
            if split is not None and m.get("split") != split:
                continue
            if not include_desync_risk and m.get("tape_desync_risk"):
                continue
            p = os.path.join(ROOT, m["tape"])
            if os.path.exists(p):
                paths.append(p)
        if paths:
            out[band] = paths
    return out


def holdout_referees(bands=None, man=None):
    return referees(split="holdout", bands=bands, man=man)


def protected_prefixes(man=None):
    """Referee basenames the referee-power pruner must never drop (they hold a
    rating band). One prefix per referee; passed as `protected=` to
    `referee_power.analyse`."""
    man = man or load()
    if not man:
        return ()
    out = []
    for band in man.get("band_names", BAND_NAMES):
        for m in (man.get("bands") or {}).get(band, []):
            out.append(os.path.basename(m["tape"]))
    return tuple(out)


def prune(man=None):
    """Report non-discriminating panel referees from accumulated evidence,
    NEVER dropping a whole band (C1.2 -- protect the strata).

    Uses `referee_power` over whatever per-candidate evidence exists. Refuses
    (as referee_power does) on thin evidence, and guarantees each band keeps
    at least one referee. Report-only: returns (droppable, kept_by_band); the
    manifest is not rewritten here.
    """
    man = man or load()
    if not man:
        return [], {}
    try:
        import kaggriculture.measure.referee_power as RP
        table = RP.gather()
    except Exception:                                              # noqa: BLE001
        return [], {}
    if not table:
        return [], {}
    name_band = {}
    for band in man.get("band_names", BAND_NAMES):
        for m in (man.get("bands") or {}).get(band, []):
            name_band[os.path.basename(m["tape"])] = band
    _rows, keep, drop = RP.analyse(table, protected=protected_prefixes(man))
    if drop and drop[0][0] == "__all__":
        return [], {b: [os.path.basename(m["tape"])
                        for m in (man.get("bands") or {}).get(b, [])]
                    for b in man.get("band_names", BAND_NAMES)}
    drop_names = {os.path.basename(x) for x, _ in drop}
    droppable = []
    per_band_left = {b: [] for b in man.get("band_names", BAND_NAMES)}
    for band in man.get("band_names", BAND_NAMES):
        refs = [os.path.basename(m["tape"])
                for m in (man.get("bands") or {}).get(band, [])]
        keepers = [r for r in refs if r not in drop_names]
        if not keepers and refs:
            keepers = [refs[0]]          # never empty a band
        per_band_left[band] = keepers
        droppable += [r for r in refs if r not in keepers]
    return droppable, per_band_left


def ladder_band_winrate(rmap=None):
    """C3.1: our ACTUAL per-band win rate on the ladder, from data/ourgames.

    Joins each of our recorded ladder games to the opponent's CURRENT LB rating
    (by team name) and aggregates win/draw/loss per rating band -- the live
    ground truth the panel's per-band prediction is checked against. Returns
    {"available": bool, "note": str, "bands": {band: {w,d,l,n,winrate}}}.

    Report-only, degrade-safe: with no ourgames index (data/ is regenerable and
    empty on a fresh checkout) it returns available=False with the reason, so a
    caller can print "needs live games" rather than crash.
    """
    idx_p = os.path.join(ROOT, "data", "ourgames", "index.json")
    if not os.path.exists(idx_p):
        return {"available": False,
                "note": "no data/ourgames/index.json -- needs live ladder "
                        "games (run kaggriculture.data.ourgames --all)",
                "bands": {}}
    try:
        games = (json.load(open(idx_p, encoding="utf-8")).get("games") or {})
    except (OSError, ValueError) as exc:
        return {"available": False, "note": f"ourgames unreadable ({exc})",
                "bands": {}}
    if rmap is None:
        rmap, _ = rating_map()
    bands = {b: {"w": 0, "d": 0, "l": 0} for b in BAND_NAMES}
    matched = 0
    for g in games.values():
        opp = str(g.get("opponent") or "").strip().lower()
        hit = rmap.get(opp)
        if not hit:
            continue
        b = band_of(hit[0])
        if not b:
            continue
        matched += 1
        if g.get("tied"):
            bands[b]["d"] += 1
        elif g.get("won"):
            bands[b]["w"] += 1
        else:
            bands[b]["l"] += 1
    for b, c in bands.items():
        n = c["w"] + c["d"] + c["l"]
        c["n"] = n
        c["winrate"] = ((c["w"] + 0.5 * c["d"]) / n) if n else None
    note = (f"{matched}/{len(games)} games matched to an LB-rated opponent"
            if matched else "no games matched an LB-rated opponent "
                            "(opponent names may not be on the current LB)")
    return {"available": matched > 0, "note": note, "bands": bands}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--per-band", type=int, default=PER_BAND)
    ap.add_argument("--holdout-per-band", type=int, default=HOLDOUT_PER_BAND)
    ap.add_argument("--ladder-error", action="store_true",
                    help="C3.1: our actual per-band ladder win rate from "
                         "data/ourgames (needs live games)")
    ap.add_argument("--bands", default=None,
                    help="comma-separated subset to build (extends the "
                         "manifest); default all bands")
    ap.add_argument("--vet-one", default=None,
                    help="internal: VET one agent in this fresh interpreter "
                         "and print VET_RESULT (keeps peak memory to one tape)")
    args = ap.parse_args()
    if args.vet_one:
        ok, why = vet(args.vet_one)
        print(f"VET_RESULT\t{'OK' if ok else 'REJECT'}\t{why}", flush=True)
        return 0
    if args.build:
        bands = ([b.strip() for b in args.bands.split(",")]
                 if args.bands else None)
        build(args.per_band, args.holdout_per_band, bands=bands)
        return 0
    if args.ladder_error:
        r = ladder_band_winrate()
        print(f"ladder per-band win rate: {r['note']}")
        for b in BAND_NAMES:
            c = r["bands"].get(b, {})
            wr = c.get("winrate")
            print(f"  {b:10}: " + (f"{wr:.3f} over {c.get('n')} games"
                                   if wr is not None else "no games"))
        return 0 if r["available"] else 1
    man = load()
    if not man:
        print("no crown panel built yet -- run --build")
        return 1
    print(f"crown panel built {man['built']} (lb {man.get('lb_source')})")
    for band in man.get("band_names", BAND_NAMES):
        ms = (man.get("bands") or {}).get(band, [])
        print(f"  {band:10}: " + ", ".join(
            f"{m['name'][:24]}({m['rating']:.0f},{m['split'][:3]})" for m in ms)
            or f"  {band:10}: (empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
