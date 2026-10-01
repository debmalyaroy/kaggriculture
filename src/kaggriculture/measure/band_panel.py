"""Rating-band panels with regime tagging (Tier 0 instrument, 2026-09-03).

Four panels. Three are rebuilt from the LIVE leaderboard (full CSV download --
the CLI page size caps at 200, so `elite_panel.live_top_teams` cannot reach
rank 2000); the fourth is REACTIVE:

  sub2000  teams rated 1700-2000 with >= 20 games in our index ("settled")
  mid      teams rated 2000-2500
  top100   the existing elite panel (src/elite_panel.py, top 100)
  reactive live agents (data/gauntlet/*.py + our own current bandit) that
           actually PLAY. Measured 2026-09-03: worlds against them sit at an
           88.7k median shared bank / 53% sub-90k, against a ladder at
           85k / 58%. The tape bands sit at 56k / 95%. This is the only
           band whose worlds look like the ladder's.

For sub2000/mid the freshest 1.32.7 winning tape of each team is rendered
(same pattern as elite_panel.build), capped at 40 teams spread evenly across
the band. Scoring is paired cells (both seats, seeds 501..) on the Rust serve
substrate, PARALLEL (one serve_match.Serve per worker), and every cell is
tagged LOW/HIGH price regime from the MILK/STRAWBERRY/EGG/WOOL prices the
driver sees on days 3-12 (the `kagg serve` reply carries the full market each
step) -- see the regime spec below for why those four products and why the
threshold is measured, not assumed.

    python src/band_panel.py --build                       # all four bands
    python src/band_panel.py --score A.py [B.py] --seeds 1 --workers 8
    python src/band_panel.py --score A.py --bands sub2000,mid
    python src/band_panel.py --calibrate-traces            # ladder cross-check

Bars (plan-2800 section 2): sub2000 must be 100% (any loss = FAIL, listed),
mid >= 0.90, top100 >= 0.70, reactive >= 0.90. Output JSON:
.local/band_panel/last_scores.json.

THREE STANDING RULES, added 2026-09-03 after the panels anti-predicted
(`docs/history/instrument-repair-2026-09-03.md`):

1. MARGIN IS REPORTED EVERYWHERE. Cells saturate (the whole v22 chassis moved
   4 of 652 cells); the median paired margin per game does not. Every band
   prints cells, ratio, `clean_ratio`, the median paired margin, and -- when
   two agents are scored -- the count of cells where the MARGIN moved but the
   winner did not. That count is the resolution the cell count threw away.
2. HELD-OUT IS THE DECISION NUMBER. Every panel is split into a SELECTION half
   and a HELD-OUT half, stratified by rating, recorded in the manifest, and
   scored separately. `v44_single` won its selection panel 77-11 and then lost
   84 of 86 discordant held-out cells; selecting and scoring on the same cells
   is worse than useless. This is the DEFAULT path, not a flag.
3. WORLD DISTRIBUTION IS REPORTED AGAINST THE LADDER. A panel whose worlds do
   not look like the ladder's is answering a different question confidently.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

# Team names are user-supplied and routinely non-ASCII (CJK, accents). The
# Windows console is cp1252, so ANY print of a team name raised
# UnicodeEncodeError and killed a completed scoring run at the report step
# (2026-09-03). Re-encode stdout/stderr instead of stripping names.
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                           # noqa: PERF203
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

BAND_DIR = os.path.join(ROOT, ".local", "band_panel")
LB_DIR = os.path.join(BAND_DIR, "lb")
REGIME_FILE = os.path.join(BAND_DIR, "regime.json")
LAST_SCORES = os.path.join(BAND_DIR, "last_scores.json")

BANDS = {"sub2000": (1700.0, 2000.0), "mid": (2000.0, 2500.0), "top100": None,
         "reactive": None}
BAND_ORDER = ("sub2000", "mid", "top100", "reactive")
TAPE_BANDS = ("sub2000", "mid", "top100")
CAP = 40
MIN_GAMES = {"sub2000": 20, "mid": 0}
BARS = {"sub2000": 1.0, "mid": 0.90, "top100": 0.70, "reactive": 0.90}
# An opponent that ends on its 3,000 starting money never played a game.
DEAD_OPPONENT_BANK = 3000
# Our own team on the ladder -- never a panel opponent for ourselves.
OUR_TEAMS = {"debmalya"}

# ------------------------------------------------------------ reactive panel
#
# The tape bands replay an open-loop schedule compiled for a DIFFERENT world,
# and the worlds that result are POOR: 56k median shared bank, 95% sub-90k,
# against a ladder at 85k / 58%. Reactive opponents fix it (88.7k / 53%).
#
# The mechanism is NOT "tapes fail to unlock shops" -- that theory was checked
# on 2026-09-03 and is FALSE. Shop unlocks are identical in every band (first
# unlock step 72, six unlocked by step 432, zero cells with none). What
# actually happens is that BOTH banks collapse: in the SAME seeds, our own
# agent banks a 61k median against a tape and 90.5k against a reactive
# opponent, while the opponent banks 51-56k vs 93.4k. A mistimed open-loop
# schedule does not trade against us, and a world with only one real trader in
# it is a poorer world for both farms. So the panel was not measuring a weaker
# opponent; it was measuring a different economy.
#
# These opponents are the roster `serve_gate.py` uses plus our own current
# bandit, FROZEN as copies here so another lane cannot change them mid-run.
REACTIVE_OPPS = [
    "data/gauntlet/aurax7_kaggriculture_shop_router_reactive_v5.py",
    "data/gauntlet/lynnsakurai_farming_score_v4_a_better_shop.py",
    "data/gauntlet/notebookfea7d71a35.py",
    "data/gauntlet/kaito_v48.py",
    "data/gauntlet/kaito_v43.py",
    "data/gauntlet/adaptive_replay.py",
    "data/gauntlet/boatlee_v16_rc5_r5a_high_score_8c_4s_recovery.py",
    "data/gauntlet/pub_rayk_c94.py",
    "data/gauntlet/reactive_router.py",
    "data/gauntlet/y3uanm_kaggriculture_market_impact_router_v4.py",
    "data/gauntlet/moon.py",
    "data/gauntlet/munib.py",
    "agents/v56y_trackp.py",
    # --- 2026-09-16: distinct public frontier agents, extracted from Kaggle
    #     notebooks (decoded to standalone agent(obs) files). The whole public
    #     field is ONE architecture (shop-dispatch fixed-tape + reactive sell),
    #     so these are the DISTINCT lineages/micro-variants, not near-clones:
    "data/gauntlet/pub_tschinkel_cart_router.py",     # 5 specialist tapes, learned CART router (the upstream original)
    "data/gauntlet/pub_prvsiyan_yhay81_13tape.py",    # yhay81 Shop Router 0909: 13 generalist tapes + aurax7 sells
    "data/gauntlet/pub_flexon_v45.py",                # "frankenstein" V45 (==reyhan): assembled meta + bounded terminal search
    "data/gauntlet/pub_tetsutani_mirror.py",          # V45 + mirror_reorder sell hill-climb
    "data/gauntlet/pub_nathanjacob_anticlone.py",     # V45 + EXP283 anti-clone race + symmetric-wheat open
    "data/gauntlet/pub_hesoponyo_transformer.py",     # only genuinely-learned seat: BC+PPO entity-token Transformer (pure-python fwd)
    "data/gauntlet/pub_djamila_v16_route.py",         # base V16 route (GNN residual notebook's baked base policy)
]

# What the LADDER's worlds look like, from 8,000 real per-turn traces
# (calibrate_from_traces, 2026-09-03). Any panel is reported against this.
LADDER_WORLD = {"median_world_bank": 85000.0, "sub90k_share": 0.58}

# ------------------------------------------------------------ holdout split
#
# `dual_bar_miner` and every hand-run sweep SELECT on these panels. Scoring the
# survivor on the same cells reads back the selection, not the strength:
# v44_single won 77-11 on the panel that chose it and then lost 84 of 86
# discordant HELD-OUT cells (p < 0.0001). So the panel carries its own split.
SPLIT_FRAC = 0.5
SPLITS = ("selection", "holdout")

# --------------------------------------------------------------- regime spec
#
# A "world" is LOW-price when the town pays badly for the products a farm
# actually banks. The plan's first cut -- min(MILK, STRAWBERRY) median ratio --
# was measured on 8,000 ladder traces (.local/band_panel/calib_study.py,
# 2026-09-03) and it BARELY SEPARATES ANYTHING:
#
#   statistic              window   AUC(sub-90k world)  Cohen d   median bank LOW/HIGH
#   min(milk, straw)       d9-12          0.738           0.82      74.7k / 92.2k
#   min(milk, straw)       d3-5      degenerate: 4 distinct values, 62% below the
#                                    lowest usable cut -- no 35% cut exists
#   mean(milk,straw,egg,wool) d9-12       0.830           1.33      67.4k / 95.0k
#   mean(milk,straw,egg,wool) d6-8        0.805           1.27      67.3k / 94.1k
#   mean(milk,straw,egg,wool) d3-5        0.734           1.36      64.5k / 92.4k
#
# So the fix is the STATISTIC, not only the window: widening from two products
# to four (the four the shops consume: MILK, STRAWBERRY, EGG, WOOL) lifts the
# d9-12 discrimination from d=0.82 to d=1.33 and -- the part that matters for
# an agent that has to ACT -- makes the DAY 3-5 window usable at d=1.36 /
# AUC 0.734, separating 64.5k worlds from 92.4k worlds. Early prices are a
# shop-unlock indicator (nobody has milk to sell yet), which is exactly the
# branch signal the plan's tier-2 shop-branch table wants; it is coarse
# (5-7 distinct values on the ladder, so the achievable LOW share quantises to
# ~26%, not 35%) but it is not degenerate.
#
# Three windows are recorded per cell: d3-5 (EARLY, actionable), d6-8, and
# d9-12 (PRIMARY -- best late discrimination, the tag the win% readout uses).
BASE = {"MILK": 160.0, "STRAWBERRY": 120.0, "EGG": 50.0, "WOOL": 200.0}
STAT_PRODUCTS = ("MILK", "STRAWBERRY", "EGG", "WOOL")
WINDOWS = {"d3-5": (3, 5), "d6-8": (6, 8), "d9-12": (9, 12)}
PRIMARY = "d9-12"
EARLY = "d3-5"
TRACK_DAYS = (min(w[0] for w in WINDOWS.values()),
              max(w[1] for w in WINDOWS.values()))
LOW_SHARE = 0.35          # calibration target: ~35% of cells LOW
POOR_WORLD = 90000.0      # "sub-90k world" -- the band the top-10 edge lives in


# ------------------------------------- splits, margins, world distribution --
# Everything in this section is PURE: no engine, no files, no network. It is
# what tests/test_band_panel.py asserts, so a change here that breaks the
# instrument's arithmetic fails a 0.2s test instead of a 20-minute sweep.

_SIG_CACHE = {}


def file_sig(path):
    """SHA-1 of a file's bytes, cached. Identity of an AGENT, not of a path."""
    import hashlib
    p = os.path.abspath(path)
    if p not in _SIG_CACHE:
        try:
            with open(p, "rb") as fh:
                _SIG_CACHE[p] = hashlib.sha1(fh.read()).hexdigest()
        except OSError:
            _SIG_CACHE[p] = f"missing:{p}"
    return _SIG_CACHE[p]


def split_key(team):
    """Stable identity of a panel opponent, for a deterministic split."""
    return str(team.get("rid") or team.get("tape") or team.get("team") or "")


def assign_splits(teams, frac=SPLIT_FRAC):
    """Stamp every team `selection` or `holdout`, stratified by rating.

    Deterministic and RNG-free: teams are ordered by (-lb_score, identity) and
    dealt with a running accumulator, so each half SPANS the band (never "the
    top 20 select, the bottom 20 score", which would confound the split with
    strength) and the assignment survives a rebuild that adds or drops teams.
    `frac` is the held-out share. Mutates and returns `teams`.
    """
    order = sorted(range(len(teams)),
                   key=lambda i: (-(teams[i].get("lb_score") or 0.0),
                                  split_key(teams[i])))
    acc = 0.0
    for i in order:
        acc += frac
        if acc >= 1.0 - 1e-9:
            teams[i]["split"] = "holdout"
            acc -= 1.0
        else:
            teams[i]["split"] = "selection"
    return teams


def ensure_splits(teams, frac=SPLIT_FRAC):
    """Back-fill splits into a manifest written before splits existed."""
    if any(t.get("split") not in SPLITS for t in teams):
        assign_splits(teams, frac)
        return True
    return False


def seed_splits(seeds, frac=SPLIT_FRAC):
    """{seed: selection|holdout}, alternating over sorted seeds.

    A single-seed run has NO held-out seed -- every seed is a selection seed
    and the team axis carries the whole holdout. Say that rather than
    pretending one seed can be split.
    """
    ss = sorted(set(seeds))
    if len(ss) < 2:
        return {s: "selection" for s in ss}
    out, acc = {}, 0.0
    for s in ss:
        acc += frac
        if acc >= 1.0 - 1e-9:
            out[s] = "holdout"
            acc -= 1.0
        else:
            out[s] = "selection"
    return out


def cell_split(team_split, seed_split_, use_seeds=False):
    """Which half a cell belongs to.

    Default: the TEAM axis decides -- that is the axis selection runs over.
    With `use_seeds` the two axes must agree, so a held-out cell shares
    neither its opponent nor its seed with anything selection ever saw; cells
    that straddle are `mixed` and belong to neither headline.
    """
    if not use_seeds:
        return team_split
    if team_split == seed_split_:
        return team_split
    return "mixed"


def summarise_cells(cells):
    """W/L/D, ratio, dead-opponent-clean record, and MARGIN, for cell dicts.

    `cells` = iterable of dicts with `own` and `opp`. The margin fields are the
    point of this function: cell counts saturate long before the margin does,
    so a band that reads 0.988 for every arm can still be moving thousands of
    dollars a game, and that is the only resolution left to us.
    """
    cs = [c for c in cells if c.get("own") is not None
          and c.get("opp") is not None]
    w = sum(1 for c in cs if c["own"] > c["opp"])
    l = sum(1 for c in cs if c["own"] < c["opp"])
    d = len(cs) - w - l
    live = [c for c in cs if c["opp"] > DEAD_OPPONENT_BANK]
    cw = sum(1 for c in live if c["own"] > c["opp"])
    cl = len(live) - cw            # draws count as NOT-a-win, as before
    mar = [c["own"] - c["opp"] for c in cs]
    lmar = [c["own"] - c["opp"] for c in live]
    return {"n": len(cs), "cells": [w, l, d],
            "ratio": (w + 0.5 * d) / len(cs) if cs else None,
            "dead_opponent_cells": len(cs) - len(live),
            "clean_cells": [cw, cl],
            "clean_ratio": cw / len(live) if live else None,
            "median_margin": statistics.median(mar) if mar else None,
            "mean_margin": statistics.fmean(mar) if mar else None,
            "clean_median_margin": statistics.median(lmar) if lmar else None,
            "median_own": statistics.median([c["own"] for c in cs]) if cs
            else None,
            "median_opp": statistics.median([c["opp"] for c in cs]) if cs
            else None}


def fmt_summary(s):
    """One line for a summarise_cells dict. None-safe."""
    if not s or not s["n"]:
        return "no cells"
    w, l, d = s["cells"]
    out = f"{w}-{l}" + (f"-{d}d" if d else "") + f" ({s['ratio']:.3f})"
    if s["clean_ratio"] is None:
        out += "  clean n/a"
    else:
        out += (f"  clean {s['clean_cells'][0]}-{s['clean_cells'][1]}"
                f" ({s['clean_ratio']:.3f})")
    out += f"  med margin {s['median_margin']:+,.0f}"
    if s["clean_median_margin"] is not None and s["dead_opponent_cells"]:
        out += f" (clean {s['clean_median_margin']:+,.0f})"
    return out


def sign_test(pos, neg):
    """Two-sided exact binomial p on `pos` vs `neg` non-zero differences."""
    import math
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def dedupe_mirrored(keys, *maps):
    """Drop the seat-1 key of any world that mirrored EXACTLY in every map.

    Both seats of one (opponent, seed) are the SAME world played from opposite
    sides. Both agents are deterministic and the engine's two farms start
    identical, so the swap usually reproduces the cell to the dollar -- and
    then counting both doubles the apparent sample size and halves every
    p-value. Keys are (band, team_index, seed, seat) and must be sorted, so
    the two seats of a world are adjacent. Returns the keys to keep.
    """
    keep, seen = [], {}
    for k in keys:
        base = k[:-1]
        sig = tuple((m[k]["own"], m[k]["opp"]) for m in maps)
        if base in seen and seen[base] == sig:
            continue
        seen[base] = sig
        keep.append(k)
    return keep


def compare_cells(a_cells, b_cells):
    """Paired winner AND margin comparison of two agents over the same cells.

    `a_cells`/`b_cells` are positionally aligned (same opponent, seed, seat).
    Returns the McNemar verdict on winners (`win_metric.paired_test`) plus the
    margin channel: how many cells moved in dollars WITHOUT changing the
    winner, the median paired margin delta, and a sign test on it. When the
    winner channel is saturated the margin channel is the only signal left.
    """
    import kaggriculture.measure.win_metric as WM
    n = min(len(a_cells), len(b_cells))
    a, b = a_cells[:n], b_cells[:n]
    sa = [WM.score(c["own"], c["opp"]) for c in a]
    sb = [WM.score(c["own"], c["opp"]) for c in b]
    t = WM.paired_test(sa, sb)
    dm = [(x["own"] - x["opp"]) - (y["own"] - y["opp"]) for x, y in zip(a, b)]
    moved = sum(1 for dd, x, y in zip(dm, sa, sb) if dd != 0 and x == y)
    pos = sum(1 for dd in dm if dd > 0)
    neg = sum(1 for dd in dm if dd < 0)
    t.update({"margin_moved_winner_same": moved,
              "winner_moved": sum(1 for x, y in zip(sa, sb) if x != y),
              "identical_cells": sum(1 for dd in dm if dd == 0),
              "median_margin_delta": statistics.median(dm) if dm else None,
              "mean_margin_delta": statistics.fmean(dm) if dm else None,
              "margin_better_a": pos, "margin_better_b": neg,
              "margin_sign_p": sign_test(pos, neg)})
    return t


def world_stats(cells):
    """Distribution of the SHARED world bank, against the ladder reference.

    The shared bank (own+opp)/2 is the world's wealth: it says whether the
    town's demand economy ever fired. A panel whose worlds sit at 50k when the
    ladder sits at 85k is not a weaker panel, it is a DIFFERENT GAME.
    """
    ws = [(c["own"] + c["opp"]) / 2.0 for c in cells
          if c.get("own") is not None and c.get("opp") is not None]
    if not ws:
        return {"n": 0}
    sub = sum(1 for x in ws if x < POOR_WORLD) / len(ws)
    med = statistics.median(ws)
    return {"n": len(ws), "median_world_bank": round(med),
            "mean_world_bank": round(statistics.fmean(ws)),
            "sub90k_share": sub,
            "ladder_median_world_bank": LADDER_WORLD["median_world_bank"],
            "ladder_sub90k_share": LADDER_WORLD["sub90k_share"],
            "median_gap": round(med - LADDER_WORLD["median_world_bank"]),
            "sub90k_gap": round(sub - LADDER_WORLD["sub90k_share"], 4),
            "ladder_like": abs(sub - LADDER_WORLD["sub90k_share"]) <= 0.12
            and abs(med - LADDER_WORLD["median_world_bank"]) <= 20000.0}


def fmt_world(s):
    if not s.get("n"):
        return "no cells"
    return (f"median shared bank {s['median_world_bank']:,} "
            f"({s['median_gap']:+,} vs ladder), "
            f"{100 * s['sub90k_share']:.0f}% sub-{POOR_WORLD / 1000:.0f}k "
            f"({100 * s['sub90k_gap']:+.0f}pp vs ladder 58%) -> "
            f"{'LADDER-LIKE' if s['ladder_like'] else 'NOT ladder-like'}")


def shop_stats(cells):
    """Shop-unlock activity in a set of cells.

    Built to test the standing theory that tape opponents make poor worlds by
    failing to unlock shops. RUN 2026-09-03: the theory is FALSE. Every band
    unlocks its first shop at step 72 and six by step 432, with zero cells
    unlocking none -- tape and reactive alike. Keep the readout anyway: it is
    the cheap check that a future panel change has not broken the town, and it
    is the reason we now blame the economy rather than the shop gate.
    """
    ns = [len(c.get("shops") or []) for c in cells]
    firsts = [c["shops"][0][0] for c in cells if c.get("shops")]
    if not ns:
        return {"n": 0}
    return {"n": len(ns), "median_shops": statistics.median(ns),
            "mean_shops": round(statistics.fmean(ns), 2),
            "share_no_shop": sum(1 for x in ns if x == 0) / len(ns),
            "median_first_unlock_step": (statistics.median(firsts)
                                         if firsts else None)}


def spearman(a, b):
    """Spearman rank correlation of two equal-length sequences (ties averaged).

    Used to ask the only question that matters about a panel: does it rank the
    agents whose LIVE outcome we already know? Returns None below n=3.
    """
    n = len(a)
    if n != len(b) or n < 3:
        return None

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(list(a)), ranks(list(b))
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if not da or not db:
        return None
    return num / (da * db)


# ----------------------------------------------------------------- leaderboard

def download_leaderboard(dest=LB_DIR, timeout=300):
    """Full leaderboard CSV via `kaggle competitions leaderboard --download`.

    Returns (rows, source) where rows = [(name, score, rank)]. Falls back to
    the freshest CSV already on disk (dest, then .local/lb) with a WARNING so a
    network blip never silently builds a panel from nothing.
    """
    os.makedirs(dest, exist_ok=True)
    src = None
    try:
        p = subprocess.run(["kaggle", "competitions", "leaderboard",
                            "kaggriculture", "--download", "-p", dest],
                           capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        z = os.path.join(dest, "kaggriculture.zip")
        if p.returncode == 0 and os.path.exists(z):
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
                if names:
                    # Member names carry ':' (a timestamp), illegal on
                    # Windows -- write under a sanitised name ourselves.
                    safe = os.path.basename(names[0]).replace(":", "_")
                    src = os.path.join(dest, safe)
                    with zf.open(names[0]) as fi, open(src, "wb") as fo:
                        fo.write(fi.read())
            os.remove(z)
    except (OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"WARNING leaderboard download failed: {exc}")
    if src is None:
        cands = sorted(glob.glob(os.path.join(dest, "*.csv"))
                       + glob.glob(os.path.join(ROOT, ".local", "lb", "*.csv")),
                       key=os.path.getmtime)
        if not cands:
            raise SystemExit("no leaderboard CSV available -- refuse")
        src = cands[-1]
        print(f"WARNING using leaderboard on disk: {src}")
    return parse_leaderboard_csv(src), src


def parse_leaderboard_csv(path):
    """[(TeamName, Score, Rank)] from the Kaggle public-leaderboard CSV."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), 1):
            try:
                score = float(row["Score"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                rank = int(row.get("Rank") or i)
            except ValueError:
                rank = i
            rows.append((row.get("TeamName", "").strip(), score, rank))
    return rows


# ----------------------------------------------------------------- index side

def index_stats(idx):
    """Per team (lowercased): games in our index and the freshest 1.32.7 win.

    'Freshest' = highest episode number, then bank -- the elite_panel rule.
    """
    from kaggriculture.measure.elite_panel import epnum
    games, best = {}, {}
    for rid, m in idx.items():
        t = str(m.get("team", "")).strip().lower()
        if not t:
            continue
        games[t] = games.get(t, 0) + 1
        if m.get("won") and m.get("engine") == "1.32.7":
            keyv = (epnum(m), float(m.get("bank") or 0))
            if t not in best or keyv > best[t][0]:
                best[t] = (keyv, rid)
    return games, best


def spread(items, cap):
    """Up to `cap` items evenly spaced over a rating-sorted list."""
    if len(items) <= cap:
        return list(items)
    n = len(items)
    picks = sorted({int(round(i * (n - 1) / (cap - 1))) for i in range(cap)})
    return [items[i] for i in picks]


def select_band(lb_rows, games, best, lo, hi, min_games, cap=CAP,
                exclude=OUR_TEAMS):
    """Candidates in [lo, hi) with a renderable 1.32.7 win, spread to `cap`.

    Returns (picked, n_candidates); each pick is a dict with team, lb_score,
    lb_rank, games, rid, bank.
    """
    cands = []
    for name, score, rank in lb_rows:
        key = name.lower()
        if key in exclude or not (lo <= score < hi):
            continue
        if games.get(key, 0) < min_games or key not in best:
            continue
        cands.append({"team": name, "lb_score": score, "lb_rank": rank,
                      "games": games[key], "rid": best[key][1],
                      "bank": best[key][0][1]})
    cands.sort(key=lambda d: -d["lb_score"])
    return spread(cands, cap), len(cands)


def band_manifest_path(band):
    return os.path.join(BAND_DIR, band, "manifest.json")


def reactive_teams(opps=REACTIVE_OPPS, dest=None):
    """Panel entries for the reactive roster; missing files are skipped loudly.

    Each opponent is COPIED into the panel directory. `agents/` belongs to
    another lane and `data/gauntlet/` is regenerable, so referencing them in
    place would let a panel opponent change under a running comparison -- the
    exact instrument instability this repair exists to remove. The manifest
    records the source path and the copy's mtime so staleness is visible.
    """
    dest = dest or os.path.join(BAND_DIR, "reactive")
    os.makedirs(dest, exist_ok=True)
    teams = []
    for rel in opps:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"WARNING reactive opponent missing, skipped: {rel}")
            continue
        name = os.path.splitext(os.path.basename(rel))[0]
        frozen = os.path.join(dest, f"{name}.py")
        with open(p, "rb") as fi, open(frozen, "wb") as fo:
            fo.write(fi.read())
        teams.append({"team": name, "lb_score": None, "lb_rank": None,
                      "games": None, "rid": name, "bank": None,
                      "tape": frozen, "kind": "reactive", "source": rel,
                      "frozen_from_mtime": time.strftime(
                          "%Y-%m-%dT%H:%M:%S",
                          time.localtime(os.path.getmtime(p)))})
    return teams


def build(bands=BAND_ORDER, cap=CAP, lb=None, frac=SPLIT_FRAC):
    import kaggriculture.data.routes as R
    import kaggriculture.train.train_arms as TA
    import kaggriculture.measure.elite_panel as EP
    os.makedirs(BAND_DIR, exist_ok=True)
    out = {}
    if any(b in ("sub2000", "mid") for b in bands):
        lb_rows, lb_src = lb if lb is not None else download_leaderboard()
        if len(lb_rows) < 1000:
            raise SystemExit(f"leaderboard too thin ({len(lb_rows)}) -- refuse")
        games, best = index_stats(R.load_index()["routes"])
    for band in bands:
        bdir = os.path.join(BAND_DIR, band)
        os.makedirs(bdir, exist_ok=True)
        if band == "reactive":
            teams = reactive_teams()
            man = {"band": band, "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "source": "data/gauntlet + agents", "range": None,
                   "candidates": len(teams), "teams": teams}
        elif band == "top100":
            ep_man = os.path.join(EP.PANEL_DIR, "manifest.json")
            if not os.path.exists(ep_man) or json.load(
                    open(ep_man, encoding="utf-8")).get("top_n", 0) < 100:
                EP.build(100)
            em = json.load(open(ep_man, encoding="utf-8"))
            teams = [{"team": t["team"], "lb_score": t["lb_score"],
                      "lb_rank": None, "games": None, "rid": t["rid"],
                      "bank": t.get("bank"), "tape": t["tape"]}
                     for t in em["teams"]]
            man = {"band": band, "built": em["built"], "source": ep_man,
                   "range": None, "candidates": len(teams), "teams": teams}
        else:
            lo, hi = BANDS[band]
            picked, n_c = select_band(lb_rows, games, best, lo, hi,
                                      MIN_GAMES[band], cap)
            for t in picked:
                tape = os.path.join(bdir, f"{t['rid']}.py")
                if not os.path.exists(tape):
                    TA.render(R.load_route(t["rid"]), tape, f"{band}_{t['rid']}")
                t["tape"] = tape
            man = {"band": band, "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "source": lb_src, "range": [lo, hi],
                   "min_games": MIN_GAMES[band], "candidates": n_c,
                   "teams": picked}
        for t in man["teams"]:
            t.setdefault("kind", "reactive" if band == "reactive" else "tape")
        assign_splits(man["teams"], frac)
        man["split"] = {"frac_holdout": frac,
                        "rule": "stratified by rating, deterministic; "
                                "HELD-OUT is the decision number",
                        "selection": sum(1 for t in man["teams"]
                                         if t["split"] == "selection"),
                        "holdout": sum(1 for t in man["teams"]
                                       if t["split"] == "holdout")}
        json.dump(man, open(band_manifest_path(band), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        rng = "" if man["range"] is None else f" {man['range']}"
        print(f"panel {band:<8}{rng}: {len(man['teams'])} teams "
              f"(from {man['candidates']} candidates); split "
              f"{man['split']['selection']} selection / "
              f"{man['split']['holdout']} held-out")
        out[band] = man
    return out


# --------------------------------------------------------------------- regime

def regime_ratios(price_rows, window=WINDOWS[PRIMARY]):
    """{product: median(price/base)} over the window's turns, or {} if empty.

    `price_rows` = iterable of (day, {product: price}); `window` =
    (first_day, last_day) inclusive.
    """
    lo, hi = window
    out = {}
    for p in STAT_PRODUCTS:
        vals = [float(pr.get(p, 0.0)) / BASE[p]
                for d, pr in price_rows if lo <= d <= hi and p in pr]
        if vals:
            out[p] = statistics.median(vals)
    return out


def cell_stat(ratios):
    """The per-cell statistic the threshold cuts: mean of the four ratios.

    Returns None unless every tracked product was observed -- a partial mean
    would silently sit on a different scale than the calibration.
    """
    if not ratios or any(p not in ratios for p in STAT_PRODUCTS):
        return None
    return sum(ratios[p] for p in STAT_PRODUCTS) / len(STAT_PRODUCTS)


def regime_tag(ratios, thr):
    """LOW if the cell statistic is below `thr`; None when untagged."""
    x = cell_stat(ratios) if isinstance(ratios, dict) else ratios
    if x is None or thr is None:
        return None
    return "LOW" if x < thr else "HIGH"


def calibrate_threshold(cell_mins, share=LOW_SHARE):
    """(threshold, achieved_share): the cut whose LOW share is closest to
    `share`.

    Candidate cuts are the midpoints between consecutive DISTINCT values of
    min(milk_r, straw_r), so the rule `x < thr` is stable under ties and the
    achieved share is reported honestly when the statistic is discrete (the
    d3-5 window jumps 0% -> 62% on the ladder; no cut gives 35%).
    """
    xs = sorted(x for x in cell_mins if x is not None)
    if len(xs) < 4:
        return None, None
    n = len(xs)
    distinct = sorted(set(round(x, 6) for x in xs))
    if len(distinct) < 2:
        return None, None
    best = None
    for a, b in zip(distinct, distinct[1:]):
        cut = (a + b) / 2.0
        got = sum(1 for x in xs if x < cut) / n
        if best is None or abs(got - share) < abs(best[1] - share):
            best = (round(cut, 4), got)
    return best


def load_regime():
    if os.path.exists(REGIME_FILE):
        try:
            return json.load(open(REGIME_FILE, encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def save_regime(d):
    os.makedirs(BAND_DIR, exist_ok=True)
    json.dump(d, open(REGIME_FILE, "w", encoding="utf-8"), indent=1)


def auc(scores, labels):
    """P(score of a labelled-True case > score of a False one), ties at 0.5.

    Rank form, so it is exact and O(n log n); returns None when one class is
    empty. Used to report how well a regime statistic predicts a poor world --
    a share is not discrimination.
    """
    pairs = sorted(zip(scores, labels))
    n1 = sum(1 for _, y in pairs if y)
    n0 = len(pairs) - n1
    if not n1 or not n0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        rank_sum += avg * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cohen_d(a, b):
    """Pooled-SD standardised mean difference (b - a); None if undersized."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    sd = (((len(a) - 1) * va + (len(b) - 1) * vb)
          / (len(a) + len(b) - 2)) ** 0.5
    if not sd:
        return None
    return (statistics.mean(b) - statistics.mean(a)) / sd


def calibrate_from_traces(limit=8000, write=True):
    """Calibrate every window on the LADDER: real per-turn traces.

    Reads data/turntrace/*.npz (49-field turn_features layout, freshest
    `limit` episodes), computes each window's cell statistic (mean price/base
    over STAT_PRODUCTS) per episode, picks the 35% cut per window, and
    MEASURES the cut instead of only reporting its share: AUC for predicting a
    sub-90k world, Cohen's d and the median world bank on each side. Freezes
    the result in regime.json by default. Returns the regime dict or None.
    """
    import numpy as np
    import kaggriculture.data.turn_features as TF
    names = TF.feature_names()
    i_day = names.index("day")
    i_p = {p: names.index(f"price_{p}") for p in STAT_PRODUCTS}
    i_us, i_them = names.index("us_money"), names.index("them_money")
    files = sorted(glob.glob(os.path.join(TF.TRACE_DIR, "*.npz")),
                   key=os.path.getmtime)[-limit:]
    stats = {w: [] for w in WINDOWS}
    worlds = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=False)
            arr = z["s0"] if "s0" in z.files else None
        except Exception:                                       # noqa: BLE001
            continue
        if arr is None or len(arr) < 600:
            continue
        world = 1e5 * (float(arr[-1, i_us]) + float(arr[-1, i_them])) / 2.0
        if world <= 0:
            continue
        day = np.rint(arr[:, i_day] * 30.0).astype(int)
        row = {}
        for w, (lo, hi) in WINDOWS.items():
            sel = (day >= lo) & (day <= hi)
            row[w] = (cell_stat({p: float(np.median(arr[sel, i_p[p]]))
                                 for p in STAT_PRODUCTS}) if sel.any() else None)
        if row.get(PRIMARY) is None:
            continue
        worlds.append(world)
        for w in WINDOWS:
            stats[w].append(row[w])
    if not worlds:
        print("no traces found")
        return None
    reg = regime_from_stats(stats, f"ladder traces ({len(worlds)} episodes)",
                            worlds=worlds)
    print(f"ladder worlds n={len(worlds)} median bank "
          f"{statistics.median(worlds):.0f}; sub-{POOR_WORLD / 1000:.0f}k share "
          f"{sum(1 for x in worlds if x < POOR_WORLD) / len(worlds):.3f}")
    for w, d in reg["windows"].items():
        dsc = d.get("discrimination") or {}
        print(f"ladder {w:<6} n={d['n']} thr {d['threshold']} -> "
              f"{100 * (d['low_share'] or 0):.0f}% LOW | "
              f"AUC {dsc.get('auc_sub90k')} d {dsc.get('cohen_d')} | "
              f"median bank LOW {dsc.get('median_bank_low')} HIGH "
              f"{dsc.get('median_bank_high')}"
              + ("  <- PRIMARY" if w == PRIMARY else
                 "  <- EARLY" if w == EARLY else ""))
    if write:
        save_regime(reg)
        print(f"wrote {REGIME_FILE}")
    return reg


def discrimination(xs, worlds, thr):
    """How well `xs < thr` separates poor worlds. None-safe, pure."""
    pairs = [(x, w) for x, w in zip(xs, worlds)
             if x is not None and w is not None]
    if len(pairs) < 20 or thr is None:
        return {}
    lo = [w for x, w in pairs if x < thr]
    hi = [w for x, w in pairs if x >= thr]
    a = auc([-x for x, _ in pairs], [w < POOR_WORLD for _, w in pairs])
    d = cohen_d(lo, hi)
    return {"n": len(pairs),
            "auc_sub90k": None if a is None else round(a, 4),
            "cohen_d": None if d is None else round(d, 3),
            "median_bank_low": round(statistics.median(lo)) if lo else None,
            "median_bank_high": round(statistics.median(hi)) if hi else None,
            "distinct_values": len({round(x, 6) for x, _ in pairs})}


def regime_from_stats(mins, source, worlds=None):
    """regime.json content from {window: [cell statistic, ...]}.

    When `worlds` (the per-episode shared bank, positionally aligned with the
    statistic lists) is supplied, every window also carries the measured
    discrimination of its cut.
    """
    reg = {"calibrated": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
           "share_target": LOW_SHARE, "primary": PRIMARY, "early": EARLY,
           "statistic": "mean(price/base) over " + "+".join(STAT_PRODUCTS),
           "rule": "LOW if mean over STAT_PRODUCTS of median(price/base) over "
                   "the window's turns < thr",
           "windows": {}}
    for w in WINDOWS:
        raw = list(mins.get(w, []))
        xs = sorted(x for x in raw if x is not None)
        thr, got = calibrate_threshold(xs)
        q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]   # noqa: E731
        reg["windows"][w] = {
            "days": list(WINDOWS[w]), "threshold": thr, "low_share": got,
            "n": len(xs),
            "quantiles": ({f"p{int(f * 100)}": round(q(f), 4)
                           for f in (.10, .25, .35, .50, .75, .90)}
                          if xs else {}),
            "discrimination": (discrimination(raw, worlds, thr)
                               if worlds is not None else {})}
    return reg


def thresholds(reg):
    """{window: threshold or None} out of a regime dict."""
    if not reg:
        return {w: None for w in WINDOWS}
    return {w: (reg.get("windows") or {}).get(w, {}).get("threshold")
            for w in WINDOWS}


def tag_cell(result, thr):
    """{window: LOW/HIGH/None} for one play() result."""
    return {w: regime_tag(result["ratios"][w], thr.get(w)) for w in WINDOWS}


def relative_tags(stats, share=0.5):
    """Within-run split of the PRIMARY statistic -> ([LOW/HIGH/None], thr, got).

    The absolute (ladder-calibrated) tag says what KIND of world a cell is and
    is comparable across runs; it can land 90/10 on a panel whose substrate
    prices sit lower than the ladder's, at which point it stops discriminating
    ANYTHING. This second tag re-cuts the run at its own median so the "does
    this agent win the cheaper half of its OWN panel" question still has an
    answer. Report both; never silently swap one for the other.

    The cut is `calibrate_threshold`, not a raw median, because the panel
    statistic is heavily tied (2026-09-03: 652 cells, 46 distinct values, 248
    of them on one value) and a raw median then throws every tied cell to one
    side. `got` is the share actually achieved -- when it is far from `share`
    the run has no half to speak of, and the caller must say so.
    """
    xs = [x for x in stats if x is not None]
    if len(xs) < 8:
        return [None] * len(stats), None, None
    thr, got = calibrate_threshold(xs, share=share)
    if thr is None:
        return [None] * len(stats), None, None
    return (["LOW" if x < thr else "HIGH" if x is not None else None
             for x in stats], thr, got)


# ------------------------------------------------------------------- playing

_SRV = None


def run_match_traced(agent_a, agent_b, seed, srv):
    """serve_match.run_match plus the per-turn market the driver sees.

    Returns (bank_a, bank_b, info) with info = {"prices": [(day, {product:
    price}) ...] for the tracked days, "shops": [(step, shop)...] unlock order}.
    """
    import kaggriculture.engine.serve_match as SM
    js = srv.cmd(f"RESET {seed}")
    prices, shops, seen = [], [], 0
    while not js.get("done"):
        day = js.get("day", 0)
        if TRACK_DAYS[0] <= day <= TRACK_DAYS[1]:
            pr = (js.get("market") or {}).get("prices") or {}
            prices.append((day, {p: float(pr[p]) for p in STAT_PRODUCTS
                                 if p in pr}))
        us = (js.get("town") or {}).get("unlocked_shops") or []
        if len(us) > seen:
            shops += [(js.get("step", 0), s) for s in us[seen:]]
            seen = len(us)
        la = SM.action_to_line(agent_a(SM.obs_for(0, js)))
        lb = SM.action_to_line(agent_b(SM.obs_for(1, js)))
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            raise RuntimeError(js["error"])
    money = [f.get("money") for f in js["farms"]]
    return float(money[0]), float(money[1]), {"prices": prices, "shops": shops}


def play(job):
    """Worker: one paired cell. (agent, tape, seed, seat) -> result dict."""
    agent, tape, seed, seat = job
    os.chdir(ROOT)
    global _SRV
    import kaggriculture.engine.serve_match as SM
    if _SRV is None:
        _SRV = SM.Serve()
    a = SM.load_agent(agent if seat == 0 else tape)
    b = SM.load_agent(tape if seat == 0 else agent)
    b0, b1, info = run_match_traced(a, b, seed, _SRV)
    own, opp = (b0, b1) if seat == 0 else (b1, b0)
    ratios = {w: regime_ratios(info["prices"], win)
              for w, win in WINDOWS.items()}
    # The unlock list used to be truncated to six, which made every band
    # read "mean 6.0 shops" and hid the very comparison it existed for.
    # Keep the whole sequence; it is a handful of tuples.
    return {"own": own, "opp": opp, "ratios": ratios,
            "stats": {w: cell_stat(r) for w, r in ratios.items()},
            "shops": info["shops"]}


def require_substrate(allow_ungated=False):
    """Engine discipline for the serve substrate: the exact-bank audit gate."""
    import kaggriculture.pipeline.refresh_cycle as RC
    ok, why = RC.serve_allowed()
    if not ok and not allow_ungated:
        raise SystemExit(f"serve substrate not green ({why}); run "
                         "`python src/serve_match.py A.py B.py --seeds 1,2,3 "
                         "--compare-official` or pass --allow-ungated")
    print(f"serve substrate: {'ok' if ok else 'UNGATED'} ({why})")
    return ok


def stamp_splits(bands, frac=SPLIT_FRAC):
    """Freeze `kind` + the holdout split into manifests written before them.

    The split has to live IN the manifest, not only in the run that used it:
    a selection sweep and the scoring run that judges it are different
    processes, and they must agree on which half is which.
    """
    for b in bands:
        p = band_manifest_path(b)
        if not os.path.exists(p):
            continue
        man = json.load(open(p, encoding="utf-8"))
        dirty = False
        for t in man["teams"]:
            if "kind" not in t:
                t["kind"] = "reactive" if b == "reactive" else "tape"
                dirty = True
        if ensure_splits(man["teams"], frac):
            dirty = True
        if dirty:
            man["split"] = {
                "frac_holdout": frac,
                "rule": "stratified by rating, deterministic; HELD-OUT is the "
                        "decision number",
                "selection": sum(1 for t in man["teams"]
                                 if t["split"] == "selection"),
                "holdout": sum(1 for t in man["teams"]
                               if t["split"] == "holdout")}
            json.dump(man, open(p, "w", encoding="utf-8"), indent=1,
                      ensure_ascii=False)
            print(f"panel {band_label(b)}: stamped split "
                  f"{man['split']['selection']} selection / "
                  f"{man['split']['holdout']} held-out into the manifest")


def band_label(b):
    return f"{b:<8}"


def panel_tapes(band, split="selection", limit=0, existing_only=True):
    """Opponent files of one band's SELECTION half (the callable half).

    THE API FOR ANY CALLER THAT SELECTS. `dual_bar_miner` and every hand-run
    sweep must screen candidates on `panel_tapes(band, "selection")` so that
    `band_panel --score` can then judge the survivor on the complement, which
    the search never saw. Pass split=None for the whole band -- but then the
    score you get back is the selection you made, and v44_single is what that
    looks like.
    """
    p = band_manifest_path(band)
    if not os.path.exists(p):
        return []
    man = json.load(open(p, encoding="utf-8"))
    ensure_splits(man["teams"])
    out = [t["tape"] for t in man["teams"]
           if (split is None or t.get("split") == split)
           and (not existing_only or os.path.exists(t["tape"]))]
    return out[:limit] if limit else out


def load_manifests(bands, limit=0, split_frac=SPLIT_FRAC, quiet=False):
    """Manifests for `bands`, with `kind` and the holdout split guaranteed."""
    manifests = {}
    for b in bands:
        p = band_manifest_path(b)
        if not os.path.exists(p):
            raise SystemExit(f"no manifest for band {b}: run --build")
        man = json.load(open(p, encoding="utf-8"))
        if limit:
            man["teams"] = man["teams"][:limit]
        for t in man["teams"]:
            t.setdefault("kind", "reactive" if b == "reactive" else "tape")
        if ensure_splits(man["teams"], split_frac) and not quiet:
            print(f"  band {b}: manifest predates the holdout split -- "
                  "assigned in memory; run --build to freeze it")
        manifests[b] = man
    return manifests


def score(agent_paths, bands=BAND_ORDER, seeds=(501,), workers=8,
          recalibrate=False, executor=None, limit=0, seed_holdout=False,
          split_frac=SPLIT_FRAC):
    manifests = load_manifests(bands, limit, split_frac)
    sseed = seed_splits(seeds, split_frac)
    if seed_holdout and len(set(seeds)) < 2:
        print("  --seed-holdout needs >= 2 seeds; falling back to the team axis")
        seed_holdout = False

    jobs, meta, self_play = [], [], []
    for agent in agent_paths:
        apath = os.path.abspath(agent)
        asig = file_sig(apath)
        for b in bands:
            for ti, t in enumerate(manifests[b]["teams"]):
                # The reactive panel contains our own current agent, FROZEN as
                # a copy under a different path. Playing an agent against
                # ITSELF is a guaranteed mirror, not evidence -- so compare
                # CONTENT, not the path the copy happens to live at.
                if (os.path.abspath(t["tape"]) == apath
                        or file_sig(t["tape"]) == asig):
                    self_play.append((os.path.basename(agent), b, t["team"]))
                    continue
                for seed in seeds:
                    for seat in (0, 1):
                        jobs.append((agent, t["tape"], seed, seat))
                        meta.append((agent, b, ti, seed, seat))
    for a, b, t in self_play:
        print(f"  skipping self-play cell: {a} vs {t} in band {b}")
    t0 = time.time()
    print(f"playing {len(jobs)} cells on {workers} worker(s)...", flush=True)
    results = {}
    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for m, r in zip(meta, ex.map(play, jobs, chunksize=1)):
                results[m] = r
    else:
        for m, r in zip(meta, executor(jobs)):
            results[m] = r
    dt = time.time() - t0
    print(f"  {len(jobs)} cells in {dt:.0f}s ({dt / max(1, len(jobs)):.2f}s/cell)")

    # Regime thresholds: frozen in regime.json (calibrated on the ladder
    # traces by default, so runs are comparable); --recalibrate recomputes
    # them from THIS run's cells instead.
    keys = list(results)
    mins = {w: [results[k]["stats"][w] for k in keys] for w in WINDOWS}
    reg = None if recalibrate else load_regime()
    if reg is None or not (reg.get("windows") or {}).get(PRIMARY, {}).get(
            "threshold") or reg.get("statistic") != (
            "mean(price/base) over " + "+".join(STAT_PRODUCTS)):
        if not recalibrate:
            print("regime.json missing or calibrated for a different "
                  "statistic -- recalibrating from the ladder traces")
        reg = calibrate_from_traces()
    if reg is None:
        reg = regime_from_stats(mins, f"panel cells ({len(results)})")
        save_regime(reg)
    thr = thresholds(reg)
    for w in WINDOWS:
        xs = [x for x in mins[w] if x is not None]
        low_now = sum(1 for x in xs if thr[w] is not None and x < thr[w])
        d = reg["windows"][w]
        dsc = d.get("discrimination") or {}
        print(f"regime {w:<6} thr {thr[w]} ({reg.get('statistic')}; at "
              f"calibration {100 * (d.get('low_share') or 0):.0f}% LOW on "
              f"{d.get('n')} {reg.get('source')}, AUC {dsc.get('auc_sub90k')} "
              f"d {dsc.get('cohen_d')}, median bank {dsc.get('median_bank_low')}"
              f"/{dsc.get('median_bank_high')}): {low_now}/{len(xs)} cells LOW "
              f"this run" + ("  <- PRIMARY" if w == PRIMARY else
                             "  <- EARLY" if w == EARLY else ""))
    rel_list, rel_thr, rel_got = relative_tags(mins[PRIMARY])
    rel = dict(zip(keys, rel_list))
    n_rel = sum(1 for v in rel_list if v == "LOW")
    n_tag = sum(1 for v in rel_list if v)
    print(f"regime REL    within-run split of the {PRIMARY} statistic at "
          f"{rel_thr}: {n_rel}/{n_tag} cells LOW "
          f"({'' if rel_got is None else f'{rel_got:.2f} share'})")

    # Coverage: an instrument that reports a regime win% has to have both
    # regimes in it. Rendered open-loop tapes do not build the town's demand,
    # so tape-panel worlds are systematically poorer than ladder worlds -- say
    # so rather than letting a 0.77 LOW win% look like a measured LOW result.
    # Reported PER BAND and PER OPPONENT KIND: that breakdown is what says
    # whether reactive opponents fixed the distribution.
    xs = [x for x in mins[PRIMARY] if x is not None]
    n_low = sum(1 for x in xs if thr[PRIMARY] is not None and x < thr[PRIMARY])
    all_cells = [{"own": results[k]["own"], "opp": results[k]["opp"]}
                 for k in keys]
    distinct = len({round(x, 6) for x in xs})
    share_low = n_low / max(1, len(xs))
    wall = world_stats(all_cells)
    report_cov = dict(wall)
    report_cov.update({"low_share": share_low, "distinct_values": distinct,
                       "rel_threshold": rel_thr, "rel_low_share": rel_got})
    by_band_cells, by_kind_cells = {}, {}
    for k in keys:
        _a, _b, _ti, _s, _seat = k
        cell = {"own": results[k]["own"], "opp": results[k]["opp"],
                "shops": results[k].get("shops")}
        by_band_cells.setdefault(_b, []).append(cell)
        kind = manifests[_b]["teams"][_ti].get("kind", "tape")
        by_kind_cells.setdefault(kind, []).append(cell)
    report_cov["by_band"] = {b: world_stats(v) for b, v in by_band_cells.items()}
    report_cov["by_kind"] = {k: world_stats(v) for k, v in by_kind_cells.items()}
    report_cov["shops_by_band"] = {b: shop_stats(v)
                                   for b, v in by_band_cells.items()}
    report_cov["shops_by_kind"] = {k: shop_stats(v)
                                   for k, v in by_kind_cells.items()}
    report_cov["shops"] = shop_stats(
        [c for v in by_band_cells.values() for c in v])
    print(f"regime COVER  ALL cells: {fmt_world(wall)}; statistic takes "
          f"{distinct} distinct values over {len(xs)} cells, "
          f"{100 * share_low:.0f}% LOW")
    for b in bands:
        if b in report_cov["by_band"]:
            sh = report_cov["shops_by_band"][b]
            print(f"  worlds {b:<9} {fmt_world(report_cov['by_band'][b])}")
            print(f"    shops  {b:<9} mean {sh['mean_shops']} unlocked, "
                  f"{100 * sh['share_no_shop']:.0f}% of cells unlock NONE, "
                  f"first unlock step {sh['median_first_unlock_step']}")
    for kd in sorted(report_cov["by_kind"]):
        print(f"  worlds kind={kd:<9} {fmt_world(report_cov['by_kind'][kd])}")
    if share_low > 0.85 or share_low < 0.15:
        print(f"  WARNING regime coverage {100 * share_low:.0f}% LOW -- this "
              "panel is ONE regime, so its per-regime win% is not a "
              "regime comparison. Fix the panel (score the `reactive` band, "
              "whose worlds ARE ladder-like), not the threshold.")

    report = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": list(seeds),
              "regime": reg, "coverage": report_cov,
              "split": {"frac_holdout": split_frac, "axis":
                        "team+seed" if seed_holdout else "team",
                        "seeds": {str(k): v for k, v in sseed.items()}},
              "self_play_skipped": self_play, "agents": []}
    cellmap = {}                       # agent -> {cell key: cell dict}
    for agent in agent_paths:
        arep = {"agent": agent, "bands": {}}
        cellmap[agent] = {}
        name = os.path.basename(agent)
        for b in bands:
            man = manifests[b]
            rows, losses, flat = [], [], []
            w = l = d = 0
            by_reg = {w: {"LOW": [0, 0], "HIGH": [0, 0]}
                      for w in list(WINDOWS) + ["rel"]}
            for ti, t in enumerate(man["teams"]):
                tw = tl = 0
                cells = []
                for seed in seeds:
                    for seat in (0, 1):
                        key = (agent, b, ti, seed, seat)
                        r = results.get(key)
                        if r is None:                       # self-play, skipped
                            continue
                        tags = tag_cell(r, thr)
                        tags["rel"] = rel.get(key)
                        tag = tags[PRIMARY]
                        won = r["own"] > r["opp"]
                        lost = r["own"] < r["opp"]
                        tw += won
                        tl += lost
                        for win, tg in tags.items():
                            if tg:
                                by_reg[win][tg][0] += won
                                by_reg[win][tg][1] += 1
                        cell = {"seed": seed, "seat": seat,
                                "own": r["own"], "opp": r["opp"],
                                "regime": tag, "regimes": tags,
                                "stats": r["stats"],
                                "ratios": r["ratios"],
                                "shops": r["shops"],
                                "team": t["team"], "rid": t.get("rid"),
                                "kind": t.get("kind", "tape"),
                                "split": cell_split(t["split"], sseed[seed],
                                                    seed_holdout)}
                        cells.append(cell)
                        flat.append(cell)
                        cellmap[agent][(b, ti, seed, seat)] = cell
                        if lost:
                            losses.append({"team": t["team"], "lb": t["lb_score"],
                                           "seed": seed, "seat": seat,
                                           "own": r["own"], "opp": r["opp"],
                                           "regime": tag,
                                           "split": cell["split"]})
                if not cells:
                    continue
                n = len(cells)
                w += tw
                l += tl
                d += n - tw - tl
                rows.append({"team": t["team"], "lb": t["lb_score"],
                             "rid": t["rid"], "won": tw, "lost": tl, "n": n,
                             "split": t["split"], "kind": t.get("kind", "tape"),
                             "cells": cells})
            if not flat:
                continue
            overall = summarise_cells(flat)
            per_split = {s: summarise_cells([c for c in flat
                                             if c["split"] == s])
                         for s in ("selection", "holdout", "mixed")}
            per_kind = {k: summarise_cells([c for c in flat if c["kind"] == k])
                        for k in sorted({c["kind"] for c in flat})}
            n_cells = overall["n"]
            ratio = overall["ratio"]
            bar = BARS[b]
            # The BAR is judged on the HELD-OUT half -- that is the number
            # selection never saw. Falls back to the overall record only when
            # the band has no held-out cells at all.
            decide = per_split["holdout"] if per_split["holdout"]["n"] else overall
            dratio = decide["ratio"]
            ok = ((decide["cells"][1] == 0 and decide["cells"][2] == 0)
                  if bar >= 1.0 else dratio >= bar)
            swept = [r["team"] for r in rows if r["won"] == 0]
            # Both seats of one (team, seed) are the SAME world played from
            # opposite sides. Both agents here are deterministic and the
            # engine's two farms start identical, so the swap reproduces the
            # cell exactly and adds no information: count the distinct worlds
            # so the effective sample size is never overstated.
            pairs = {}
            for r in rows:
                for c in r["cells"]:
                    pairs.setdefault((r["rid"], c["seed"]), []).append(
                        (c["own"], c["opp"]))
            mirrored = sum(1 for v in pairs.values()
                           if len(v) > 1 and len(set(v)) == 1)
            print(f"\n{b.upper():<8} {name}: {fmt_summary(overall)} | swept-by "
                  f"{len(swept)} | bar {bar:.2f} on HELD-OUT "
                  f"{'n/a' if dratio is None else f'{dratio:.3f}'} -> "
                  f"{'PASS' if ok else 'FAIL'}")
            print(f"  SELECTION  {fmt_summary(per_split['selection'])}")
            print(f"  HELD-OUT   {fmt_summary(per_split['holdout'])}"
                  "   <- DECISION NUMBER")
            if per_split["mixed"]["n"]:
                print(f"  mixed      {fmt_summary(per_split['mixed'])}"
                      "   (straddles the split; belongs to neither)")
            if len(per_kind) > 1:
                for kd, s in per_kind.items():
                    print(f"  kind {kd:<9} {fmt_summary(s)}")
            print(f"  worlds     {fmt_world(world_stats(flat))}")
            print(f"  cells {n_cells} over {len(pairs)} distinct worlds "
                  f"({mirrored} of them seat-mirrored exactly -- those seat "
                  f"pairs carry one game's worth of evidence, not two)")
            for win in list(WINDOWS) + ["rel"]:
                parts = [f"{tag} {v[0]}/{v[1]} = {v[0] / v[1]:.3f}"
                         for tag, v in by_reg[win].items() if v[1]]
                if parts:
                    print(f"  regime {win:<6}: " + "   ".join(parts)
                          + ("   <- PRIMARY" if win == PRIMARY else
                             "   <- EARLY" if win == EARLY else
                             "   <- within-run median split" if win == "rel"
                             else ""))
            if swept:
                print(f"  swept by: {', '.join(swept)}")
            if losses and (bar >= 1.0 or len(losses) <= 12):
                for x in losses:
                    lbs = "   n/a" if x["lb"] is None else f"{x['lb']:>7.1f}"
                    print(f"  LOSS  {x['team']:<28} lb {lbs} seed {x['seed']}"
                          f" seat {x['seat']}  {x['own']:>8.0f} vs {x['opp']:>8.0f}"
                          f"  [{x['regime']}/{x['split'][:4]}]")
            for r in sorted(rows, key=lambda r: (r["won"], -(r["lb"] or 0.0))):
                lbs = "   n/a" if r["lb"] is None else f"{r['lb']:>7.1f}"
                print(f"    {r['team'][:28]:<28} lb {lbs}  "
                      f"{r['won']}/{r['n']}  [{r['split'][:4]}]")
            # DEAD-OPPONENT GUARD (2026-09-03). A panel opponent that banks
            # <= its 3,000 starting money did not play a game: it either
            # crashed, or our agent denied it the market so completely that
            # its economy never started. Either way the cell is a free win
            # and flatters the score. On 2026-09-03 the shipped v43 base
            # zeroed 22 of 83 top-100 tapes (44 of 326 cells, 20%) by feed
            # denial -- real, reproducible, and worth knowing about, but the
            # score must be readable WITHOUT it. Ranking was unchanged then;
            # do not assume it always will be. Always read `clean_ratio`.
            if overall["dead_opponent_cells"]:
                cw, cl = overall["clean_cells"]
                cr = overall["clean_ratio"]
                print(f"  DEAD OPPONENTS: {overall['dead_opponent_cells']} "
                      f"cell(s) where the opponent banked <= "
                      f"{DEAD_OPPONENT_BANK:,} -- excluded, clean {cw}-{cl} "
                      + (f"({cr:.3f})" if cr is not None else "(no live cells)"))
            arep["bands"][b] = {"cells": overall["cells"], "ratio": ratio,
                                "bar": bar, "pass": ok,
                                "decision_ratio": dratio,
                                "decision_split": ("holdout"
                                                   if per_split["holdout"]["n"]
                                                   else "overall"),
                                "swept_by": swept,
                                "dead_opponent_cells":
                                    overall["dead_opponent_cells"],
                                "clean_cells": overall["clean_cells"],
                                "clean_ratio": overall["clean_ratio"],
                                "median_margin": overall["median_margin"],
                                "mean_margin": overall["mean_margin"],
                                "clean_median_margin":
                                    overall["clean_median_margin"],
                                "summary": overall,
                                "by_split": per_split,
                                "by_kind": per_kind,
                                "worlds": world_stats(flat),
                                "distinct_worlds": len(pairs),
                                "seat_mirrored": mirrored,
                                "by_regime": {
                                    win: {k: {"won": v[0], "n": v[1]}
                                          for k, v in dd.items()}
                                    for win, dd in by_reg.items()},
                                "losses": losses, "rows": rows,
                                "panel_built": man["built"]}
        report["agents"].append(arep)

    # ------------------------------------------------ paired agent comparison
    #
    # Cells saturate; margins do not. When more than one agent is scored, the
    # first is the CANDIDATE and every other is compared to it on the cells
    # they share -- winners (McNemar) AND dollars (sign test), split into the
    # selection and held-out halves. The held-out row is the verdict.
    comparisons = []
    if len(agent_paths) > 1:
        base = agent_paths[0]
        for other in agent_paths[1:]:
            shared = sorted(set(cellmap[base]) & set(cellmap[other]))
            entry = {"a": base, "b": other, "bands": {}}
            print(f"\nPAIRED  {os.path.basename(base)}  vs  "
                  f"{os.path.basename(other)}   ({len(shared)} shared cells)")
            scopes = [("ALL", bands, ("selection", "holdout", "mixed"))]
            for b in bands:
                scopes.append((b.upper(), (b,),
                               ("selection", "holdout", "mixed")))
            for label, bsel, _ in scopes:
                for sp in ("all", "selection", "holdout"):
                    ks = [k for k in shared if k[0] in bsel
                          and (sp == "all"
                               or cellmap[base][k]["split"] == sp)]
                    if not ks:
                        continue
                    n_raw = len(ks)
                    ks = dedupe_mirrored(ks, cellmap[base], cellmap[other])
                    cmp_ = compare_cells([cellmap[base][k] for k in ks],
                                         [cellmap[other][k] for k in ks])
                    cmp_["cells_before_mirror_dedupe"] = n_raw
                    entry["bands"].setdefault(label, {})[sp] = cmp_
                    mark = "  <- DECISION" if (sp == "holdout"
                                               and label == "ALL") else ""
                    print(f"  {label:<9} {sp:<9} "
                          f"n={cmp_['n_pairs']}/{n_raw} worlds "
                          f"winners {cmp_['better_a']}b/{cmp_['better_b']}w "
                          f"p={cmp_['p_value']:.4f} | margin "
                          f"{cmp_['margin_better_a']}b/"
                          f"{cmp_['margin_better_b']}w "
                          f"p={cmp_['margin_sign_p']:.4f} med "
                          f"{cmp_['median_margin_delta']:+,.0f} | "
                          f"moved-not-flipped {cmp_['margin_moved_winner_same']}"
                          f", identical {cmp_['identical_cells']}" + mark)
            comparisons.append(entry)
    report["comparisons"] = comparisons

    os.makedirs(BAND_DIR, exist_ok=True)
    json.dump(report, open(LAST_SCORES, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    # A 2,110-cell run is 15 minutes of the machine and it used to be erased by
    # the next two-cell smoke test. Keep a timestamped copy; `last_scores.json`
    # stays the "most recent" pointer every existing caller reads.
    arch_dir = os.path.join(BAND_DIR, "runs")
    os.makedirs(arch_dir, exist_ok=True)
    arch = os.path.join(arch_dir, time.strftime("scores_%Y%m%d_%H%M%S.json"))
    json.dump(report, open(arch, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"\nwrote {LAST_SCORES}\n      {arch}")
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", nargs="*", default=[])
    ap.add_argument("--bands", default=",".join(BAND_ORDER))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N teams per band (smoke tests only)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="recompute the regime threshold from this run")
    ap.add_argument("--calibrate-traces", action="store_true",
                    help="report the ladder-trace distribution of the statistic")
    ap.add_argument("--allow-ungated", action="store_true")
    ap.add_argument("--split-frac", type=float, default=SPLIT_FRAC,
                    help="held-out share of each panel (default 0.5)")
    ap.add_argument("--seed-holdout", action="store_true",
                    help="hold out seeds as well as teams (needs >= 2 seeds); "
                         "a held-out cell then shares neither opponent nor "
                         "seed with anything selection saw")
    args = ap.parse_args()
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    bad = [b for b in bands if b not in BANDS]
    if bad:
        ap.error(f"unknown band(s) {bad}; choose from {list(BANDS)}")
    if args.calibrate_traces:
        calibrate_from_traces()
    missing = [b for b in bands if not os.path.exists(band_manifest_path(b))]
    if args.build:
        build(bands, cap=args.cap, frac=args.split_frac)
    elif missing:
        # Build ONLY what is absent. Re-mining a band that already exists
        # would silently change the panel mid-comparison (and re-download the
        # leaderboard) just because a new band was added to the default set.
        print(f"missing manifest(s) {missing} -- building only those")
        build(missing, cap=args.cap, frac=args.split_frac)
    stamp_splits(bands, args.split_frac)
    if args.score:
        require_substrate(args.allow_ungated)
        score([os.path.abspath(a) for a in args.score], bands=bands,
              seeds=tuple(500 + i for i in range(1, args.seeds + 1)),
              workers=args.workers, recalibrate=args.recalibrate,
              limit=args.limit, seed_holdout=args.seed_holdout,
              split_frac=args.split_frac)


if __name__ == "__main__":
    main()
