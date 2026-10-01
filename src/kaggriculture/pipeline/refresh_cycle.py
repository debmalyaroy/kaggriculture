"""The daily refresh cycle: mine -> diagnose -> crown -> arms -> build -> gate.

One command reproduces the loop that produced v19.1/v22.1/v23, unattended:

    python -m kaggriculture.pipeline.refresh_cycle                # full cycle
    python -m kaggriculture.pipeline.refresh_cycle --stop-after crown
    python -m kaggriculture.pipeline.refresh_cycle --report       # show the last cycle's report

Stages (all idempotent, all logged under data/refresh/<date>/):

  detect    the two active submission refs, from Kaggle
  games     refresh our own ladder games for both refs
  mine      newest archive day(s); abort cycle if nothing new AND no new losses
  tapes     loss tapes from our newest losses (2 per family, capped roster)
  tourney   fresh fit-window candidates (per-family best) + incumbent control,
            refereed by the loss tapes + crowd guard, two seed sets
  crown     gate: winner must beat the incumbent control by >= CROWN_GATE
            percentage points on the referee roster, else HOLD
  arms      retrain counter-arms on the crowned base for families present in
            the loss column ((1+lambda) ES, bounded)
  build     v{N}_route.py + v{N}_bandit.py, graphs, registry
  gate      paired validation bandit-vs-route (zero regressions), dry-run
            submission check; writes the report with the two submit commands

The cycle NEVER submits. A human reads the report and submits.
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

CROWN_GATE = 10          # percentage points over incumbent on the referees
# NO MARGIN CRITERION EXISTS ANY MORE. The ladder pays win/draw/loss, so $1 and
# $10,000 of margin score identically and a dollar bar decides in the wrong
# currency. Margin survives as a LOGGED DIAGNOSTIC only; when win% saturates
# the decision is the playoff (stage_playoff, WINS), and with no playoff the
# answer is HOLD. See src/win_metric.py for why a mean margin cannot even be
# converted into wins without the per-game distribution.
TAPES_PER_FAMILY = 2
MAX_LOSS_TAPES = 8
FIELD_STRATA = 8          # field-matched panel tapes spanning opponent sell volume
                          # (widened 4 -> 8 per plan: the field spans 413..3,763
                          # opponent sell units and 4 strata was coarse)
N_HOLDOUT = 4             # loss tapes RESERVED for validation, never selection
CAND_PER_FAMILY = 2
MAX_CANDIDATES = 14
# Reserved tournament tickets for hinge-trio sellers (2026-08-20): the
# own-games forensics showed the flagship sells ZERO CARROT/TOMATO/EGG while
# the field's egg volume tracks its wins, and the hinge screen measured
# fresh hinge-selling routes up to +42pp over the incumbent (discordant
# 37-3, p~0) -- yet medoid/rank selection kept leaving them without a
# ticket. See models/factory/hinge_screen.json and src/hinge_screen.py.
HINGE_SLOTS = 4
# Foreign artifacts (source == "notebook": another team's published route,
# decoded and registered) stay out of the unattended tournament. Set by
# --include-foreign for a deliberate, attended evaluation only.
INCLUDE_FOREIGN = False
SEED_SETS = (60000, 150000)
# CROWN-2 B2/B3: when the pre-ranker gate is open, the Rust engine screens
# PRERANK_POOL fresh routes and the top PRERANK_K get official tickets; the
# budget the narrower width frees is spent on DEPTH -- the final round adds
# SEED_SETS_EXTRA. Gate closed => neither applies.
PRERANK_K = 24
PRERANK_POOL = 200
SEED_SETS_EXTRA = (240000, 330000)
_PRERANK_ORDER = None
# Concurrent evaluate.py processes in the tournament. Each is one core of
# pure-python engine. CAPPED AT 3 (2026-08-13): sustained 5-6-wide load
# BSOD'd this laptop twice in one day (bugcheck 0x133 DPC_WATCHDOG_VIOLATION
# at 11:09 and 12:00, minidumps 081326-228*/226*) -- a driver starves the
# DPC queue when the box runs flat-out for tens of minutes. 3-wide keeps
# most of the speedup (~3x over sequential) without saturating the machine.
TOURNEY_JOBS = 3
GUARD_TAPE = os.path.join(ROOT, "data", "panel", "opp_r004_90558188_s0.py")

ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def work_dir():
    d = os.path.join(ROOT, "data", "refresh", dt.date.today().isoformat())
    os.makedirs(d, exist_ok=True)
    return d


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(work_dir(), "cycle.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd, timeout=3600):
    """Run a child; a TIMEOUT is a FAILED CHILD (code 124), never a raise.

    2026-08-16: seq_dataset.py overran its budget and the uncaught
    TimeoutExpired aborted the whole 04:30 cycle before the tournament ran --
    the morning shipped nothing. Every call site already degrades on a
    nonzero code (a failed candidate, a stale model, a skipped enhancement),
    so a hung child must present the same way. Stages where a failure IS
    fatal (stage_build) check the code and raise explicitly -- that contract
    is unchanged.
    """
    log("$ " + " ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=ENV,
                           cwd=ROOT)
        code, raw = p.returncode, (p.stdout or b"") + (p.stderr or b"")
    except subprocess.TimeoutExpired as exc:
        code = 124
        raw = ((exc.stdout or b"") + (exc.stderr or b""))
        log(f"child TIMED OUT after {timeout}s (treated as failed, "
            f"cycle continues): {os.path.basename(cmd[-1] if len(cmd) < 3 else cmd[1])}")
    out = raw.decode("utf-8", "replace")
    with open(os.path.join(work_dir(), "cmds.log"), "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n{out}\n")
    return code, out


SERVE_EQUIV = os.path.join(ROOT, "models", "serve_equiv.json")
# Minimum genuinely-reactive agents a serve audit must certify before the
# tournament may trust serve to RANK reactive candidates (A2.3).
REACTIVE_MIN = 6


def serve_allowed():
    """May the tournament run on the Rust serve substrate?

    Evidence-gated like every adoption here: the audit must show exact-bank
    matches with ZERO mismatches on the CURRENT engine, recently. Any
    mismatch -- from the audit or from the post-tournament spot-check --
    closes the gate until a fresh audit passes.
    """
    try:
        d = json.load(open(SERVE_EQUIV, encoding="utf-8"))
        eng = json.load(open(os.path.join(
            ROOT, "models", "engine_version.json"),
            encoding="utf-8")).get("engine")
        age_h = (dt.datetime.now().timestamp()
                 - os.path.getmtime(SERVE_EQUIV)) / 3600
        if d.get("mismatches", 1) != 0:
            return False, f"{d['mismatches']} bank mismatch(es) on record"
        if d.get("matches", 0) < 12:
            return False, f"only {d.get('matches', 0)} exact matches (<12)"
        # Reactive coverage (A2.3): tape-shell agents IGNORE obs and cannot
        # expose a serve invocation-layer bug (this is exactly how the step-719
        # mis-rank, S5, sat undetected while 12/12 tape shells read green). The
        # audit MUST now certify genuinely reactive agents. A serve_equiv.json
        # without a reactive roster (the legacy tape-shell cert) fails here.
        if d.get("reactive_covered", 0) < REACTIVE_MIN:
            return False, (f"only {d.get('reactive_covered', 0)} reactive agents"
                           f" certified (<{REACTIVE_MIN}) -- tape-shell cert "
                           f"cannot certify reactive ranking (run "
                           f"kaggriculture.engine.reactive_parity)")
        if d.get("engine") != eng:
            return False, f"audit ran on {d.get('engine')}, engine is {eng}"
        if age_h > 7 * 24:
            return False, f"audit {age_h / 24:.1f}d old (>7d)"
        return True, (f"{d['matches']} exact-bank matches, 0 mismatches, "
                      f"{d.get('reactive_covered', 0)} reactive agents, "
                      f"{d.get('serve_s_per_game', '?')}s/game vs "
                      f"{d.get('official_s_per_game', '?')}s official")
    except (OSError, ValueError, KeyError) as exc:
        return False, f"no usable audit ({type(exc).__name__})"


def serve_spot_check(picks, log=print):
    """Replay serve-decided pairings on the official engine; revoke on any
    bank mismatch. `picks` = [(agent_path, opp_path, seed), ...]."""
    bad = 0
    for agent, opp, seed in picks:
        code, out = run([sys.executable, "src/kaggriculture/engine/serve_match.py", agent, opp,
                         "--seed", str(seed), "--compare-official"],
                        timeout=900)
        if code != 0 or "MISMATCH" in out:
            bad += 1
            log(f"SERVE SPOT-CHECK MISMATCH: {os.path.basename(agent)} vs "
                f"{os.path.basename(opp)} seed {seed}")
    if bad:
        try:
            d = json.load(open(SERVE_EQUIV, encoding="utf-8"))
            d["mismatches"] = int(d.get("mismatches", 0)) + bad
            d["revoked_by"] = f"spot-check {dt.date.today().isoformat()}"
            with open(SERVE_EQUIV, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1)
        except (OSError, ValueError):
            pass
        log(f"SERVE SUBSTRATE REVOKED ({bad} mismatch(es)) -- future "
            f"tournaments fall back to the official engine until a fresh "
            f"audit passes")
    else:
        log(f"serve spot-check: {len(picks)}/{len(picks)} exact-bank "
            f"matches on the official engine")
    return bad == 0


# ------------------------------------------------ crown de-saturation (C) --
# The banded referee panel (Workstream C) that replaces the SATURATED
# CROWN_GATE=10pp win-rate bar. `crown_panel.py` assembles LB-rating-banded
# referees; `crown_gate.py` paired-tests the challenger vs the incumbent per
# band and applies the ship rule (no band significantly regressed + a positive
# ladder-weighted aggregate). This block is the wiring the crown stage
# consumes; the serve gate above (serve_allowed / REACTIVE_MIN) is untouched.
CROWN_PANEL = os.path.join(ROOT, "models", "crown_panel.json")
# Per-referee measurement budget in the banded crown gate. Small on purpose --
# this box is memory-constrained (memory: c9-opening-nogo). Both seats are
# always played so the cells pair.
CROWN_PANEL_SEEDS = (60000, 150000)


def crown_panel_referees(split="selection"):
    """{band: [referee path]} from models/crown_panel.json, degrade-safe.

    Returns {} (never raises) when no panel is built or it cannot be read, so a
    cycle with no panel simply falls back to the legacy win%/playoff gate.
    """
    try:
        import kaggriculture.measure.crown_panel as CP
        return CP.referees(split=split)
    except Exception as exc:                                       # noqa: BLE001
        log(f"crown panel unavailable ({type(exc).__name__}: {str(exc)[:80]})")
        return {}


def _score_vs_referees(agent_path, ref_paths, seeds=CROWN_PANEL_SEEDS,
                       srv=None):
    """Per-cell win scores of `agent_path` vs each referee, both seats.

    Returns a dict keyed by (referee_basename, seed, seat) -> win score
    (win_metric.score), so two agents scored with the SAME key set pair up
    exactly for the McNemar test. One shared serve process; missing/crashing
    cells are simply absent (both agents drop the same key, so pairing holds).
    """
    import kaggriculture.engine.serve_match as SM
    import kaggriculture.measure.win_metric as WM
    own = srv is None
    if own:
        srv = SM.Serve()
    out = {}
    try:
        agent = SM.load_agent(agent_path)
        for ref in ref_paths:
            base = os.path.basename(ref)
            try:
                oppfn = SM.load_agent(ref)
            except Exception:                                      # noqa: BLE001
                continue
            for seed in seeds:
                for seat in (0, 1):
                    try:
                        if seat == 0:
                            b0, b1 = SM.run_match(agent, oppfn, seed, srv)
                        else:
                            b1, b0 = SM.run_match(oppfn, agent, seed, srv)
                    except Exception:                              # noqa: BLE001
                        continue
                    out[(base, seed, seat)] = WM.score(b0, b1)
    finally:
        if own:
            srv.close()
    return out


def _paired_band_cells(best_path, inc_path, ref_by_band, seeds, srv):
    """{band: (best_scores, inc_scores)} aligned over the cells BOTH produced."""
    per_band = {}
    for band, refs in ref_by_band.items():
        bs = _score_vs_referees(best_path, refs, seeds, srv)
        is_ = _score_vs_referees(inc_path, refs, seeds, srv)
        keys = sorted(set(bs) & set(is_))
        if keys:
            per_band[band] = ([bs[k] for k in keys], [is_[k] for k in keys])
    return per_band


def stage_crown_panel(best_path, inc_path, seeds=CROWN_PANEL_SEEDS):
    """Recalibrated crown gate (C2): paired McNemar per rating band on the
    banded panel, plus the held-out overfit alarm.

    Returns (verdict, holdout) or (None, None) when no panel/serve is available
    (the caller then keeps the legacy decision). REPORT + advisory: the caller
    treats a significant BAND REGRESSION as a HOLD (the ship rule), never
    crowns anything the legacy selector rejected.
    """
    import kaggriculture.measure.crown_gate as CG
    sel = crown_panel_referees("selection")
    if not sel:
        log("crown panel: none built -- legacy win%/playoff gate stands")
        return None, None
    ok, why = serve_allowed()
    if not ok:
        log(f"crown panel: serve substrate not green ({why}) -- skipping the "
            f"banded gate this cycle; legacy decision stands")
        return None, None
    import kaggriculture.engine.serve_match as SM
    srv = SM.Serve()
    try:
        per_band = _paired_band_cells(best_path, inc_path, sel, seeds, srv)
        verdict = CG.crown_gate_banded(per_band)
        hold_refs = crown_panel_referees("holdout")
        holdout = None
        if hold_refs:
            ho_band = _paired_band_cells(best_path, inc_path, hold_refs,
                                         seeds, srv)
            holdout = CG.holdout_alarm(verdict["ship"], ho_band)
    finally:
        srv.close()
    log(f"banded crown gate: {verdict['reason']}")
    for band, v in sorted(verdict["bands"].items()):
        log(f"    {band:10} best vs incumbent {v['score_a']:.2f} vs "
            f"{v['score_b']:.2f} (diff {v['score_diff']:+.3f}, "
            f"disc {v['better_a']}-{v['better_b']}, p={v['p_value']:.3f})")
    if holdout:
        log(f"    holdout: {holdout['reason']}")
    return verdict, holdout


# ---------------------------------------------------------------- stages --

def stage_detect(attempts=5):
    """The refs of our two active submissions.

    Retries. On 2026-08-12 the 06:30 release died two seconds in with "could not
    detect two active submissions": a single throttled/empty response from the
    submissions endpoint (after a night of bulk replay downloads) raised
    SystemExit and took the whole unattended pipeline with it. The identical
    call succeeded minutes later. A transient API hiccup must not cost a day's
    release, so back off and retry before giving up.

    The ref pattern is \\d{6,} rather than exactly 8 digits: ids are 8 wide
    today and will roll over to 9, which would have silently emptied `refs` and
    produced the same fatal error for a completely different reason.
    """
    last = ""
    for i in range(max(1, attempts)):
        code, out = run(["kaggle", "competitions", "submissions",
                         "kaggriculture"])
        refs = []
        for line in out.splitlines():
            m = re.match(r"\s*(\d{6,})\s", line)
            if m and "COMPLETE" in line:
                refs.append(m.group(1))
            if len(refs) == 2:
                break
        if len(refs) == 2:
            log(f"active pair: {refs}")
            return refs
        last = (out or "").strip().splitlines()[-1][:160] if out else "(no output)"
        if i + 1 < attempts:
            wait = 20 * (i + 1)
            log(f"detect attempt {i + 1}/{attempts} found {len(refs)} ref(s) "
                f"[{last}] -- retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"could not detect two active submissions after "
                     f"{attempts} attempts; last response: {last}")


def stage_games(refs):
    for ref in refs:
        run([sys.executable, "src/kaggriculture/data/ourgames.py", "--submission", ref,
             "--jobs", "6", "--limit", "200"], timeout=3600)
    idx = json.load(open(os.path.join(ROOT, "data", "ourgames", "index.json"),
                         encoding="utf-8"))
    rows = [g for g in idx["games"].values() if str(g.get("submission")) in refs]
    losses = [g for g in rows if not g["won"] and not g["tied"]]
    log(f"{len(rows)} games for the active pair, {len(losses)} losses")
    return rows, losses


def stage_mine(refs=(), fetch=True):
    import kaggriculture.data.routes as R
    before = set(R.load_index()["routes"])
    if fetch:
        # 1. Same-day LEADERBOARD harvest FIRST -- the top teams' current
        #    episodes via the authenticated Kaggle API (no browser), ahead of
        #    the archive's ~1-day lag. Freshest and largest foreign-route
        #    source; plus our own submissions' games. Delta-only.
        # Budgets raised 2026-08-14 by operator order ("don't budget the
        # size"): they remain only as runaway-fetch safety valves, sized so
        # an honest delta never hits them.
        run([sys.executable, "src/kaggriculture/data/fresh_data.py", "--subs",
             ",".join(str(r) for r in refs), "--quick", "--no-retrain",
             "--lb-gb", "40"], timeout=14400)
        # 2. Archive mine (lagged but broad).
        run([sys.executable, "src/kaggriculture/data/routes.py", "--mine", "--top", "200",
             "--max-gb", "30", "--jobs", "8"], timeout=7200)
        # 3. Breadth pass for the identifier's confusion zone (mid-ladder).
        run([sys.executable, "src/kaggriculture/data/routes.py", "--mine", "--breadth",
             "--max-gb", "12", "--jobs", "8"], timeout=5400)
    else:
        # --no-fetch: the caller (daily_release stage 1, or the hourly
        # KaggricultureSameDay scrape) has already brought the index current.
        # Measured 2026-08-12: the cycle re-running the full fetch after
        # daily_release's own cost ~55 duplicated minutes per release.
        log("mine: fetch skipped (--no-fetch); using the index as-is")
    idx = R.load_index()
    R.assign_windows(idx, verbose=False)
    R.save_index(idx)
    new = set(idx["routes"]) - before
    newest_day = max((r.get("date", "") for r in idx["routes"].values()),
                     default="")
    if not fetch:
        # "new" must mean "new since the last BUILD", not "new during this
        # call", or a --no-fetch cycle would always see 0 and hold.
        today = dt.date.today().isoformat()
        new = {rid for rid, r in idx["routes"].items()
               if r.get("date") in (today, newest_day)}
    log(f"mine: {len(new)} new routes; newest day {newest_day}")

    # Version-drift alarm: the 2026-08-07 rebalance changed logic without
    # touching any fingerprinted constant, so engine_check could not see it.
    # The archive tells us what the ladder actually runs.
    from collections import Counter as _C
    fresh_engines = _C(r.get("engine") for r in idx["routes"].values()
                       if r.get("date") == newest_day and r.get("engine"))
    if fresh_engines:
        modal = fresh_engines.most_common(1)[0][0]
        try:
            import importlib.metadata as _md
            local = None
            vinit = os.path.join(ROOT, "vendor",
                                 "kaggle_environments", "envs", "kaggriculture",
                                 "kaggriculture.py")
            src = open(vinit, encoding="utf-8").read()
            local = ("1.32.6+" if "MAX_SHOP_INSTANCES" in src else "<=1.32.4")
        except Exception:                                          # noqa: BLE001
            local = "unknown"
        drift = not (modal.startswith("1.32.6") and local == "1.32.6+")
        msg = (f"ENGINE {'DRIFT ALARM' if drift else 'ok'}: ladder runs "
               f"{dict(fresh_engines)}, vendored {local}")
        log(msg)
        if drift:
            with open(os.path.join(work_dir(), "ENGINE_DRIFT"), "w") as fh:
                fh.write(msg)
    return new, newest_day


def field_strata(rows, n_strata=FIELD_STRATA):
    """Pick games spanning the field's OPPONENT SELL-VOLUME distribution.

    Why this exists (measured 2026-08-13 on 22 real replays): the opponent's
    cumulative sell volume drives the late-game price level, which is what our
    fixed-basket route actually realises -- corr(their volume, our bank) -0.46,
    corr(late price, our bank) +0.88. Losses cluster in the depressed regime
    (late price ~50) and wins in the healthy one (~82).

    The panel was built from LOSSES ONLY, sorted worst-first, so it sampled
    only heavy producers: decoded roster volumes were 1,014..3,742 (mean
    1,963) against a real field of 413..3,763 (mean 1,633) with NO light
    seller at all. A market-timing or price-regime change measured on that
    panel is being asked about a regime the panel never shows, which is how
    `_price_hold` came back "inert". Stratifying on the volume itself makes
    the panel's distribution match the field's, so panel results carry over.

    Returns one representative game per stratum, nearest the stratum centre.
    """
    scored = []
    for g in rows:
        vol = sum(int(v or 0) for v in (g.get("opp_sold") or {}).values())
        if vol > 0:
            scored.append((vol, g))
    if not scored:
        return []
    # Sort on the volume ONLY. A bare tuple sort falls through to comparing
    # the game dicts whenever two volumes tie -- and dicts do not order, so a
    # tie is a TypeError that kills the whole cycle at the tape stage.
    scored.sort(key=lambda vg: vg[0])
    lo, hi = scored[0][0], scored[-1][0]
    if hi <= lo:
        return [scored[0][1]]
    picks, seen = [], set()
    for k in range(n_strata):
        target = lo + (hi - lo) * (k + 0.5) / n_strata
        vol, g = min(scored, key=lambda vg: abs(vg[0] - target))
        eid = str(g["episode"])
        if eid in seen:
            continue
        seen.add(eid)
        picks.append(g)
    return picks


def stage_tapes(refs, losses, rows=()):
    import kaggriculture.agentbuild.counter_agent as CA
    import kaggriculture.measure.opponents as opponents
    lib, teams = CA.build_library(exclude_id="")
    finals = []
    for sched in lib:
        cum = {i: 0 for i in CA.TRACKED}
        for sells in sched.values():
            for item, q in sells:
                cum[item] += q
        finals.append(cum)

    def label(g):
        opp = {i: int((g.get("opp_sold") or {}).get(i, 0) or 0) for i in CA.TRACKED}
        total = sum(opp.values())
        if total < 50:
            return "unknown"
        ds = sorted((sum(abs(opp[i] - f[i]) for i in CA.TRACKED), n)
                    for n, f in enumerate(finals))
        best, who = ds[0]
        return teams[who] if best <= 0.35 * total else "NOVEL"

    # Strict-future property (Kaito's rigor, formalized): referee tapes come
    # from LIVE games, which postdate every archive day the candidates are
    # built from -- assert it rather than assume it.
    def _epnum(v):
        m = re.search(r"(\d{6,})", str(v or ""))
        return int(m.group(1)) if m else 0
    newest_route_ep = max((_epnum(r.get("episode"))
                           for r in __import__("routes").load_index()["routes"].values()),
                          default=0)
    strict = [g for g in losses if _epnum(g.get("episode")) > newest_route_ep]
    log(f"strict-future check: {len(strict)}/{len(losses)} loss episodes "
        f"postdate the newest archived route (ep {newest_route_ep})")

    by_family = defaultdict(list)
    for g in losses:
        by_family[label(g)].append((float(g.get("margin") or 0), g))
    picks = []
    spare = []
    for fam, fam_rows in by_family.items():
        # key= is load-bearing: equal margins (several rows defaulting to 0.0
        # is enough) would fall through to comparing the game DICTS -- a
        # TypeError, not a tie-break.
        fam_rows.sort(key=lambda mg: mg[0])
        picks.extend((fam, g) for _, g in fam_rows[:TAPES_PER_FAMILY])
        spare.extend((m, fam, g) for m, g in fam_rows[TAPES_PER_FAMILY:])
    picks = picks[:MAX_LOSS_TAPES]

    # HELD-OUT PANEL (plan: "split the panel"). The same roster used to
    # SELECT the crown also validated it, so a candidate could be crowned for
    # fitting the panel and nothing would show it. Reserve the next-worst
    # losses -- never seen by the tournament, the playoff, arm training or the
    # bandit gate -- and report the crown decision against them. Divergence
    # between panels is the overfit alarm this project previously lacked.
    taken = {str(g["episode"]) for _, g in picks}
    spare.sort(key=lambda t: (t[0], t[1]))
    for _, fam, g in spare:
        if len([1 for f, _g in picks if f == "HOLDOUT"]) >= N_HOLDOUT:
            break
        eid = str(g["episode"])
        if eid not in taken:
            taken.add(eid)
            picks.append(("HOLDOUT", g))

    # Loss tapes teach what beats us; they cannot tell us whether a change
    # survives the HEALTHY market, because we do not lose there. Add
    # field-matched strata (wins included) so the panel spans the real
    # sell-volume distribution rather than only its heavy tail.
    have = set(str(g["episode"]) for _, g in picks)
    strata = [g for g in field_strata(list(rows) or list(losses))
              if str(g["episode"]) not in have]
    if strata:
        vols = [sum(int(v or 0) for v in (g.get("opp_sold") or {}).values())
                for g in strata]
        log(f"field strata: +{len(strata)} tapes spanning opponent sell "
            f"volume {min(vols):,}..{max(vols):,} "
            f"({sum(1 for g in strata if g.get('won'))} from wins)")
        picks.extend(("FIELD", g) for g in strata)

    tape_dir = os.path.join(work_dir(), "loss_tapes")
    os.makedirs(tape_dir, exist_ok=True)
    stage = os.path.join(work_dir(), "_stage")

    def _fetch_replay(eid, dest):
        """Local staged cache -> direct -> CLI, with bounded 429 backoff.

        On 2026-08-15 04:32 BOTH remote paths were 429-throttled (the hourly
        job had spent the API budget overnight) and all 13 one-shot fetches
        failed -> cycle HELD, no v26. Retry rounds with backoff ride out a
        throttle window instead of dying on it; the sameday _stage is free.
        """
        cached = os.path.join(ROOT, "data", "sameday", "_stage", eid,
                              f"{eid}.json")
        if os.path.exists(cached) and os.path.getsize(cached) > 2_000_000:
            shutil.copy(cached, os.path.join(dest, f"{eid}.json"))
            return
        import kaggriculture.data.sameday as sameday
        delays = (0, 90, 240)          # ~5.5 min worst case per episode
        last = None
        for i, d in enumerate(delays):
            if d:
                time.sleep(d)
            try:
                sameday._grab_direct(eid, dest)
                return
            except Exception as exc:                           # noqa: BLE001
                last = exc
            try:
                subprocess.run(["kaggle", "competitions", "replay", eid,
                                "-p", dest], check=True,
                               capture_output=True, env=ENV, timeout=300)
                import zipfile
                for f in os.listdir(dest):
                    if f.endswith(".zip"):
                        with zipfile.ZipFile(os.path.join(dest, f)) as z:
                            z.extractall(dest)
                return
            except Exception as exc:                           # noqa: BLE001
                last = exc
        raise last

    made = []
    ladder_failures = 0
    for fam, g in picks:
        eid = str(g["episode"])
        dest = os.path.join(stage, eid)
        os.makedirs(dest, exist_ok=True)
        try:
            if not any(f.endswith(".json") for f in os.listdir(dest)):
                # Quota exhaustion is GLOBAL: once two episodes have burned a
                # full retry ladder each, the rest will too -- skip straight
                # to the stale-tape fallback instead of spending ~6 min per
                # episode proving the same 429 (the 2026-08-15 run spent 76
                # min doing exactly that and tripped the outer timeout).
                if ladder_failures >= 2:
                    raise RuntimeError("skipped: remote quota exhausted")
                try:
                    _fetch_replay(eid, dest)
                except Exception:
                    ladder_failures += 1
                    raise
            path = next(os.path.join(dest, f) for f in os.listdir(dest)
                        if f.endswith(".json"))
            # The label lands in the tape's docstring and sink._route_identity
            # reads docstrings, so it must state what the game actually was --
            # field-strata tapes come from games we WON.
            verb = "beat us" if not g.get("won") else "lost to us"
            rel = opponents.build_tape(path, 1 - int(g["seat"]),
                                       out_dir=tape_dir,
                                       label=f"{fam} {verb} ep {eid}")
            os.remove(path)
            if rel:
                path = os.path.join(ROOT, rel) if not os.path.isabs(rel) else rel
                made.append((fam, path))
                log(f"  taped {fam[:24]} ep {eid} ({g.get('margin'):+,.0f})")
        except Exception as exc:                                   # noqa: BLE001
            log(f"  ! tape {eid}: {str(exc).splitlines()[0]}")
    shutil.rmtree(stage, ignore_errors=True)

    if not made:
        # DEGRADED MODE (2026-08-15): a 24h replay-quota exhaustion left every
        # fetch 429ing for hours, so "no fresh tapes" no longer means "hold" --
        # it means reuse the NEWEST previous day's tape set. Archive freshness
        # is normally ~2 days behind anyway (memory: archive-lag), so one-day-
        # stale tapes are inside normal operating conditions. The report line
        # below makes the degradation loud rather than silent.
        for day_dir in sorted(glob.glob(os.path.join(
                ROOT, "data", "refresh", "*", "loss_tapes")), reverse=True):
            if os.path.abspath(day_dir) == os.path.abspath(tape_dir):
                continue
            old = sorted(glob.glob(os.path.join(day_dir, "tape_*.py")))
            if not old:
                continue
            for src_tape in old:
                dst = os.path.join(tape_dir, os.path.basename(src_tape))
                shutil.copy(src_tape, dst)
                fam = "STALE"
                try:
                    head = open(src_tape, encoding="utf-8",
                                errors="ignore").read(400)
                    for known in ("HOLDOUT", "FIELD"):
                        if known in head:
                            fam = known
                            break
                except OSError:
                    pass
                made.append((fam, dst))
            log(f"TAPE FALLBACK: fresh fetches all failed; reusing "
                f"{len(made)} tapes from {os.path.dirname(day_dir)} "
                f"(quota-throttle degraded mode)")
            break
    return made


def rank_by_score(res, names):
    """Names best-first on SCORE. Margin is a TIE-BREAK ONLY, never a ranking
    currency of its own -- the ladder pays win/draw/loss (src/win_metric.py)."""
    def key(nm):
        w, g, m_ = res.get(nm, (0, 0, 0.0))
        return (-(w / g if g else 0.0), -m_, nm)
    return sorted(names, key=key)


def run_halving(cand_names, screen, roster, seed_sets, play, log=print):
    """Successive halving over candidates. Returns (final_results, finalists).

    Every candidate used to play the full roster on both seed sets -- 1,120+
    games, ~51 min at 3 workers -- including the ones eliminated in their first
    four games. The budget now escalates with survival:

      round 1  all candidates, SCREEN roster, one seed set, n=1
      round 2  survivors,      full roster,   one seed set, n=1
      final    finalists,      full roster,   ALL seed sets, n=2

    ONLY THE FINAL ROUND IS RETURNED, so stage_crown still compares finalists
    and the incumbent over an identical roster and both seed sets -- the
    contract it already had. Earlier rounds are elimination only and are never
    scored against. The incumbent is the control and is never eliminated.

    `play(names, opps, seeds, n) -> {name: (wins, games, margin)}` is injected
    so the control flow is testable without running episodes.
    """
    r1 = play(["__incumbent__"] + cand_names, screen, seed_sets[:1], 1)
    keep = rank_by_score(r1, cand_names)[
        :max(PLAYOFF_FINALISTS + 1, (len(cand_names) + 1) // 2)]
    log(f"tourney round 1 (screen, {len(screen)} referees): "
        f"{len(cand_names)} -> {len(keep)} survive")

    if len(keep) > PLAYOFF_FINALISTS:
        r2 = play(["__incumbent__"] + keep, roster, seed_sets[:1], 1)
        keep = rank_by_score(r2, keep)[:PLAYOFF_FINALISTS]
        log(f"tourney round 2 (full roster, {len(roster)} referees): "
            f"-> {len(keep)} finalists")

    final = play(["__incumbent__"] + keep, roster, seed_sets, 2)
    log(f"tourney final ({len(keep)} finalists + incumbent, "
        f"{len(roster)} referees, {len(seed_sets)} seed sets)")
    return final, keep


def stage_tourney(newest_day, tapes, incumbent):
    import kaggriculture.data.routes as R
    import kaggriculture.data.features as features
    idx = R.load_index()
    # Freshest data wins: same-day scraped routes (source=sameday, dated
    # today) rank ahead of the newest archive day. Consider both.
    today = dt.date.today().isoformat()
    # source == "notebook" is a FOREIGN artifact decoded from someone else's
    # public notebook, registered so the crown gate *can* be asked about it by
    # hand. It is deliberately excluded from the unattended pool: this cycle
    # crowns and daily_release then submits, and shipping a competitor's
    # published route as our own must be a human decision, never a 06:30 job's.
    # Evaluate one on purpose with --include-foreign.
    foreign = [r for r in idx["routes"].values()
               if r.get("source") == "notebook"]
    fresh = [r for r in idx["routes"].values()
             if r.get("window") == "fit"
             and (r.get("date") in (newest_day, today)
                  or r.get("source") == "sameday")
             and (INCLUDE_FOREIGN or r.get("source") != "notebook")]
    log(f"tournament pool: {len(fresh)} fit routes "
        f"({sum(1 for r in fresh if r.get('source') == 'sameday')} same-day)")
    if foreign and not INCLUDE_FOREIGN:
        log(f"excluded {len(foreign)} foreign notebook route(s) from the "
            f"unattended pool (use --include-foreign to evaluate them)")
    # Per-family candidate cap, family = broad behavioural signature cluster
    # (283 dims: sell curves, buys, plantings, herd, labor, builds, endgame).
    reps, buckets = [], defaultdict(list)
    for rec in sorted(fresh, key=lambda r: (r.get("rank") or 999)):
        try:
            vec, _ = features.route_features(R.load_route(rec["id"]))
        except Exception:                                          # noqa: BLE001
            continue
        placed = None
        for fi, rep in enumerate(reps):
            if features.distance(vec, rep) < 0.055:
                placed = fi
                break
        if placed is None:
            reps.append(vec)
            placed = len(reps) - 1
        buckets[placed].append(rec)
    cands = []
    for fi in sorted(buckets, key=lambda f: (buckets[f][0].get("rank") or 999)):
        cands.extend(buckets[fi][:CAND_PER_FAMILY])
    cands = cands[:MAX_CANDIDATES]
    log(f"tournament: {len(cands)} candidates from {len(buckets)} fresh families")

    # CROWN-2 B2: Rust-wide funnel. When the PRE-RANKER GATE is open
    # (measured recall, train_gates.preranker_allowed), the Rust engine
    # screens the WHOLE fresh pool open-loop against the panel's source
    # routes (~2 min at 143 ep/s) and the official tournament takes the
    # top PRERANK_K instead of only the family medoids. Gate closed or any
    # failure => exactly the selection above. Every release decision still
    # happens on the official engine; this only chooses who gets a ticket.
    global _PRERANK_ORDER
    _PRERANK_ORDER = None
    try:
        import kaggriculture.train.train_gates as TG
        ok_pr, why_pr = TG.preranker_allowed()
        if ok_pr:
            import kaggriculture.engine.rust_prerank as RP
            ref_ids = []
            for _fam, tp in (t for t in tapes
                             if isinstance(t, (tuple, list)) and len(t) == 2):
                m = re.match(r"tape_(\d+)_s(\d)\.py", os.path.basename(tp))
                if m and str(_fam) != "HOLDOUT":
                    ref_ids.append(f"{m.group(1)}_s{m.group(2)}")
            pool_ids = [r["id"] for r in sorted(
                fresh, key=lambda r: r.get("episode", ""), reverse=True)
                [:PRERANK_POOL]]
            if ref_ids and len(pool_ids) >= PRERANK_K:
                order = RP.rank(pool_ids, ref_ids, log=lambda *_: None)
                if len(order) >= PRERANK_K:
                    by_id = {r["id"]: r for r in fresh}
                    cands = [by_id[rid] for rid in order[:PRERANK_K]
                             if rid in by_id]
                    _PRERANK_ORDER = order
                    log(f"rust funnel ({why_pr}): {len(pool_ids)} routes "
                        f"pre-ranked -> top {len(cands)} enter the official "
                        f"tournament")
    except Exception as exc:                                       # noqa: BLE001
        log(f"rust funnel unavailable ({type(exc).__name__}: "
            f"{str(exc)[:80]}) -- family-medoid selection stands")

    # HINGE-SELLER QUOTA: reserve tickets for the top hinge-trio sellers
    # (obsfeat sell-share x bank) the medoid/rank funnel did not pick. They
    # face the same referees, the same overlap exclusion below, and the same
    # crown bar -- this only guarantees they get MEASURED.
    try:
        import kaggriculture.measure.hinge_screen as HS
        have = {r["id"] for r in cands}
        hrows = []
        for rec in fresh:
            if rec["id"] in have or float(rec.get("bank", 0)) < 30000:
                continue
            hs = HS.hinge_share(rec["id"])
            if hs and hs > 0.02:
                hrows.append((hs * float(rec.get("bank", 0)), rec))
        hrows.sort(key=lambda t: (-t[0], t[1]["id"]))
        picked = [rec for _, rec in hrows[:HINGE_SLOTS]]
        if picked:
            cands.extend(picked)
            log(f"hinge quota: +{len(picked)} hinge-seller candidate(s): "
                f"{[r['id'] for r in picked]}")
    except Exception as exc:                                       # noqa: BLE001
        log(f"hinge quota unavailable ({type(exc).__name__}: "
            f"{str(exc)[:80]}) -- continuing")

    # CROWN-2 D: OVERLAP EXCLUSION, replacing the strict-future property that
    # same-day data made unenforceable (measured 0/74 on 2026-08-14). The
    # leakage that property actually guarded against is a candidate being
    # refereed by a tape from its own episode or its own team -- exclude
    # exactly that, nothing more.
    ref_eps, ref_teams = set(), set()
    for t in tapes:
        if not (isinstance(t, (tuple, list)) and len(t) == 2):
            continue
        m = re.match(r"tape_(\d+)_s(\d)\.py", os.path.basename(t[1]))
        if not m:
            continue
        ref_eps.add(m.group(1))
        team = (idx["routes"].get(f"{m.group(1)}_s{m.group(2)}") or {}
                ).get("team")
        if team and team != "?":
            ref_teams.add(team)
    before_x = len(cands)
    cands = [r for r in cands
             if str(r.get("episode")) not in ref_eps
             and (r.get("team") or "?") not in ref_teams]
    if len(cands) != before_x:
        log(f"overlap exclusion: {before_x - len(cands)} candidate(s) "
            f"dropped (own episode/team appears on the referee panel)")

    cdir = os.path.join(work_dir(), "cand")
    os.makedirs(cdir, exist_ok=True)
    paths = {"__incumbent__": incumbent}
    for rec in cands:
        out = os.path.join(cdir, f"c_{rec['id']}.py")
        code, _ = run([sys.executable, "src/kaggriculture/data/routes.py", "--build", rec["id"],
                       "--out", os.path.relpath(out, ROOT)])
        if code == 0:
            paths[rec["id"]] = out

    # FACTORY ENTRANTS (2026-08-18, lever L5): a searched agent whose
    # sell_search HOLDOUT gate passed (fresh seeds, paired sign test vs its
    # base) enters the tournament like any candidate -- no special seeding,
    # the crown decides. The id carries a "factory::" prefix so downstream
    # stages know no index route sits behind it.
    try:
        fv = json.load(open(os.path.join(ROOT, "models", "factory",
                                         "sell_search.json"),
                            encoding="utf-8"))
        f_agent = fv.get("agent")
        f_when = dt.datetime.fromisoformat(str(fv.get("when")))
        f_fresh = (dt.datetime.now() - f_when).total_seconds() < 7 * 86400
        if fv.get("passed") and f_agent and f_fresh:
            fp = os.path.join(ROOT, f_agent)
            if os.path.exists(fp):
                paths[f"factory::{f_agent}"] = fp
                log(f"factory entrant: {f_agent} (sell_search holdout "
                    f"passed {fv.get('when')})")
    except (OSError, ValueError, TypeError):
        pass

    # tapes may arrive labelled [(family, path)] or bare [path]. The FIELD
    # strata are the screening slice below, so keep the labels when given.
    labelled = [t for t in tapes if isinstance(t, (tuple, list)) and len(t) == 2]
    if labelled:
        # HOLDOUT tapes exist to VALIDATE the crown, never to select it --
        # letting them into the roster would collapse the two panels back
        # into one and reintroduce the overfit blindspot they exist to catch.
        tape_paths = [p for fam, p in labelled if str(fam) != "HOLDOUT"]
        strata = [p for fam, p in labelled if str(fam) == "FIELD"]
    else:
        tape_paths = list(tapes)
        strata = []
    roster = tape_paths + [GUARD_TAPE]
    # The screening roster spans the opponent sell-volume range (413..3,763
    # units), so a candidate that fails it fails across the price regime
    # rather than in one corner of it. Without strata, fall back to a spread
    # of the full roster instead of an arbitrary prefix.
    screen = strata or roster[::max(1, len(roster) // 4)]

    # REFEREE PRUNING (src/referee_power.py). A tape that every candidate beats
    # -- or that beats every candidate -- costs full price and changes no
    # ranking. This drops those, but only on real evidence: with fewer than
    # MIN_CANDIDATES distinct candidates on record every referee looks
    # non-discriminating, and the FIELD strata are protected because they exist
    # for price-regime coverage rather than discrimination.
    try:
        import kaggriculture.measure.referee_power as RP
        table = RP.gather()
        if table:
            # C1.2: protect the crown panel's BAND STRATA -- a banded referee
            # exists to hold a rating band, so it must never be pruned for low
            # discrimination the way a redundant loss tape is.
            try:
                import kaggriculture.measure.crown_panel as CP
                band_protected = CP.protected_prefixes()
            except Exception:                                      # noqa: BLE001
                band_protected = ()
            _rows, keep_ref, drop_ref = RP.analyse(
                table, protected=RP.PROTECTED_PREFIXES + band_protected)
            if drop_ref and drop_ref[0][0] == "__all__":
                log(f"referee pruning skipped: {drop_ref[0][1]}")
            elif drop_ref and len(keep_ref) >= RP.MIN_KEEP:
                names = {os.path.basename(x) for x, _ in drop_ref}
                before = len(roster)
                roster = [r for r in roster
                          if os.path.basename(r) not in names]
                screen = [r for r in screen
                          if os.path.basename(r) not in names] or screen
                log(f"referee pruning: {before} -> {len(roster)} "
                    f"(dropped {sorted(names)})")
    except Exception as exc:                                       # noqa: BLE001
        log(f"referee pruning unavailable ({type(exc).__name__}) -- "
            f"using the full roster")

    # Every (candidate, roster, seed-set) cell is an independent child
    # process; running them one at a time left the tournament single-core
    # (~40-60 min measured 2026-08-12). Fan out to TOURNEY_JOBS concurrent
    # children -- threads are fine here, the work lives in the children.
    #
    # SUBSTRATE (2026-08-16): when the serve-equivalence audit is green
    # (models/serve_equiv.json: exact-bank matches, zero mismatches, current
    # engine), the cells run on `kagg serve` via src/serve_match.py --
    # identical CLI, identical RESULT lines, measured 0.6s/game against the
    # official engine's 12-15s. A cell that comes back empty on serve is
    # replayed on the official engine (defensive), and stage_tourney ends
    # with an official spot-check that REVOKES the audit on any bank
    # mismatch -- the funnel's adopt-with-kill-switch pattern.
    from concurrent.futures import ThreadPoolExecutor

    # Before any measurement: does the vendored engine still reproduce real
    # ladder episodes? A failed parity audit means every number this
    # tournament produces answers a different question than the ladder asks.
    try:
        from kaggriculture.engine.ladder_parity import parity_ok
        par_ok, par_why = parity_ok()
        if not par_ok and "mismatch" not in par_why:
            # Stale or missing audit: refresh it now (a mismatch on record,
            # by contrast, blocks until a human looks).
            log(f"ladder parity audit stale ({par_why}); re-auditing")
            run([sys.executable, "src/kaggriculture/engine/ladder_parity.py", "--replays", "6"],
                timeout=1800)
            par_ok, par_why = parity_ok()
    except Exception as exc:                                       # noqa: BLE001
        par_ok, par_why = False, f"parity check unavailable ({exc})"
    if not par_ok:
        log(f"LADDER PARITY BLOCK: {par_why} -- run src/ladder_parity.py "
            f"before trusting or shipping anything from this cycle")
        raise SystemExit(f"ladder parity gate: {par_why}")
    log(f"ladder parity: {par_why}")

    serve_ok, serve_why = serve_allowed()
    eval_script = "src/kaggriculture/engine/serve_match.py" if serve_ok else "src/kaggriculture/measure/evaluate.py"
    log(f"tournament substrate: "
        f"{'RUST-SERVE' if serve_ok else 'official engine'} ({serve_why})")

    def _play(names, opps, seeds, n_matches):
        """{name: (wins, games, margin)} over one round."""
        cells = [(nm, paths[nm], s) for nm in names for s in seeds]

        def _cell(args_):
            name, agent, seed0 = args_
            code, out = run([sys.executable, eval_script, agent,
                             "--vs", *opps, "-n", str(n_matches),
                             "--seed0", str(seed0), "--no-record"],
                            timeout=3600)
            # Read the stable RESULT lines, never the human table.
            import kaggriculture.measure.win_metric as WM
            try:
                rows = WM.parse_eval(out)
            except ValueError as exc:
                rows = []
                if eval_script.endswith("serve_match.py"):
                    log(f"  ! {name} seed0={seed0} empty on serve -- "
                        f"replaying the cell on the official engine")
                    code, out = run([sys.executable, "src/kaggriculture/measure/evaluate.py",
                                     agent, "--vs", *opps, "-n",
                                     str(n_matches), "--seed0", str(seed0),
                                     "--no-record"], timeout=3600)
                    try:
                        rows = WM.parse_eval(out)
                    except ValueError as exc2:
                        log(f"  ! {name} seed0={seed0}: {exc2}")
                else:
                    log(f"  ! {name} seed0={seed0}: {exc}")
            return (name, sum(r["wins"] for r in rows),
                    sum(r["games"] for r in rows),
                    sum(r["margin"] for r in rows))

        out_ = {}
        with ThreadPoolExecutor(max_workers=TOURNEY_JOBS) as pool:
            for name, w, g, m_ in pool.map(_cell, cells):
                pw, pg, pm = out_.get(name, (0, 0, 0.0))
                out_[name] = (pw + w, pg + g, pm + m_)
        return out_

    cand_names = [n for n in paths if n != "__incumbent__"]
    if not cand_names:
        return {"__incumbent__": (0, 0, 0.0)}, paths

    # DETERMINISM PRE-CHECK before spending the tournament budget. A paired
    # comparison over a non-reproducible agent produces a number that reads
    # exactly like a result -- the whole design leans on paired seeds, and that
    # variance reduction evaporates if either side wanders.
    #
    # EXCLUDE offenders rather than abort (2026-08-14, operator: "nothing
    # should crash"): one flaky candidate is an elimination, not a reason to
    # lose the nightly release. Only a non-reproducible INCUMBENT is fatal --
    # every comparison is against it, so nothing valid can be measured.
    try:
        import kaggriculture.measure.determinism as determinism
        rows_d = determinism.check(list(paths.values()) + list(roster),
                                   verbose=False)
        bad = {a for a, ok_, _ in rows_d if not ok_}
        if bad:
            if paths["__incumbent__"] in bad:
                raise SystemExit("the INCUMBENT does not reproduce on a "
                                 "fixed seed; no paired result is valid")
            dropped_c = [n for n, p in list(paths.items()) if p in bad]
            for n in dropped_c:
                del paths[n]
            before_r = len(roster)
            roster = [r for r in roster if r not in bad]
            screen = [s for s in screen if s not in bad] or roster
            log(f"determinism: EXCLUDED {len(dropped_c)} candidate(s) "
                f"{dropped_c} and {before_r - len(roster)} referee(s) that "
                f"do not reproduce; continuing with the rest")
            if not roster:
                raise SystemExit("no reproducible referees remain")
        else:
            log(f"determinism: {len(paths)} agent(s) + {len(roster)} "
                f"referee(s) reproduce on a fixed seed")
    except SystemExit as exc:
        log(f"ABORTING TOURNEY -- {exc}")
        raise
    except Exception as exc:                                       # noqa: BLE001
        log(f"determinism check unavailable ({type(exc).__name__}); "
            f"continuing")
    # Recompute after the determinism exclusion: a dropped candidate left in
    # cand_names would KeyError inside _play.
    cand_names = [n for n in paths if n != "__incumbent__"]
    if not cand_names:
        return {"__incumbent__": (0, 0, 0.0)}, paths
    # B3: the width the funnel saved is spent on final-round depth.
    seed_sets = (SEED_SETS + SEED_SETS_EXTRA if _PRERANK_ORDER
                 else SEED_SETS)
    results, keep2 = run_halving(cand_names, screen, roster, seed_sets,
                                 _play, log=log)

    # Serve substrate kill-switch: replay two finalist pairings on the
    # official engine; any bank mismatch revokes the serve audit and the
    # next tournament falls back automatically.
    if serve_ok and keep2:
        try:
            picks = [(paths[n], roster[i % len(roster)],
                      int(seed_sets[0]) + i)
                     for i, n in enumerate(list(keep2)[:2]) if n in paths]
            if picks:
                serve_spot_check(picks, log=log)
        except Exception as exc:                                   # noqa: BLE001
            log(f"serve spot-check unavailable ({type(exc).__name__}) -- "
                f"substrate stays as-is this cycle")

    # B4: trailing recall audit -- does the Rust pre-rank keep putting the
    # official finalists near its own top? Each cycle appends a point; the
    # gate reads the TRAILING MEAN, so drifting recall closes the funnel by
    # itself, the same way a parity failure revokes it.
    if _PRERANK_ORDER and keep2:
        try:
            pos = {rid: i for i, rid in enumerate(_PRERANK_ORDER)}
            half = max(PLAYOFF_FINALISTS, len(cand_names) // 2)
            hits = sum(1 for n in keep2 if pos.get(n, 10 ** 6) < half)
            point = hits / len(keep2)
            hist_p = os.path.join(ROOT, "models", "lab",
                                  "preranker_recall_history.jsonl")
            with open(hist_p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"date": dt.date.today().isoformat(),
                                     "recall": point, "n": len(keep2),
                                     "candidates": len(cand_names)}) + "\n")
            pts = [json.loads(l) for l in open(hist_p, encoding="utf-8")]
            tail = pts[-5:]
            trailing = sum(p["recall"] for p in tail) / len(tail)
            json.dump({"recall_at_n": trailing, "n": len(keep2),
                       "candidates": sum(p["candidates"] for p in tail),
                       "source": "trailing-cycle-audit",
                       "points": len(tail)},
                      open(os.path.join(ROOT, "models", "lab",
                                        "preranker_recall.json"), "w",
                           encoding="utf-8"), indent=1)
            log(f"funnel audit: cycle recall {point:.2f}, trailing "
                f"{trailing:.2f} over {len(tail)} cycle(s)")
        except Exception as exc:                                   # noqa: BLE001
            log(f"funnel audit failed ({type(exc).__name__}) -- "
                f"gate evidence unchanged")
    for name, (w, g, m_) in results.items():
        log(f"  {name:<18} {w}/{g}  {m_:+,.0f}")
    return results, paths


PLAYOFF_FINALISTS = 3        # top candidates joining the incumbent
PLAYOFF_GATE = 10            # percentage points over the incumbent, on WINS


def stage_playoff(results, paths, incumbent):
    """Round-robin among the top finalists + incumbent when the referee panel
    saturates on win%.

    The ladder pays WINS, not coin margin (house rule #6) -- so when every
    strong agent beats the whole loss-tape panel and win% stops
    discriminating, the answer is a HARDER field, not a different currency
    (the 2026-08-12 margin fallback was a stopgap; operator review the same
    night called it out). Strong-vs-strong round-robin play cannot saturate
    the way strong-vs-referee does, approximates the top-field population
    better than any single head-to-head, and its metric is pure win rate.

    Returns {name: (wins, games)} over the round-robin, incumbent included,
    or None when a playoff is impossible (fewer than 2 finalists built)."""
    ranked = sorted(((n, r) for n, r in results.items()
                     if n != "__incumbent__" and n in paths),
                    key=lambda kv: (-kv[1][0], -kv[1][2], kv[0]))
    finalists = [(n, paths[n]) for n, _ in ranked[:PLAYOFF_FINALISTS]]
    finalists.append(("__incumbent__", paths["__incumbent__"]))
    if len(finalists) < 3:
        return None
    log(f"playoff: round-robin among {[n for n, _ in finalists]}")

    from concurrent.futures import ThreadPoolExecutor
    pairs = [(a, pa, b, pb)
             for i, (a, pa) in enumerate(finalists)
             for b, pb in finalists[i + 1:]]
    cells = [(a, pa, b, pb, s) for a, pa, b, pb in pairs for s in SEED_SETS]

    def _duel(args_):
        a, pa, b, pb, seed0 = args_
        code, out = run([sys.executable, "src/kaggriculture/measure/evaluate.py", pa, "--vs", pb,
                         "-n", "2", "--seed0", str(seed0), "--no-record"],
                        timeout=1800)
        import kaggriculture.measure.win_metric as WM
        try:
            rows = WM.parse_eval(out)
        except ValueError as exc:
            log(f"  ! playoff {a} vs {b}: {exc}")
            return a, b, 0, 0
        return (a, b, sum(r["wins"] for r in rows),
                sum(r["games"] for r in rows))

    tally = {n: [0, 0] for n, _ in finalists}
    with ThreadPoolExecutor(max_workers=TOURNEY_JOBS) as pool:
        for a, b, aw, g in pool.map(_duel, cells):
            tally[a][0] += aw
            tally[a][1] += g
            tally[b][0] += g - aw
            tally[b][1] += g
    for n, (w, g) in sorted(tally.items(), key=lambda kv: -kv[1][0]):
        log(f"  playoff {n:<18} {w}/{g}")
    return {n: (w, g) for n, (w, g) in tally.items()}


def _base_class(route_id):
    """Identifier class of a route -- for basin-saturation awareness."""
    try:
        import kaggriculture.data.features as features
        import kaggriculture.data.routes as R
        import kaggriculture.train.train_identifier as TI
        idw = json.load(open(os.path.join(ROOT, "models", "v22", "identifier",
                                          "weights.json"), encoding="utf-8"))
        vec = features.prefix_features(R.load_route(route_id), 480) + [480 / 720.0]
        p = TI.stdlib_predict(idw, vec)
        cls = max(range(len(p)), key=lambda i: p[i])
        return cls, p[cls]
    except Exception:                                              # noqa: BLE001
        return None, 0.0


def incumbent_base_id(agent_path):
    """The route id an existing route agent was built from.

    routes.py --build writes it into the agent's docstring as
    "Route: <id> (<team>)", so the base is recoverable from the artifact itself
    without a side-car record that could drift out of sync with it.
    """
    try:
        head = open(agent_path, encoding="utf-8").read(2000)
    except OSError:
        return None
    m = re.search(r"^Route:\s*(\S+)", head, re.M)
    if m:
        return m.group(1)
    m = re.search(r"^Base route\s+(\S+?);", head, re.M)
    return m.group(1) if m else None


def stage_crown(results, playoff=None):
    inc_w, inc_g, inc_m = results.pop("__incumbent__")
    inc_pct = 100 * inc_w / max(1, inc_g)
    ranked = sorted(results.items(),
                    key=lambda kv: (-kv[1][0], -kv[1][2], kv[0]))
    if not ranked:
        return None, "no candidates"

    # Basin-saturation awareness (Kaito's 22/30 finding): when the leader's
    # behavioral class dominates the fresh top field and a within-noise
    # runner-up sits in a DIFFERENT basin, prefer the runner-up -- a
    # saturated basin is mirror-rich and pays less rating per win.
    import kaggriculture.data.routes as R
    idx = R.load_index()
    fresh_top = [r for r in idx["routes"].values()
                 if (r.get("rank") or 999) <= 30]
    shares = Counter(_base_class(r["id"])[0] for r in fresh_top[:40])
    best, (w, g, m) = ranked[0]
    best_cls, _ = _base_class(best)
    note = ""
    if (best_cls is not None and shares.get(best_cls, 0) > 0.5 * max(1, sum(shares.values()))
            and len(ranked) > 1):
        rb, (w2, g2, m2) = ranked[1]
        rb_cls, _ = _base_class(rb)
        if rb_cls != best_cls and w2 >= w - max(2, g // 24):
            note = (f" [basin pivot: {best} (class {best_cls}, saturated "
                    f"{shares.get(best_cls, 0)}/{sum(shares.values())}) "
                    f"deferred to {rb} (class {rb_cls}) within noise]")
            best, (w, g, m) = rb, (w2, g2, m2)

    pct = 100 * w / max(1, g)
    log(f"crown check: best {best} {pct:.0f}% vs incumbent {inc_pct:.0f}%{note}")
    if pct >= inc_pct + CROWN_GATE:
        return best, (f"crowned {best}: {pct:.0f}% vs incumbent "
                      f"{inc_pct:.0f}%{note}")

    # Win-percentage saturation: when the incumbent leaves no room for a
    # +CROWN_GATE win-rate margin, the loss-tape panel has stopped
    # discriminating. The 2026-08-12 stopgap fell through to per-game MARGIN,
    # but the ladder pays WINS, not coins (house rule #6; operator called the
    # margin criterion out the same night) -- so the decisive tiebreaker is
    # now the PLAYOFF: round-robin win rate among the top finalists +
    # incumbent (stage_playoff), same currency the ladder scores. Margin is
    # logged as a diagnostic only, and remains a last-resort fallback solely
    # when no playoff could be run.
    if inc_pct > 100 - CROWN_GATE and pct >= inc_pct:
        per_game = (m - inc_m) / max(1, g)
        log(f"win% saturated (incumbent {inc_pct:.0f}%): margin diagnostic "
            f"{per_game:+,.0f}/game; deciding on playoff wins")
        if playoff and best in playoff and "__incumbent__" in playoff:
            bw, bg = playoff[best]
            iw, ig = playoff["__incumbent__"]
            bpct = 100 * bw / max(1, bg)
            ipct = 100 * iw / max(1, ig)
            if bpct >= ipct + PLAYOFF_GATE:
                return best, (f"crowned {best} on PLAYOFF wins: {bpct:.0f}% "
                              f"vs incumbent {ipct:.0f}% in the finalists' "
                              f"round-robin (panel saturated at {pct:.0f}%; "
                              f"margin diag {per_game:+,.0f}/game){note}")
            return None, (f"HOLD: panel saturated and playoff gives {best} "
                          f"{bpct:.0f}% vs incumbent {ipct:.0f}% "
                          f"(needs +{PLAYOFF_GATE}; margin diag "
                          f"{per_game:+,.0f}/game){note}")
        # No playoff ran. There is therefore NO win-denominated evidence, and
        # the ladder pays only wins -- so hold. The old code crowned on a
        # $3,000/game margin floor here, which is a decision made in a
        # currency the competition does not use: $1 and $10,000 of margin are
        # paid identically. Worse, mean margin cannot even be converted to
        # wins without the per-game distribution (see src/win_metric.py: a
        # uniform +$3,000 is worth +15pp, the same mean concentrated in a few
        # blowouts is worth ~0). Never crown on dollars; fix the playoff.
        return None, (f"HOLD: panel saturated (both {pct:.0f}%) and NO PLAYOFF "
                      f"ran, so there is no win-based evidence to crown on "
                      f"(margin diag {per_game:+,.0f}/game is not a "
                      f"criterion -- investigate why stage_playoff produced "
                      f"nothing){note}")

    return None, (f"HOLD: best candidate {best} at {pct:.0f}% does not "
                  f"clear incumbent {inc_pct:.0f}% + {CROWN_GATE}{note}")


def stage_holdout(best, paths, holdout_tapes):
    """Report the crown decision against the RESERVED panel.

    The selection roster picked `best`; these tapes never touched selection,
    so if `best` beats the incumbent here too, the crown generalises. If the
    two panels diverge, the tournament is overfitting its referees -- which is
    exactly the failure the panel split exists to make visible. REPORT-ONLY by
    design: the crown decision stands either way, the divergence is the alarm.
    """
    if not holdout_tapes or best not in paths:
        return None
    import kaggriculture.measure.win_metric as WM
    out = {}
    for name in (best, "__incumbent__"):
        wins = games = 0
        for seed0 in SEED_SETS:
            code, txt = run([sys.executable, "src/kaggriculture/measure/evaluate.py", paths[name],
                             "--vs", *holdout_tapes, "-n", "1", "--seed0",
                             str(seed0), "--no-record"], timeout=1800)
            try:
                rows = WM.parse_eval(txt)
            except ValueError as exc:
                log(f"  ! holdout {name}: {exc}")
                continue
            wins += sum(r["wins"] for r in rows)
            games += sum(r["games"] for r in rows)
        out[name] = (wins, games)
    bw, bg = out.get(best, (0, 0))
    iw, ig = out.get("__incumbent__", (0, 0))
    if not bg or not ig:
        log("holdout report: no games completed -- inconclusive")
        return None
    bpct, ipct = 100 * bw / bg, 100 * iw / ig
    agrees = bpct >= ipct
    log(f"HOLDOUT panel ({len(holdout_tapes)} reserved tapes): {best} "
        f"{bpct:.0f}% vs incumbent {ipct:.0f}% -- "
        f"{'AGREES with the crown' if agrees else 'DIVERGES: possible referee overfit'}")
    # Persist the series (CROWN-2 D): a single divergence is noise, a RUN of
    # divergences is referee overfit -- which is only visible as a trend.
    try:
        hist = os.path.join(ROOT, "models", "lab", "holdout_history.jsonl")
        with open(hist, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"date": dt.date.today().isoformat(),
                                 "best": best, "best_w": bw, "best_g": bg,
                                 "inc_w": iw, "inc_g": ig,
                                 "agrees": agrees}) + "\n")
    except OSError:
        pass
    return {"best": (bw, bg), "incumbent": (iw, ig), "agrees": agrees}


def stage_arms(base_id, made):
    """Train one counter-arm per major loss family, bounded for a daily run.

    Guards: a novel-family tape when one exists plus the crowd tape, so the
    lookalike lesson is inside the objective, not just the validation."""
    by_fam = defaultdict(list)
    for fam, path in made:
        # HOLDOUT is the validation reserve -- training an arm on it would
        # contaminate the one panel that never touches selection.
        if fam not in ("unknown", "HOLDOUT"):
            by_fam[fam].append(path)
    ranked = sorted(by_fam.items(), key=lambda kv: -len(kv[1]))
    targets = [kv for kv in ranked if kv[0] != "NOVEL"][:2]
    guards = [GUARD_TAPE] + by_fam.get("NOVEL", [])[:1]
    # While the ARM GUARD is on (judged commits < 25) the trained arms are
    # emptied out of the shipped bandit anyway -- stage_bandit only needs
    # them to EXIST. A full (1+lambda) search (6x10) cost 30-60 min of the
    # morning critical path on 2026-08-13 for schedules that cannot ship,
    # so under the guard train a fast warm-up profile instead; the full
    # search resumes automatically the day field evidence clears the gate.
    gens, pop = "6", "10"
    try:
        import kaggriculture.train.train_gates as TG
        allowed, why = TG.arms_allowed()
        if not allowed:
            # Keyed on n_commits alone this went back to the FULL 6x10 search
            # the moment the audit reached 27 rows -- 30-60 min of critical
            # path spent on schedules the (fixed) ARM GUARD then discards.
            # Same predicate as the guard, so the two can never disagree.
            gens, pop = "2", "4"
            log(f"arm guard active ({why}): fast arm profile "
                f"gens={gens} pop={pop}")
    except (OSError, ValueError, ImportError):
        pass
    arms = {}
    for i, (fam, tapes) in enumerate(targets):
        slug = re.sub(r"[^a-z0-9]+", "", fam.lower())[:12] or f"f{i}"
        arm = f"arm{i}_{slug}"
        code, _ = run([sys.executable, "src/kaggriculture/train/train_arms.py", "--arm", arm,
                       "--base", base_id, "--targets", *tapes[:3],
                       "--guards", *guards, "--gens", gens, "--pop", pop,
                       "--seeds", "2", "--sigma", "5"], timeout=5400)
        route_json = os.path.join(ROOT, "models", "v22", "arms", arm,
                                  "best_route.json")
        if code == 0 and os.path.exists(route_json):
            arms[arm] = fam
            log(f"arm {arm} trained for family {fam}")
    return arms


def stage_bandit(base_id, version, arms, made, route_out):
    """Assemble the bandit variant and gate it paired against the route.

    SECOND-SLOT RULE (2026-08-16): the bandit KEEPS the second submission
    slot only if it beats the route on a sign-tested paired comparison --
    same roster, same seeds, so the (seed set x opponent) cells pair up and
    McNemar's exact test decides. While its adaptive layers are guard-locked
    the bandit is the route plus dead weight, and v24.1->v26 showed what that
    ships: a heavier twin that cannot out-rate its partner. A tie is a LOSS
    of the seat -- the slot then goes to a diversity route (stage_second).
    Returns (path, verdict, passed).
    """
    if not arms:
        return None, "no arms trained; route-only cycle", False
    import kaggriculture.agentbuild.v22_agent as v22_agent
    v22_agent.TEAM2ARM = {fam: arm for arm, fam in arms.items()}
    out = f"agents/v{version}_bandit.py"
    # Every daily pair carries the full adaptive stack, each piece behind its
    # own self-deciding gate (operator order 2026-08-14): --policy embeds the
    # deep-RL sell head only when policy_allowed() is favourable; --gru-auto
    # embeds the GRU only when it is fresh AND measured better than the
    # logistic; --duel-policy is structurally checked at build. A gate that
    # says no is a log line, never a failed build.
    argv, sys.argv = sys.argv, ["v22_agent", "--base", base_id, "--out", out,
                                "--arms", *arms,
                                "--policy", "--gru-auto", "--duel-policy"]
    try:
        v22_agent.main()
    finally:
        sys.argv = argv
    run([sys.executable, "-m", "py_compile", out])

    roster = [p for fam, p in made if fam != "HOLDOUT"][:8] + [GUARD_TAPE]
    import kaggriculture.measure.win_metric as WM
    cells = {}          # (seed0, opponent) -> {name: (score, wins, games, margin)}
    scores = {}
    for name, agent in (("route", route_out), ("bandit", out)):
        wins = games = 0
        margin = 0.0
        for seed0 in SEED_SETS:
            code, txt = run([sys.executable, "src/kaggriculture/measure/evaluate.py", agent,
                             "--vs", *roster, "-n", "2", "--seed0",
                             str(seed0), "--no-record"], timeout=3600)
            try:
                rows = WM.parse_eval(txt)
            except ValueError as exc:
                log(f"  ! {name} seed0={seed0}: {exc}")
                rows = []
            for r in rows:
                cells.setdefault((seed0, r["opponent"]), {})[name] = r["score"]
            wins += sum(r["wins"] for r in rows)
            games += sum(r["games"] for r in rows)
            margin += sum(r["margin"] for r in rows)
        scores[name] = (wins, games, margin)
        log(f"  {name}: {wins}/{games} {margin:+,.0f}")
    rw, rg, rm = scores["route"]
    bw, bg, bm = scores["bandit"]
    # Sign test over the paired cells. Only cells where BOTH agents produced a
    # result pair up; a missing side (crashed eval) is dropped, never guessed.
    a = [v["bandit"] for v in cells.values() if "bandit" in v and "route" in v]
    b = [v["route"] for v in cells.values() if "bandit" in v and "route" in v]
    t = WM.paired_test(a, b)
    passed = (bool(a) and t["score_diff"] > 0 and t["significant"])
    detail = (f"{bw}/{bg} wins vs route {rw}/{rg}; paired cells {len(a)}, "
              f"diff {t['score_diff']:+.3f}, discordant "
              f"{t['better_a']}-{t['better_b']}, p={t['p_value']:.3f}")
    if passed:
        return out, f"bandit KEEPS slot 2 (sign-tested edge): {detail}", True
    return out, (f"bandit LOSES slot 2 (no sign-tested edge over the route): "
                 f"{detail}"), False


def second_slot_force():
    """Operator override for the slot-2 seat (docs/history/pair-improvement-plan.md).

    `models/second_slot_force.json`: {"force": "bandit"|"route2",
    "until": "YYYY-MM-DD"}. Dated and SELF-EXPIRING -- past `until` the
    marker is dead and the SECOND-SLOT RULE decides as usual. The force is
    logged into second_slot.json's `why`, so the audit trail never shows a
    bandit shipping on a failed sign test without saying who ordered it.
    Returns "bandit", "route2", or None.
    """
    p = os.path.join(ROOT, "models", "second_slot_force.json")
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p, encoding="utf-8"))
        until = dt.date.fromisoformat(str(m.get("until", "")))
    except (OSError, ValueError) as exc:
        log(f"second_slot_force.json unreadable ({exc}) -- ignored")
        return None
    if dt.date.today() > until:
        log(f"second-slot force expired ({until}) -- rule decides as usual")
        return None
    kind = str(m.get("force", "")).lower()
    if kind not in ("bandit", "route2"):
        log(f"second_slot_force.json unknown kind {kind!r} -- ignored")
        return None
    return kind


def stage_second(base_id, version, results, cand_paths):
    """Fill slot 2 with the best tournament candidate from a DIFFERENT
    opening family than the crowned base.

    Two diverse routes cover more of the field than one route plus its
    heavier twin: matchmaking plays both submissions against the same rating
    band, so a second agent that wins the games the first one loses is worth
    strictly more than a near-copy. "Different" is decided by the exact
    day-3 stream hash (turn 72) -- the same observation stream_hashes uses
    for exposure. Returns the built path or None.
    """
    try:
        import kaggriculture.data.stream_hashes as SH
        import kaggriculture.data.routes as R
    except ImportError as exc:
        log(f"stage_second unavailable ({exc}); no diversity route")
        return None

    def day3(rid):
        try:
            return SH.prefix_hashes(R.load_route(rid)).get(72)
        except Exception as exc:                                   # noqa: BLE001
            log(f"  day-3 hash failed for {rid} ({type(exc).__name__})")
            return None

    base_hash = day3(base_id)
    ranked = sorted(((rid, wgm) for rid, wgm in results.items()
                     if rid != base_id and rid in cand_paths),
                    key=lambda kv: (-kv[1][0], -kv[1][2], kv[0]))
    for rid, (w, g, m) in ranked:
        h = day3(rid)
        if base_hash is not None and h is not None and h == base_hash:
            log(f"  {rid}: same day-3 opening as the base -- skipped")
            continue
        out = f"agents/v{version}_route2.py"
        code, _ = run([sys.executable, "src/kaggriculture/data/routes.py", "--build", rid,
                       "--out", out])
        if code != 0:
            log(f"  {rid}: build failed -- trying the next candidate")
            continue
        # Adaptive sell-timing ships OFF everywhere (retired 2026-08-18).
        # Five measurements, zero wins: three paired panels (-1.3 to -1.6pp,
        # p>=0.625) and two live ladder windows where the adaptive-ON slot 2
        # sat 478-760 points under the flagship (v27/v28 rating curves).
        run([sys.executable, "src/kaggriculture/agentbuild/model_graph.py", out])
        log(f"diversity route: {rid} ({w}/{g} in the tournament, "
            f"different day-3 opening) -> {out}")
        return out
    log("no diversity candidate available (all shared the base's opening "
        "or failed to build)")
    return None


def stage_build(base_id, version):
    route_out = f"agents/v{version}_route.py"
    if str(base_id).startswith("factory::"):
        # A factory winner is already a finished single-file agent -- there
        # is no index route to render. Copy it under the release name.
        import shutil
        src_p = os.path.join(ROOT, str(base_id).split("::", 1)[1])
        shutil.copyfile(src_p, os.path.join(ROOT, route_out))
        log(f"factory winner: {base_id} -> {route_out}")
        run([sys.executable, "-m", "py_compile", route_out])
        run([sys.executable, "src/kaggriculture/agentbuild/model_graph.py", route_out])
        return route_out
    code, _ = run([sys.executable, "src/kaggriculture/data/routes.py", "--build", base_id,
                   "--out", route_out])
    if code != 0:
        raise SystemExit(f"build failed for {base_id}")
    run([sys.executable, "src/kaggriculture/agentbuild/model_graph.py", route_out])
    # OPENING EXPOSURE (2026-08-14, from the public stream-hash technique):
    # every released agent records how many field routes/teams share its
    # exact opening prefix at day 1/3/6/10. A widely-shared opening is a
    # near-mirror magnet and a copyability signal; the series accumulates in
    # models/lab/opening_hashes.jsonl. Report-only -- never blocks a build.
    try:
        import kaggriculture.data.stream_hashes as SH
        exposure = SH.check_route(base_id, log=log)
        worst = max((v["routes"] for v in exposure.values()), default=0)
        log(f"opening exposure: base {base_id} shares its day-10 prefix "
            f"with {exposure.get(240, {}).get('routes', 0)} route(s) "
            f"(max across checkpoints: {worst})")
    except Exception as exc:                                       # noqa: BLE001
        log(f"opening-exposure check unavailable ({type(exc).__name__}) -- "
            f"continuing")
    return route_out


def next_version():
    """Version scheme v{x}.{y} (2026-08-11): x steps once per day -- the
    daily release built on that day's data plus real-time delta -- and y
    steps for intra-day on-demand fixes. Self-determining: if the newest
    versioned agent was already built today, this build is a fix (y+1);
    otherwise it opens a new day (x+1, y=0)."""
    pat = re.compile(r"v(\d+)(?:\.(\d+))?_.*\.py$")
    best = (-1, -1)
    newest_path = None
    adir = os.path.join(ROOT, "agents")
    for f in os.listdir(adir):
        m = pat.match(f)
        if not m:
            continue
        xy = (int(m.group(1)), int(m.group(2) or 0))
        if xy > best:
            best = xy
            newest_path = os.path.join(adir, f)
    if best[0] < 0:
        return "23.0"
    built_today = (newest_path and dt.date.fromtimestamp(
        os.path.getmtime(newest_path)) == dt.date.today())
    if built_today:
        return f"{best[0]}.{best[1] + 1}"
    return f"{best[0] + 1}.0"


def report(lines):
    path = os.path.join(work_dir(), "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Refresh cycle {dt.date.today().isoformat()}\n\n")
        fh.write("\n".join(lines) + "\n")
    log(f"report -> {os.path.relpath(path, ROOT)}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stop-after",
                    choices=["detect", "games", "mine", "tapes", "tourney",
                             "crown", "build"],
                    default=None)
    ap.add_argument("--incumbent", default=None,
                    help="agent file used as the tournament control; default "
                         "is the newest v*_route.py in agents/")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--on-demand", action="store_true",
                    help="mid-day/on-demand rebuild: fetch delta data via "
                         "fresh_data first, then run the full cycle -- so the "
                         "model is trained on current data regardless of when "
                         "it is built")
    ap.add_argument("--include-foreign", action="store_true",
                    help="let routes decoded from other teams' public "
                         "notebooks (source=notebook) enter the tournament. "
                         "Attended runs only -- crowning one means submitting "
                         "someone else's published route as ours.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip ALL data fetching (fresh_data + archive/breadth "
                         "mines): the caller has already brought the index "
                         "current (daily_release stage 1, hourly scrape). "
                         "Kills the ~55 duplicated minutes measured 2026-08-12.")
    ap.add_argument("--version", default=None, metavar="X.Y",
                    help="explicit version for the built pair, overriding "
                         "next_version()'s self-determination -- for operator-"
                         "ordered releases (e.g. 'build v25 pairs' while a "
                         "same-day fix already bumped y)")
    args = ap.parse_args()

    global INCLUDE_FOREIGN
    INCLUDE_FOREIGN = bool(args.include_foreign)

    if args.report:
        path = os.path.join(work_dir(), "REPORT.md")
        print(open(path, encoding="utf-8").read()
              if os.path.exists(path) else "no report today")
        return 0

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    refs = stage_detect()
    if args.on_demand and not args.no_fetch:
        # Any model built off-schedule still trains on current data: pull the
        # delta before the cycle's own mine (which then finds nothing new and
        # returns fast).
        run([sys.executable, "src/kaggriculture/data/fresh_data.py", "--subs", ",".join(refs),
             "--no-retrain"], timeout=7200)
    if args.stop_after == "detect":
        return 0
    rows, losses = stage_games(refs)
    if args.stop_after == "games":
        return 0
    new_routes, newest_day = stage_mine(refs, fetch=not args.no_fetch)
    if args.stop_after == "mine":
        return 0
    if not new_routes and not losses:
        report(["Nothing new: no fresh routes and no new losses. Holding."])
        return 0
    tapes = stage_tapes(refs, losses, rows)
    if args.stop_after == "tapes":
        return 0
    if not tapes:
        report(["No loss tapes could be built. Holding."])
        return 0

    # Daily model refresh: identifier retrains on the newest index; the gate
    # and surrogate trainers self-activate when their data thresholds clear.
    # EVERY child here is a model refresh, not the release itself, so no
    # failure MODE may kill the cycle -- 2026-08-16 04:30 the uncaught
    # TimeoutExpired from seq_dataset.py aborted the whole build before the
    # tournament ever ran and the morning shipped nothing. A nonzero exit
    # already degraded gracefully; a timeout must degrade the same way.
    def refresh(cmd, timeout):
        try:
            return run(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f"model refresh TIMED OUT after {timeout}s (degrading, the "
                f"release continues): {' '.join(cmd[1:])}")
            return 1, ""

    refresh([sys.executable, "src/kaggriculture/train/train_identifier.py"], timeout=1800)
    # family_dumps.json is keyed by identifier CLASS ID, so it must be
    # regenerated whenever the identifier is -- 2026-08-12 the class count
    # went 11 -> 17 and the stale file would have aimed the M2 relay's
    # per-class schedules at the wrong classes (the arm bug's cousin).
    refresh([sys.executable, "src/kaggriculture/agentbuild/relay_config.py"], timeout=900)
    refresh([sys.executable, "src/kaggriculture/train/train_gates.py"], timeout=600)
    refresh([sys.executable, "src/kaggriculture/train/train_surrogate.py"], timeout=1800)
    # GRU refresh (CROWN-2 / operator order: features must stay available
    # unsupervised). The identifier retrain above changes the class labels,
    # which STALES the GRU -- without a same-day retrain, --gru-auto would
    # correctly refuse to embed it and the bandit would silently ship
    # logistic-only. seq_dataset regenerates on today's labels, gru_export
    # trains the measurement twin (held-out-day acc for the auto-compare)
    # plus the full shipped weights. GPU (llm env), ~10-15 min, and any
    # failure degrades to the logistic -- never blocks the release.
    llm_py = r"C:\ProgramData\anaconda3\envs\llm\python.exe"
    if os.path.exists(llm_py):
        code, _ = refresh([sys.executable, "src/kaggriculture/experiments/seq_dataset.py"],
                          timeout=2700)
        if code == 0:
            code, _ = refresh([llm_py, "-X", "utf8",
                               "src/kaggriculture/experiments/gru_export.py"], timeout=2400)
        if code != 0:
            log("GRU refresh failed -- --gru-auto will keep the logistic "
                "(stale-class guard)")
    else:
        log("llm env python missing -- GRU stays stale; logistic ships")

    incumbent = args.incumbent or max(
        glob.glob(os.path.join(ROOT, "agents", "v*_route.py")),
        key=os.path.getmtime)
    # Pass the LABELLED tapes: stage_tourney uses the FIELD-strata subset as
    # its cheap screening roster, and that identity is lost if only paths go.
    results, cand_paths = stage_tourney(newest_day, tapes, incumbent)
    if args.stop_after == "tourney":
        return 0
    # Playoff only when the panel saturated (the incumbent's win% leaves no
    # room for the +CROWN_GATE test) -- otherwise the panel already
    # discriminates and the plain win% gate decides.
    playoff = None
    iw, ig, _ = results["__incumbent__"]
    if 100 * iw / max(1, ig) > 100 - CROWN_GATE:
        playoff = stage_playoff(results, cand_paths, incumbent)
    base_id, verdict = stage_crown(results, playoff)
    log(verdict)
    # Report a REAL crown (a new base, not a HOLD-rebuild) against the
    # reserved panel. Report-only: divergence is logged into the verdict so it
    # reaches the morning report, but the crown decision stands.
    if base_id is not None and base_id in cand_paths:
        holdout_tapes = [p for t in tapes
                         if isinstance(t, (tuple, list)) and len(t) == 2
                         for f, p in [t] if str(f) == "HOLDOUT"]
        try:
            ho = stage_holdout(base_id, cand_paths, holdout_tapes)
        except Exception as exc:                                   # noqa: BLE001
            # Report-only stage: it must never cost a crowned build.
            log(f"holdout report FAILED ({type(exc).__name__}); skipping")
            ho = None
        if ho is not None and not ho["agrees"]:
            verdict += ("\n- WARNING: held-out panel DIVERGES from the "
                        "selection panel -- possible referee overfit; "
                        "see the cycle log")
        # C2: the RECALIBRATED gate on the banded rating panel. Authoritative
        # on regression ONLY -- a band that significantly regresses turns the
        # crown into a HOLD (the ship rule: no band regressed + a positive
        # ladder-weighted aggregate); it never crowns what the selector above
        # rejected. Degrades to the legacy decision when no panel/serve exists.
        try:
            bverdict, bholdout = stage_crown_panel(cand_paths[base_id],
                                                   incumbent)
        except Exception as exc:                                   # noqa: BLE001
            log(f"banded crown gate FAILED ({type(exc).__name__}: "
                f"{str(exc)[:120]}) -- legacy decision stands")
            bverdict = bholdout = None
        if bverdict is not None:
            verdict += f"\n- banded crown gate: {bverdict['reason']}"
            if bholdout is not None:
                verdict += f"\n- banded holdout: {bholdout['reason']}"
                if bholdout["diverges"]:
                    verdict += ("\n- WARNING: banded held-out panel DIVERGES "
                                "-- possible referee overfit")
            if not bverdict["ship"] and bverdict["regressed_bands"]:
                verdict += (f"\n- OVERRIDE: banded gate HOLDS the crown "
                            f"(regressed bands {bverdict['regressed_bands']}); "
                            f"rebuilding the pair on the incumbent base")
                log(f"banded gate OVERRIDE: crown {base_id} regresses "
                    f"{bverdict['regressed_bands']} -- HOLD")
                base_id = None
    if args.stop_after == "crown":
        report([verdict])
        return 0
    if base_id is None:
        # A HOLD means the BASE does not change -- it does not mean there is
        # nothing to release. The models on top of the base are retrained every
        # cycle (2026-08-12: the identifier went from 2,703 to 4,140 routes),
        # and blocking the whole build on a crown miss is why the pair sat at
        # v23 while a broken bandit stayed live. Rebuild the daily pair on the
        # INCUMBENT's base instead, and let the gate in daily_release decide
        # whether the result is worth uploading.
        base_id = incumbent_base_id(incumbent)
        if base_id is None:
            report([verdict, "- could not recover the incumbent's base route "
                             "id; nothing rebuilt"])
            return 0
        verdict += (f"\n- base unchanged ({base_id}); rebuilding the pair on it "
                    f"with models retrained on today's data")
        log(f"HOLD but rebuilding on the incumbent base {base_id}")

    version = args.version or next_version()
    route_out = stage_build(base_id, version)
    if args.stop_after == "build":
        return 0

    # Arms and the bandit are ENHANCEMENTS on top of a route that is already
    # built and gated -- a crash in either must degrade the release to
    # route-only, never lose it (operator 2026-08-14: "nothing should crash").
    try:
        arms = stage_arms(base_id, tapes)
    except Exception as exc:                                       # noqa: BLE001
        log(f"stage_arms FAILED ({type(exc).__name__}: "
            f"{str(exc)[:120]}) -- continuing route-only")
        arms = {}
    try:
        bandit_out, bandit_verdict, bandit_passed = stage_bandit(
            base_id, version, arms, tapes, route_out)
    except Exception as exc:                                       # noqa: BLE001
        bandit_out = None
        bandit_passed = False
        bandit_verdict = (f"bandit build FAILED ({type(exc).__name__}: "
                          f"{str(exc)[:120]}); route-only cycle")
    log(bandit_verdict)

    # SECOND-SLOT RULE: a bandit without a sign-tested edge hands the seat to
    # a diversity route. The bandit file still exists (registry, audit, and
    # the day its layers clear their guards it wins the seat back on merit).
    # A dated operator marker (second_slot_force) overrides the seat decision
    # -- but never the measurement: stage_bandit's paired verdict is still
    # taken and recorded either way.
    second_out = bandit_out if bandit_passed else None
    second_why = bandit_verdict
    forced = second_slot_force()
    if forced == "bandit" and bandit_out:
        second_out = bandit_out
        second_why = (f"OPERATOR FORCED bandit into slot 2 "
                      f"(models/second_slot_force.json; measured: "
                      f"{bandit_verdict})")
        log(second_why)
    elif not bandit_passed:
        if forced == "bandit":
            log("second-slot force asked for the bandit but none was built "
                "-- falling back to the rule")
        try:
            div = stage_second(base_id, version, results, cand_paths)
        except Exception as exc:                                   # noqa: BLE001
            log(f"stage_second FAILED ({type(exc).__name__}: "
                f"{str(exc)[:120]})")
            div = None
        if div:
            second_out = div
            second_why = (f"diversity route (bandit had no sign-tested "
                          f"edge): {div}")
        elif bandit_out:
            second_out = bandit_out
            second_why = ("bandit keeps slot 2 by DEFAULT (no diversity "
                          "candidate built)")
            log(second_why)
    with open(os.path.join(ROOT, "models", "second_slot.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"version": str(version),
                   "second": second_out,
                   "bandit_passed": bool(bandit_passed),
                   "why": second_why,
                   "when": dt.datetime.now().isoformat(timespec="seconds")},
                  fh, indent=1)

    code, out = run([sys.executable, "src/kaggriculture/pipeline/submit.py", "--agent", route_out,
                     "--dry-run"], timeout=1800)
    gate = "PASSED" if "stopping here" in out else "CHECK LOG"
    report([
        verdict,
        f"- built `{route_out}` (v{version}), graph regenerated",
        f"- bandit: {bandit_verdict}",
        f"- slot 2: {second_why}",
        f"- submission gate ({os.path.basename(route_out)}): {gate}",
        f"- pair to ship: `{route_out}` + `{second_out or 'NONE'}`",
        "",
        "To ship (human decision):",
        "```",
        f"python -m kaggriculture.agentbuild.build_notebook --agent {route_out} --push",
        "# wait for the kernel run, verify sha, then:",
        "kaggle competitions submit kaggriculture -k <kernel> -v <N> -f main.py -m \"...\"",
        "```",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
