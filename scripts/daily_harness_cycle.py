"""DAILY harness + policy cycle -- runs until we are top 10 and stay there.

Operator order 2026-09-04: "The new model training and harness must be done
everyday to improve the harness till we are reaching and staying within top
10." This is that cycle. It NEVER submits; it builds, gates and reports.

Each run does four things, in this order, because each one feeds the next:

  1. GROW THE INSTRUMENT. Harvest every new game we played against opponents
     rated >= 2300 and fold it into the hard band. The band is the only
     offline instrument verified against the ladder (56/58 cells exact on
     2026-09-04), and it is also the training/eval set for everything else,
     so it must grow before anything is measured on it.

  2. RE-SCREEN THE ECONOMY. Dedupe every fresh winning route by opening
     prefix and score the distinct economies on the band. The base economy is
     the ONLY lever that has ever moved this agent (19 -> 26 credited wins on
     2026-09-04, larger than every layer ever built), and fresh routes appear
     daily.

  3. REBUILD THE CORPUS AND TRAIN. Re-simulate top-100 episodes from their
     seeds, capture per-turn state, and fit the policy on the segment AFTER
     each team's script ends -- measured per team, because the top of the
     leaderboard scripts a fixed opening and then decides at runtime, at a
     team-specific day (Crop Dusta d3, Jesse Bullard d6, MtN d12, Wang H2O
     d18) while the mid-field is scripted end to end. See
     docs/history/bc-policy-plan-2026-09-04.md.

  4. GATE AND REPORT. Everything is judged on hard-band CREDITED wins and
     median paired margin against the live agent. Nothing ships from here.

    python scripts/daily_harness_cycle.py            # full cycle
    python scripts/daily_harness_cycle.py --stages instrument,economy
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HB = os.path.join(ROOT, ".local", "hardband")
REPORT_DIR = os.path.join(ROOT, "data", "harness_cycle")
STAGES = ("instrument", "economy", "meta", "corpus", "train", "gate", "gauntlet", "elite", "bt", "hedge")


def run(cmd, label, timeout=7200):
    print(f"\n=== {label} ===", flush=True)
    t0 = dt.datetime.now()
    p = subprocess.run([sys.executable] + cmd, cwd=ROOT, timeout=timeout)
    dtm = (dt.datetime.now() - t0).total_seconds()
    print(f"  {label}: exit {p.returncode} in {dtm:.0f}s", flush=True)
    return p.returncode


def stage_instrument():
    """Grow the hard band with every new >=2300 game we have played."""
    rc = run([os.path.join(HB, "harvest.py"), "--download"], "harvest new 2300+ games")
    if rc == 0:
        run([os.path.join(HB, "build_band.py"), "--build"], "rebuild band")
        run([os.path.join(HB, "make_factory_panel.py")], "rebuild factory panel")
    return rc


def stage_economy():
    """Re-screen distinct economies on the band; the base is the real lever."""
    return run([os.path.join(HB, "score_bases_guarded.py")],
               "re-screen economies (guarded)")


def stage_corpus():
    """Top-100 BC corpus, INCREMENTAL: the resume guard in bc_corpus skips
    every (episode, seat) already sharded, and --rust runs the simulation on
    kagg serve (verified row-EXACT vs the official engine 2026-09-05)."""
    run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness", "bc_corpus.py"), "--scan"],
        "rescan script-end days")
    return run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness", "bc_corpus.py"), "--build",
                "--rust", "--workers", "6", "--top", "100",
                "--replay-dirs",
                os.path.join(ROOT, ".local", "lossreplays"),
                os.path.join(ROOT, ".local", "hardband", "replays")],
               "grow top-100 corpus (rust, replay-direct)")


def stage_train():
    py = os.environ.get("KAGG_TRAIN_PY",
                        "C:/ProgramData/anaconda3/envs/llm/python.exe")
    trainer = os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness", "bc_train_gru.py")       # Stage 2
    if not os.path.exists(trainer):
        trainer = os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness", "bc_arch_test.py")
    print(f"\n=== train policy ({os.path.basename(trainer)}) ===", flush=True)
    # two heads per cycle (operator 2026-09-05: use ALL leaderboard data):
    #   1. rank-1 GRU (the single team measured deeply learnable, 58.6%);
    #   2. all-reactive-team peak MLP (the shared skill: +10.1% cross-team
    #      on the peak slice vs -31% full-policy) on the 144-team corpus.
    p = subprocess.run([py, trainer], cwd=ROOT, timeout=14400)
    print(f"  train (rank-1 gru): exit {p.returncode}", flush=True)
    p2 = subprocess.run([py, os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                                          "bc_train.py"),
                         "--peak-only", "--epochs", "25",
                         "--tag", "allteams_peak"],
                        cwd=ROOT, timeout=14400)
    print(f"  train (cohort peak): exit {p2.returncode}", flush=True)
    return p.returncode or p2.returncode


def stage_gate():
    """Everything is judged against the LIVE agent on the hard band."""
    live = os.path.join(ROOT, "agents", "v44.0_bandit.py")
    return run([os.path.join(HB, "counterfactual.py"), "--rust",
                "--calibrate"],
               "recalibrate band on the live agent (rust, certified "
               "56/58 exact 2026-09-05)")




def stage_meta():
    """Replay-wave tracker: is the field cloning us, and who is the new wave?
    First run 2026-09-05: OUR opening = 12.3% of fresh seats, 192 teams."""
    return run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                             "meta_tracker.py"), "--days", "2"],
               "meta / replay-wave tracker")


def stage_gauntlet():
    """Zero-upset pre-flight: refresh upset cells and certify them."""
    return run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                             "upset_gauntlet.py"), "--certify"],
               "certify upset gauntlet")


def stage_elite():
    """Elite band: the live pair vs >=2500 teams' recorded worlds.

    6,874 certifiable cells across 201 elite teams (B4, 2026-09-05).
    Refreshes the team-score backfill first so the panel tracks today's
    leaderboard. Operator bar: 70%+ credited to sustain above 2500."""
    run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                      "backfill_team_scores.py")], "refresh team scores")
    return run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                             "elite_band.py"),
                os.path.join(ROOT, "agents", "v45.0_bandit.py"),
                "--cells", "80", "--per-team", "2"],
               "elite band (>=2500)")


def stage_bt():
    """Bradley-Terry rank the live pair + newest candidate (final metric)."""
    cands = [os.path.join(ROOT, "agents", "v44.1_bandit.py"),
             os.path.join(ROOT, "agents", "v45.0_bandit.py")]
    newest = os.path.join(ROOT, ".local", "candidates", "v451_flush.py")
    if os.path.exists(newest):
        cands.append(newest)
    return run([os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness", "bt_rank.py")]
               + cands + ["--seeds", "6"], "bt tournament")

def stage_hedge():
    """Propose (NEVER submit) the hedge-slot candidate.

    Host ruling 2026-09-05: final score = better of the two active
    submissions -- slot 2 is a hedge with no downside. This stage names the
    best gated candidate and prints the EXACT notebook submit sequence for
    the operator; eviction takes the OLDER active slot, so the report also
    says which live submission would fall.
    """
    bt = os.path.join(ROOT, "models", "trackp", "bt_rank.json")
    pick = None
    if os.path.exists(bt):
        d = json.load(open(bt, encoding="utf-8"))
        order = sorted(range(len(d["names"])),
                       key=lambda k: -d["strength"][k])
        for k in order:
            if "candidates" in str(d["names"][k]) or d["names"][k].startswith("v45"):
                pick = d["names"][k]
                break
    print("\n=== hedge proposal (NEVER submits) ===")
    print(f"  deadline 2026-09-23: "
          f"{(dt.date(2026, 9, 23) - dt.date.today()).days} days left")
    if pick:
        print(f"  BT-best candidate: {pick}")
        print("  operator sequence (bandit notebook path):")
        print("    python src/build_notebook.py --agent <candidate> "
              "--slug kaggriculture-route-refresh-price-impact --private --push")
        print("    kaggle kernels status ... until COMPLETE; verify sha; then")
        print("    kaggle competitions submit kaggriculture -k "
              "debmalya84/kaggriculture-route-refresh-price-impact "
              "-v <version> -f main.py -m \"...\"")
        print("  NOTE: submission evicts the OLDER active slot.")
    else:
        print("  no BT ranking on disk yet")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    want = [s.strip() for s in a.stages.split(",") if s.strip()]
    os.makedirs(REPORT_DIR, exist_ok=True)
    started = dt.datetime.now()
    results = {}
    for s in want:
        fn = globals().get(f"stage_{s}")
        if fn is None:
            print(f"unknown stage {s!r}"); continue
        try:
            results[s] = fn()
        except subprocess.TimeoutExpired:
            print(f"  {s}: TIMEOUT"); results[s] = "timeout"
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {s}: {type(exc).__name__}: {exc}"); results[s] = "error"
    rep = {"when": started.isoformat(timespec="seconds"),
           "stages": results,
           "minutes": round((dt.datetime.now() - started).total_seconds() / 60, 1)}
    out = os.path.join(REPORT_DIR,
                       f"cycle_{started:%Y%m%d_%H%M}.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\ncycle report -> {os.path.relpath(out, ROOT)}")
    print("NEVER SUBMITS. Read hard-band CREDITED wins vs the live agent "
          "(19/56 on 2026-09-04) before shipping anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
