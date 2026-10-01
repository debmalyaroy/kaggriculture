"""Autopilot: the continuous, self-improving Slot-2 pipeline with STATUS signals.

One command runs the whole loop unattended:
  [ train a round -> gate BOTH harnesses -> lock configs on pass -> release ]  x N
resuming yesterday's best each cycle (league grows -> keeps improving without new
data). It NEVER submits — it produces release candidates + a submit checklist and
tells you, via a STATUS file, exactly where it is.

STATUS (this answers "how do I know?"):
  * models/rl/AUTOPILOT_STATUS.json  — machine-readable state, rewritten every phase
  * models/rl/AUTOPILOT_STATUS.md    — the same, human-readable
    Both carry: phase, cycle, last gate result PER HARNESS (per-band + aggregate +
    ship verdict), whether a RELEASE CANDIDATE exists (+ which version/notebook),
    the training curve, and the next action. Run `--status` to print it.
  * A RELEASE CANDIDATE also drops a file: .local/candidates/RELEASE_CANDIDATE_<ver>.md
    and updates .local/candidates/SUBMIT_INSTRUCTIONS.md — so a candidate is
    impossible to miss.

CONTROL:
  * start:  python -m kaggriculture.pipeline.autopilot            (foreground)
            (run detached with nohup/Start-Process for true autopilot)
  * status: python -m kaggriculture.pipeline.autopilot --status
  * stop:   python -m kaggriculture.pipeline.autopilot --stop     (graceful, after
            the current phase) — or delete the process. A STOP flag file is honoured.

Memory-safe: one training job at a time, compile workers hard-capped (default 3).
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

RLDIR = os.path.join(ROOT, "models", "rl")
STATUS_JSON = os.path.join(RLDIR, "AUTOPILOT_STATUS.json")
STATUS_MD = os.path.join(RLDIR, "AUTOPILOT_STATUS.md")
STOP_FLAG = os.path.join(RLDIR, "AUTOPILOT_STOP")
HEARTBEAT = os.path.join(RLDIR, "AUTOPILOT_HEARTBEAT")
CAND = os.path.join(ROOT, ".local", "candidates")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_status() -> dict:
    if os.path.exists(STATUS_JSON):
        try:
            return json.load(open(STATUS_JSON, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def write_status(**kw):
    """Merge kw into the status file (JSON + human MD). Always safe to call."""
    os.makedirs(RLDIR, exist_ok=True)
    st = _read_status()
    st.update(kw)
    st["updated"] = _now()
    with open(STATUS_JSON, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    # human-readable mirror
    g = st.get("last_gate", {})
    rc = st.get("release_candidates", {})
    lines = [
        f"# Autopilot status — {st.get('updated')}",
        "",
        f"**State:** {st.get('state', '?')}   |   **Phase:** {st.get('phase', '?')}"
        f"   |   **Cycle:** {st.get('cycle', 0)}",
        f"**Next version:** {st.get('next_version', '?')}   |   "
        f"**Best gate agg:** {st.get('best_gate_agg', '—')}",
        "",
        "## Release candidates (a submit is ready when these say a version)",
        f"- **BANDIT:** {rc.get('bandit') or 'none yet'}"
        f"  (notebook: kaggriculture-private-submission-bandit)",
        f"- **TRACK-P:** {rc.get('trackp') or 'none yet'}"
        f"  (notebook: kaggriculture-private-submission-trackp)",
        "",
        "## Last gate result (per harness)",
    ]
    for h in ("bandit", "trackp"):
        gh = g.get(h) or {}
        lines.append(
            f"- **{h}**: verdict **{gh.get('verdict', '—')}**  agg "
            f"{gh.get('aggregate', '—')}  bands {gh.get('bands', {})}")
    lines += [
        "",
        f"## Training  (curve: {st.get('train_curve', [])[-6:]})",
        f"- last round battery/gate: {st.get('last_train', '—')}",
        f"- league size: {st.get('league_size', '—')}   games this session: "
        f"{st.get('games', 0)}",
        "",
        f"## Next action\n{st.get('next_action', '—')}",
        "",
        f"_heartbeat: {st.get('heartbeat', '—')}_",
    ]
    with open(STATUS_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        fh.write(_now())
    return st


def print_status():
    if not os.path.exists(STATUS_MD):
        print("[autopilot] no status yet — not started.")
        return 0
    print(open(STATUS_MD, encoding="utf-8").read())
    # liveness hint
    if os.path.exists(HEARTBEAT):
        age = time.time() - os.path.getmtime(HEARTBEAT)
        print(f"(heartbeat {age/60:.1f} min ago — "
              f"{'LIKELY RUNNING' if age < 900 else 'STALE / stopped?'})")
    return 0


def _device(cpu):
    if cpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


GM_PARTS = os.path.join(ROOT, ".local", "scratch", "gm", "top100_replays_parts")
SUPERVISOR = os.path.join(ROOT, ".local", "scratch", "gm", "download_supervisor.sh")


def _refresh_data(a, device, cfg):
    """Fetch + fold NEW downloaded games into the corpus, then re-warm BC.

    The download supervisor (CDN-gated) drains new top-200 replays; here we detect
    growth in the corpus parts, rebuild `bc_corpus` (richer macro corpus + opponent
    pool), and set fresh_bc so the next BC learns from the enlarged set. Best-effort
    and bounded — never fatal to the loop."""
    import kaggriculture.pipeline.slot2_loop as LOOP
    # (a) DELTA-ingest the georgymamarin full-ladder dataset (only NEW shards/episodes)
    try:
        import kaggriculture.data.gm_episodes as GM
        # (a0) DELTA DOWNLOAD first when --fetch is set: pull only new/changed shards
        #      (frozen months skipped by size match). No-op if Kaggle API unavailable.
        if cfg.get("fetch"):
            try:
                got = GM.delta_download(verbose=False)
                if got:
                    write_status(gm_download=f"delta-downloaded {len(got)} shard(s)")
            except Exception as e:
                write_status(gm_note=f"gm delta-download skipped: {str(e)[:100]}")
        if os.path.exists(os.path.join(GM.DATA, "episodes.csv")):
            n = GM.ingest(min_rating=cfg.get("gm_min_rating", 2200.0), verbose=False)
            write_status(gm_rows=n, data_note=f"gm delta-ingest → macro_gm {n:,} rows")
            a.fresh_bc = True                     # new data → re-warm on next BC
    except Exception as e:
        write_status(gm_note=f"gm ingest skipped: {str(e)[:120]}")
    # (b) fold new top-200 download replays if the supervisor drained any
    parts = len(glob.glob(os.path.join(GM_PARTS, "*.parquet")))
    last = _read_status().get("corpus_parts", -1)
    write_status(phase="data", next_action=f"data refresh: {parts} corpus parts "
                 f"(was {last})")
    if parts > 0 and parts != last:
        try:
            subprocess.run([sys.executable, "-m", "kaggriculture.data.bc_corpus"],
                           cwd=ROOT, env=dict(os.environ, PYTHONPATH="src",
                                              PYTHONUTF8="1"),
                           capture_output=True, text=True, timeout=1800)
            a.fresh_bc = True
            LOOP.stage_bc(a, device)          # re-warm on the bigger corpus
            a.fresh_bc = False
            write_status(data_note=f"corpus rebuilt from {parts} parts; BC re-warmed")
        except Exception as e:
            write_status(data_note=f"corpus rebuild skipped: {str(e)[:120]}")
    write_status(corpus_parts=parts)


def _distill(cfg, device):
    """Generate self-play gameplays (tape+world gen) → keep winners → grow the BC
    corpus, and re-warm BC on it (AlphaZero-style self-distillation)."""
    import random
    import kaggriculture.train.bc_warmup as BC
    import kaggriculture.train.selfplay_corpus as SP
    from kaggriculture.train.macro_env import BatchRoller, OpponentPool
    from kaggriculture.train.worlds import WorldSampler
    import kaggriculture.train.macro_rl as RL
    import torch
    ckpt = RL.RL_OUT if os.path.exists(RL.RL_OUT) else BC.OUT_PATH
    model, ck = BC.load_policy(ckpt, device)
    ls = ck.get("log_std")
    ls = ls.to(device) if hasattr(ls, "to") else torch.zeros(BC.VEC_LEN, device=device)
    pool = OpponentPool()
    br = BatchRoller(pool=pool)
    opp = [e for e in pool.entries if e.get("kind") == "destbreso_tape"][:20]
    write_status(phase="distill", next_action="generating self-play gameplays "
                 "(tape+world gen) → distilling winners into the BC corpus")
    try:
        n = SP.generate(model, ls, ck["norm"], device, br, WorldSampler(),
                        random.Random(int(time.time())),
                        n_games=cfg["distill_games"], keep_frac=cfg["distill_keep"],
                        opp_pool=opp)
        write_status(distill_rows=n)
    except Exception as e:
        write_status(distill_note=f"skipped: {str(e)[:120]}")


def _start_supervisor():
    """Kick the CDN-aware download supervisor (drains new games on the 429 reset)."""
    if not os.path.exists(SUPERVISOR):
        return
    try:
        subprocess.Popen(["bash", SUPERVISOR], cwd=ROOT,
                         stdout=open(os.path.join(ROOT, ".local", "scratch", "gm",
                                                  "supervisor.out"), "a"),
                         stderr=subprocess.STDOUT)
        write_status(fetch="download supervisor started (drains on CDN reset)")
    except Exception as e:
        write_status(fetch=f"supervisor start failed: {str(e)[:120]}")


def _args_bag(a):
    """A lightweight arg object for the daily_slot2 / slot2_loop stage fns."""
    class A:
        pass
    o = A()
    o.__dict__.update(a)
    return o


def run(cfg):
    import kaggriculture.pipeline.slot2_loop as LOOP
    import kaggriculture.pipeline.daily_slot2 as DAILY
    import kaggriculture.pipeline.release as REL
    import kaggriculture.train.macro_rl as RL

    device = _device(cfg["cpu"])
    a = _args_bag(dict(
        cpu=cfg["cpu"], fresh_bc=cfg["fresh_bc"], d_model=cfg["d_model"],
        layers=cfg["layers"], heads=cfg["heads"], batch_size=cfg["batch_size"],
        bc_epochs=cfg["bc_epochs"], iters=cfg["iters"],
        games_per_iter=cfg["games_per_iter"], games_per_plan=cfg["games_per_plan"],
        ppo_epochs=cfg["ppo_epochs"], rl_lr=cfg["rl_lr"], ent_coef=cfg["ent_coef"],
        kl_coef=cfg["kl_coef"], init_std=cfg["init_std"],
        margin_shaping=cfg["margin_shaping"], battery=cfg["battery"],
        no_batch=False, no_self_play=False, seed=cfg["seed"], refs="",
        gate_seeds=cfg["gate_seeds"], bar=cfg["bar"], rounds=1, resume=True,
        ship_only=False, force_ship=False, skip=[], no_route2=cfg["no_route2"]))

    if os.path.exists(STOP_FLAG):
        os.remove(STOP_FLAG)
    if cfg.get("fetch"):
        _start_supervisor()
    write_status(state="starting", phase="bc", cycle=0, device=device,
                 next_version=REL.peek_version(), release_candidates={},
                 last_gate={}, train_curve=[], games=0,
                 next_action="BC warmup, then the train->gate->release loop")
    t0 = time.time()

    # BC once (unless a checkpoint exists / fresh requested)
    try:
        LOOP.stage_bc(a, device)
    except Exception as e:
        write_status(state="error", phase="bc", error=str(e)[:300])
        raise

    st = LOOP._load_state()
    curve = []
    games = 0
    no_gain = 0
    for cycle in range(1, cfg["max_cycles"] + 1):
        # ---- DATA REFRESH: fetch + rebuild corpus periodically (answers "how do
        # we fetch more data and train") ----
        if cfg["refresh_every"] and cycle > 1 and (cycle - 1) % cfg["refresh_every"] == 0:
            _refresh_data(a, device, cfg)
        # ---- SELF-DISTILLATION: generate gameplays → grow BC corpus → re-warm ----
        if cfg["distill_every"] and cycle > 1 and (cycle - 1) % cfg["distill_every"] == 0:
            _distill(cfg, device)
            a.fresh_bc = True                     # learn from the new self-play games
            LOOP.stage_bc(a, device)
            a.fresh_bc = False
        if os.path.exists(STOP_FLAG):
            write_status(state="stopped", phase="idle",
                         next_action="STOP flag seen — exiting cleanly.")
            os.remove(STOP_FLAG)
            break
        # ---- TRAIN a round ----
        write_status(state="running", phase="train", cycle=cycle,
                     next_action=f"training cycle {cycle} "
                     f"({cfg['iters']} iters x {cfg['games_per_iter']} games)")
        try:
            res = LOOP.stage_rl(a, device, st.get("round", 0))
        except Exception as e:
            write_status(state="error", phase="train", error=str(e)[:300],
                         traceback=traceback.format_exc()[-800:])
            raise
        st = LOOP._load_state()
        games += cfg["iters"] * cfg["games_per_iter"]
        bw = (res or {}).get("best_winrate")
        curve.append(round(bw, 3) if bw is not None else None)
        import glob
        league_size = len(glob.glob(os.path.join(RL.LEAGUE_DIR, "*.tape")))

        # ---- PACKAGE + GATE both harnesses ----
        write_status(state="running", phase="gate", cycle=cycle,
                     train_curve=curve, games=games, league_size=league_size,
                     last_train=bw, next_action="gating both harnesses vs the crown panel")
        LOOP.stage_package(prefer_rl=True)
        bandit = DAILY.stage_bandit(a, device)
        trackp = DAILY.stage_trackp(a, device)

        # ---- RELEASE (lock configs on ship, write candidate files) ----
        write_status(state="running", phase="release", cycle=cycle)
        version, seats = DAILY.stage_release(a, None, bandit, trackp, device)
        last_gate = {h: {"aggregate": (seats[h] or {}).get("gate"),
                         "bands": (seats[h] or {}).get("bands"),
                         "verdict": "ship" if (seats[h] or {}).get("ship") else "hold"}
                     for h in ("bandit", "trackp")}
        rc = {h: (version if (seats[h] or {}).get("ship") else None)
              for h in ("bandit", "trackp")}
        # drop an unmissable RELEASE CANDIDATE file
        for h in ("bandit", "trackp"):
            if rc[h]:
                p = os.path.join(CAND, f"RELEASE_CANDIDATE_{version}_{h}.md")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(f"# RELEASE CANDIDATE {version} — {h}\n\n"
                             f"Gate PASSED ({_now()}). Locked config: "
                             f"{(seats[h] or {}).get('locked')}\n"
                             f"Submit via notebook kaggriculture-private-submission-{h} "
                             f"as version {version}. See SUBMIT_INSTRUCTIONS.md.\n")
        best = _read_status().get("best_gate_agg")
        aggs = [v.get("aggregate") for v in last_gate.values() if v.get("aggregate") is not None]
        cur_best = max(aggs) if aggs else None
        improved = cur_best is not None and (best is None or cur_best > (best or 0) + cfg["min_gain"])
        if improved:
            best = cur_best
            no_gain = 0                                 # reset ONLY on real improvement
        else:
            no_gain += 1

        # ---- OVERFIT WATCH: the gate is on HELD-OUT crown refs (never trained on)
        # + BC early-stops on its val split, so overfitting shows as train_win >>
        # gate. Warn (and it counts as no-gain, so it can't run forever). ----
        overfit = (bw is not None and cur_best is not None
                   and bw - cur_best > cfg["overfit_gap"])
        if overfit:
            write_status(overfit_warn=f"train_win {bw:.2f} >> gate {cur_best:.2f}: "
                         "possible overfit to training opponents — the held-out crown "
                         "gate is the guard; best checkpoint (rl_policy_best.pt) is kept")

        # ---- AUTO-IMPROVE: escalate budget/knobs when the gate stalls ----
        escalations = _read_status().get("escalations", 0)
        if not any(rc.values()) and no_gain > 0 and no_gain % cfg["escalate_after"] == 0 \
                and escalations < cfg["max_escalations"]:
            a.iters = min(int(a.iters * 1.5), cfg["iters_cap"])
            a.games_per_iter = min(int(a.games_per_iter * 1.25), cfg["games_cap"])
            a.kl_coef = max(a.kl_coef * 0.7, 0.02)      # drift further from BC
            a.init_std = min(a.init_std * 1.15, 0.6)    # explore more
            escalations += 1
            if escalations % 3 == 0:                    # deep plateau -> bigger fresh BC
                a.fresh_bc = True
                a.d_model = min(a.d_model + 64, 512)
                LOOP.stage_bc(a, device)
                a.fresh_bc = False
            write_status(escalations=escalations,
                         escalation_note=f"stalled → iters={a.iters} "
                         f"games/iter={a.games_per_iter} kl={a.kl_coef:.3f} "
                         f"std={a.init_std:.2f} d_model={a.d_model}")

        # ---- CONVERGENCE LADDER: knobs stalled → DATA (distill + download) →
        # resume; only truly STOP when even new data/gameplays don't help ----
        data_boosts = _read_status().get("data_boosts", 0)
        if not any(rc.values()) and escalations >= cfg["max_escalations"] \
                and no_gain >= cfg["patience"]:
            if data_boosts < cfg["max_data_boosts"]:
                write_status(phase="data_boost", data_boosts=data_boosts + 1,
                             next_action=f"knob escalations exhausted with no gate gain "
                             f"→ DATA BOOST {data_boosts + 1}: generate self-play "
                             "gameplays + fold new downloads → re-warm BC → RESUME")
                _distill(cfg, device)                 # generate more gameplays
                if cfg.get("fetch"):
                    _refresh_data(a, device, cfg)     # fold any new downloaded games
                a.fresh_bc = True
                LOOP.stage_bc(a, device)              # re-warm on the bigger corpus
                a.fresh_bc = False
                escalations = 0                       # reset the ladder; give data a chance
                no_gain = 0
                write_status(escalations=0)
            else:
                write_status(state="converged", phase="idle", best_gate_agg=best,
                             next_action=f"CONVERGED — no gate gain after "
                             f"{escalations} escalations + {data_boosts} data boosts "
                             f"(best {best}). Best checkpoint kept. Genuinely out of "
                             "automatic levers; needs new configs / external data.")
                print(f"[autopilot] CONVERGED at cycle {cycle}; best gate {best}")
                break

        na = ("RELEASE CANDIDATE ready — submit via the notebook(s)."
              if any(rc.values())
              else "no candidate yet (gate below bar) — training continues.")
        write_status(state="running", phase="idle", cycle=cycle,
                     next_version=REL.peek_version(), last_gate=last_gate,
                     release_candidates=rc, best_gate_agg=best,
                     train_curve=curve, games=games, league_size=league_size,
                     seconds=round(time.time() - t0), next_action=na)
        # optional pacing between cycles
        if cfg["cycle_sleep"] > 0:
            time.sleep(cfg["cycle_sleep"])

    write_status(state="done", phase="idle",
                 next_action="max cycles reached — restart to continue.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true", help="print current status")
    ap.add_argument("--stop", action="store_true", help="signal a graceful stop")
    ap.add_argument("--max-cycles", type=int, default=1000)
    ap.add_argument("--cycle-sleep", type=int, default=0)
    ap.add_argument("--bar", type=float, default=0.9)
    # auto-improve (escalate when the gate stalls)
    ap.add_argument("--escalate-after", type=int, default=3,
                    help="cycles with no gate improvement before escalating budget/knobs")
    ap.add_argument("--iters-cap", type=int, default=200)
    ap.add_argument("--games-cap", type=int, default=1024)
    # convergence / early-stop (don't train forever / overfit)
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="minimum gate-aggregate improvement to count as progress")
    ap.add_argument("--patience", type=int, default=12,
                    help="cycles with no gate gain (after escalations exhausted) → STOP")
    ap.add_argument("--max-escalations", type=int, default=6,
                    help="knob-escalation rounds before trying a DATA boost")
    ap.add_argument("--max-data-boosts", type=int, default=3,
                    help="data boosts (self-play + download) before truly stopping")
    ap.add_argument("--overfit-gap", type=float, default=0.35,
                    help="warn if train_win exceeds gate aggregate by more than this")
    # data (fetch + fold new games in periodically)
    ap.add_argument("--fetch", action="store_true",
                    help="start the download supervisor (drains new games on CDN reset)")
    ap.add_argument("--refresh-every", type=int, default=0,
                    help="every N cycles, DELTA-ingest new gm/download data + re-warm BC (0=off)")
    ap.add_argument("--gm-min-rating", type=float, default=2200.0,
                    help="rating filter for the georgymamarin dataset delta-ingest")
    # self-distillation (generate gameplays → grow BC corpus)
    ap.add_argument("--distill-every", type=int, default=5,
                    help="every N cycles, generate self-play gameplays → distil winners into BC (0=off)")
    ap.add_argument("--distill-games", type=int, default=200)
    ap.add_argument("--distill-keep", type=float, default=0.3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fresh-bc", action="store_true")
    ap.add_argument("--no-route2", action="store_true")
    # BC
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    # RL per cycle
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--games-per-plan", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--rl-lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--margin-shaping", type=float, default=0.3)
    ap.add_argument("--battery", type=int, default=8)
    ap.add_argument("--gate-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.status:
        return print_status()
    if args.stop:
        os.makedirs(RLDIR, exist_ok=True)
        open(STOP_FLAG, "w").write(_now())
        print("[autopilot] STOP flag set — it will exit after the current phase.")
        return 0
    return run(vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
