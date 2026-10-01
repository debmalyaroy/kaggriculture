"""Kaggriculture end-to-end improvement pipeline.

    fetch  ->  analyze  ->  refine  ->  evaluate  ->  [improve]*  ->  submit

1. **fetch**    top-ladder replays from Kaggle's daily episode datasets, plus
                the episodes your own submissions have played
2. **analyze**  extract what the winners actually do (portfolio, hiring curve,
                land timing, what they sell) and diff it against our agent
3. **refine**   blend those observations into the agent's PARAMS and write
                agents/agent_v2_<timestamp>.py
4. **evaluate** play it under the *exact* configuration Kaggle uses
5. **improve**  optional self-play search rounds, as many as you want
6. **submit**   only after you approve, interactively

Submission is never automatic. The pipeline stops at a prompt, shows the test
statistics, and waits for you to choose.

Usage
-----
    python -m kaggriculture.pipeline.pipeline                       # full run
    python -m kaggriculture.pipeline.pipeline --skip-fetch          # reuse replays already on disk
    python -m kaggriculture.pipeline.pipeline --offline             # no Kaggle at all: tune + evaluate
    python -m kaggriculture.pipeline.pipeline --seeds 6 --improve-budget 30
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.train.refine as refiner  # noqa: E402
import kaggriculture.data.replay_analysis as ra  # noqa: E402

LOCAL = os.path.join(ROOT, ".local")
ANALYSIS_DIR = os.path.join(LOCAL, "analysis")
EPISODE_DIR = os.path.join(LOCAL, "episodes")
BUILTINS = {"pass", "random", "starter"}

# Stock competition configuration, used when no real replay is available to
# copy one from. Matches kaggriculture.json defaults.
STOCK_CONFIG = {"episodeSteps": 720, "actTimeout": 1, "runTimeout": 1200}


# ---------------------------------------------------------------- utilities
def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def money(x):
    return f"${x:,.0f}"


def ask(prompt, default=""):
    """input() that degrades to `default` when there is no console.

    The pipeline is often run with piped stdin or from a scheduler; an
    unhandled EOFError there would look like a crash rather than "nobody
    answered". Never used to approve a submission -- see stage_submit.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  (no interactive input available)")
        return default


# ------------------------------------------------------------------ 1 fetch
def stage_fetch(args):
    if args.offline or args.skip_fetch:
        top = _existing(os.path.join(EPISODE_DIR, "top"))
        mine = _existing(os.path.join(EPISODE_DIR, "mine"))
        print(f"  reusing {len(top)} top and {len(mine)} own replays already on disk")
        return top, mine

    import kaggriculture.data.episodes as episodes
    hr("STAGE 1/6  fetch replays from Kaggle")
    try:
        top = episodes.fetch_top_episodes(
            EPISODE_DIR, days=args.days, per_day=args.per_day,
            max_bytes=int(args.max_mb * 1e6) if args.max_mb else None)
    except episodes.KaggleError as exc:
        print(f"  ! could not fetch top replays: {exc}")
        print("  continuing with whatever is already on disk")
        top = _existing(os.path.join(EPISODE_DIR, "top"))
    try:
        mine = episodes.fetch_own_episodes(EPISODE_DIR, limit=args.own)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  ! could not fetch your own replays: {exc}")
        mine = _existing(os.path.join(EPISODE_DIR, "mine"))
    return top, mine


def _existing(directory):
    out = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".json") and f != "manifest.csv":
                out.append(os.path.join(root, f))
    return sorted(out)


# ---------------------------------------------------------------- 2 analyze
def stage_analyze(args, top_paths, mine_paths, our_agent):
    hr("STAGE 2/6  analyze replays")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    def parse_all(paths, label):
        profs = []
        for path in paths[: args.max_replays]:
            try:
                print(f"  parsing {label}: {os.path.basename(path)} "
                      f"({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
                profs.append(ra.profile_replay(path))
            except Exception as exc:                               # noqa: BLE001
                print(f"    ! {exc}")
        return [p for p in profs if p]

    top_profiles = parse_all(top_paths, "top")
    mine_profiles = parse_all(mine_paths, "yours")

    top_agg = ra.aggregate(top_profiles, only_winners=True)
    mine_agg = ra.aggregate(mine_profiles, only_winners=False)

    # Our current agent, measured the same way: play one match and profile it.
    print("  profiling our current agent (one local match) ...", flush=True)
    ours_agg = _profile_our_agent(our_agent, args.config)

    report_path = os.path.join(ANALYSIS_DIR, f"analysis_{args.timestamp}.md")
    ra.render_report(top_agg, mine_agg, ours_agg, report_path)
    if top_agg:
        with open(os.path.join(ANALYSIS_DIR, f"top_profile_{args.timestamp}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(top_agg, f, indent=1, default=str)
    print(f"  report -> {report_path}")

    if top_agg:
        print(f"\n  top ladder median bank {money(top_agg['final_bank'])} "
              f"| herd {top_agg['peak_herd']:.0f} "
              f"| crops {top_agg['peak_crops']:.0f} "
              f"| hands {top_agg['peak_hands']:.0f}")
    if ours_agg:
        print(f"  our agent        bank {money(ours_agg['final_bank'])} "
              f"| herd {ours_agg['peak_herd']:.0f} "
              f"| crops {ours_agg['peak_crops']:.0f} "
              f"| hands {ours_agg['peak_hands']:.0f}")
    return top_agg, mine_agg, ours_agg, report_path


def _profile_our_agent(agent_path, config):
    from kaggle_environments import make
    env = make("kaggriculture", configuration=dict(config), debug=False)
    env.run([os.path.abspath(agent_path), "starter"])
    tmp = os.path.join(ANALYSIS_DIR, "_ours.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(env.toJSON(), f)
    prof = ra.profile_replay(tmp)
    os.remove(tmp)
    return ra.aggregate([prof], only_winners=False) if prof else None


# ----------------------------------------------------------------- 3 refine
def stage_refine(args, top_agg, base_agent):
    hr("STAGE 3/6  refine the agent from what the ladder does")
    base = paramio.load(base_agent)
    new, changes = refiner.derive_params(base, top_agg, args.learn_rate)
    if changes:
        for c in changes:
            print("  " + c)
    else:
        print("  no parameter changes derived")
    prov = (f"Derived from {top_agg['n']} top-ladder player-games "
            f"(median bank {money(top_agg['final_bank'])}), blended at "
            f"learn-rate {args.learn_rate:g} onto {os.path.basename(base_agent)}."
            if top_agg else
            f"No ladder data available; parameters inherited from "
            f"{os.path.basename(base_agent)}.")
    path = refiner.write_agent(os.path.join(ROOT, "agents", "v1_heuristic.py"),
                               new, os.path.join(ROOT, "agents"),
                               timestamp=args.timestamp, provenance=prov)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


# --------------------------------------------------------------- 4 evaluate
def _mute_stdin():
    """Detach worker processes from stdin.

    Forked pool workers inherit fd 0 and share its file offset, so they quietly
    consume whatever the user has typed (or piped) while matches are running --
    and the approval prompt afterwards then hits EOF. Pointing the child's fd 0
    at /dev/null keeps the parent's stdin intact for the prompt.
    """
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
    except OSError:
        pass


def _play(job):
    left, right, seed, config = job
    from kaggle_environments import make
    cfg = dict(config)
    if seed is not None:
        cfg["seed"] = seed
    env = make("kaggriculture", configuration=cfg, debug=False)
    env.run([left, right])
    final = env.steps[-1]
    return (float(final[0]["reward"] or 0), float(final[1]["reward"] or 0),
            final[0]["status"], final[1]["status"])


def evaluate(agent, opponents, seeds, config, workers):
    """Paired evaluation: every opponent x seed, played in both seats."""
    jobs, meta = [], []
    me = os.path.abspath(agent)
    for opp in opponents:
        o = opp if opp in BUILTINS else os.path.abspath(opp)
        for s in seeds:
            jobs.append((me, o, s, config)); meta.append((opp, 0))
            jobs.append((o, me, s, config)); meta.append((opp, 1))
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_mute_stdin) as ex:
        results = list(ex.map(_play, jobs))

    per_opp, bad_status = {}, []
    for (opp, seat), (r0, r1, s0, s1) in zip(meta, results):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        my_status = s0 if seat == 0 else s1
        if my_status not in ("DONE", "ACTIVE"):
            bad_status.append((opp, my_status))
        per_opp.setdefault(opp, []).append((mine, theirs))

    table, all_mine, all_margin = [], [], []
    for opp, rows in per_opp.items():
        mine = [m for m, _ in rows]
        theirs = [t for _, t in rows]
        wins = sum(1 for m, t in rows if m > t)
        table.append({
            "opponent": opp, "n": len(rows),
            "win_pct": 100.0 * wins / len(rows),
            "my_mean": statistics.mean(mine),
            "opp_mean": statistics.mean(theirs),
            "margin": statistics.mean(mine) - statistics.mean(theirs),
        })
        all_mine += mine
        all_margin += [m - t for m, t in rows]
    return {
        "table": table,
        "mean_bank": statistics.mean(all_mine),
        "mean_margin": statistics.mean(all_margin),
        "win_pct": 100.0 * sum(1 for m in all_margin if m > 0) / len(all_margin),
        "bad_status": bad_status,
    }


def stage_evaluate(args, agent, opponents, label="STAGE 4/6  evaluate"):
    hr(f"{label}   ({os.path.basename(agent)})")
    print(f"  configuration: {json.dumps(args.config)}")
    print(f"  {len(args.seeds)} seeds x both seats x {len(opponents)} opponents"
          f" = {len(args.seeds) * 2 * len(opponents)} matches\n", flush=True)
    t0 = time.time()
    stats = evaluate(agent, opponents, args.seeds, args.config, args.workers)
    stats["elapsed"] = time.time() - t0
    stats["timing"] = _turn_timing(agent, args.config)
    print_stats(stats)
    return stats


def _turn_timing(agent_path, config):
    """Worst/mean turn latency against the 1-second actTimeout."""
    import importlib.util
    from kaggle_environments import make
    spec = importlib.util.spec_from_file_location("cand", os.path.abspath(agent_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise RuntimeError(f"{agent_path} defines no `agent` function -- "
                           f"Kaggle would reject this submission")
    worst, total, n = 0.0, 0.0, 0
    two_arg = mod.agent.__code__.co_argcount > 1

    def wrapped(obs, cfg=None):
        nonlocal worst, total, n
        t0 = time.time()
        out = mod.agent(obs, cfg) if two_arg else mod.agent(obs)
        dt = time.time() - t0
        worst = max(worst, dt); total += dt; n += 1
        return out

    cfg = dict(config)
    cfg["actTimeout"] = 60          # measure, don't forfeit, while timing
    env = make("kaggriculture", configuration=cfg, debug=False)
    env.run([wrapped, "starter"])
    return {"mean_ms": 1000 * total / max(1, n), "worst_ms": 1000 * worst}


def print_stats(stats):
    print(f"  {'opponent':<34}{'win%':>7}{'our mean':>12}{'opp mean':>12}{'margin':>12}")
    print("  " + "-" * 75)
    for row in sorted(stats["table"], key=lambda r: -r["margin"]):
        print(f"  {row['opponent'][:34]:<34}{row['win_pct']:>6.0f}%"
              f"{row['my_mean']:>12,.0f}{row['opp_mean']:>12,.0f}"
              f"{row['margin']:>+12,.0f}")
    print("  " + "-" * 75)
    print(f"  {'OVERALL':<34}{stats['win_pct']:>6.0f}%"
          f"{stats['mean_bank']:>12,.0f}{'':>12}{stats['mean_margin']:>+12,.0f}")
    t = stats.get("timing") or {}
    if t:
        verdict = "OK" if t["worst_ms"] < 500 else "TOO SLOW"
        print(f"\n  turn latency: mean {t['mean_ms']:.1f} ms, worst "
              f"{t['worst_ms']:.1f} ms vs 1000 ms actTimeout  [{verdict}]")
    if stats.get("bad_status"):
        print(f"  !! non-DONE statuses: {stats['bad_status']}")
    print(f"  evaluated in {stats.get('elapsed', 0):.0f}s")


# ---------------------------------------------------------------- 5 improve
def stage_improve(args, agent, rounds_done):
    hr(f"STAGE 5/6  improve  (round {rounds_done + 1})")
    # Each round writes a NEW file. Pointing --out at --base makes the tuner
    # rewrite the source it is still reading from, which corrupts it midway.
    out = os.path.join(ROOT, "agents",
                       f"agent_v2_{args.timestamp}_r{rounds_done + 1}.py")
    cmd = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "train", "tune.py"),
           "--base", os.path.abspath(agent),
           "--out", out,
           "--seeds", str(max(3, len(args.seeds) - 1)),
           "--passes", "1",
           "--workers", str(args.workers),
           "--budget-min", str(args.improve_budget),
           "--work", os.path.join(LOCAL, "tune")]
    print("  " + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, check=False)
    if not os.path.exists(out):
        print("  ! tuning produced no output; keeping the previous candidate")
        return agent
    try:
        paramio.load(out)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  ! tuned agent is unreadable ({exc}); keeping the previous one")
        return agent
    return out


# ----------------------------------------------------------------- 6 submit
def stage_submit(args, agent, stats):
    hr("STAGE 6/6  submit to Kaggle")
    build = os.path.join(ROOT, "build")
    os.makedirs(build, exist_ok=True)
    main_py = os.path.join(build, "main.py")
    shutil.copy(agent, main_py)
    print(f"  packaged {os.path.relpath(agent, ROOT)} -> build/main.py "
          f"({os.path.getsize(main_py):,} bytes)")

    if stats["bad_status"]:
        print("  !! refusing to submit: the agent did not finish cleanly in "
              "local evaluation")
        return False
    if (stats.get("timing") or {}).get("worst_ms", 0) >= 500:
        print("  !! refusing to submit: worst turn is too close to the 1 s "
              "actTimeout")
        return False

    msg = args.message or f"pipeline {args.timestamp} " \
                          f"bank {stats['mean_bank']:,.0f} " \
                          f"margin {stats['mean_margin']:+,.0f}"
    cmd = ["kaggle", "competitions", "submit", "kaggriculture",
           "-f", main_py, "-m", msg]
    print("  " + " ".join(cmd))
    confirm = ask("\n  Type SUBMIT to send this to Kaggle: ")
    if confirm != "SUBMIT":
        print("  not submitted")
        return False
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout or p.stderr)
    if p.returncode == 0:
        print("  submitted. Track it with:")
        print("    kaggle competitions submissions kaggriculture")
        return True
    print("  !! submission failed (have you joined the competition in a browser?)")
    return False


# -------------------------------------------------------------------- main
def resolve_config(args, top_paths):
    """The configuration to evaluate under.

    Preference order:
      1. copied verbatim out of a real downloaded Kaggle replay -- as close to
         "what happens on Kaggle" as it is possible to get locally
      2. the stock competition defaults
    """
    if not args.stock_config:
        for path in top_paths[:1]:
            try:
                with open(path, encoding="utf-8") as f:
                    head = json.load(f)
                cfg = head.get("configuration")
                if cfg:
                    cfg = {k: v for k, v in cfg.items() if v is not None}
                    cfg.pop("seed", None)
                    print(f"  configuration copied from real episode "
                          f"{os.path.basename(path)}")
                    return cfg
            except Exception:                                       # noqa: BLE001
                pass
    print("  configuration: stock competition defaults")
    return dict(STOCK_CONFIG)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base", default=os.path.join(ROOT, "agents", "v2_tuned.py"),
                    help="agent whose PARAMS are the starting point")
    ap.add_argument("--days", type=int, default=2, help="daily datasets to mine")
    ap.add_argument("--per-day", type=int, default=3, help="top replays per day")
    ap.add_argument("--own", type=int, default=3, help="own replays to fetch")
    ap.add_argument("--max-mb", type=float, default=250.0, help="download budget")
    ap.add_argument("--max-replays", type=int, default=8, help="replays to parse")
    ap.add_argument("--learn-rate", type=float, default=0.5,
                    help="how far to move toward the ladder's numbers (0..1)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--improve-budget", type=float, default=20.0,
                    help="minutes per improve round")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse replays already in data/episodes")
    ap.add_argument("--offline", action="store_true",
                    help="no Kaggle calls at all")
    ap.add_argument("--stock-config", action="store_true",
                    help="use stock defaults rather than a real episode's config")
    ap.add_argument("--non-interactive", action="store_true",
                    help="run the pipeline and stop before the submit prompt")
    ap.add_argument("--message", default=None, help="submission message")
    args = ap.parse_args()

    args.timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.seeds = [args.seed0 + i for i in range(args.seeds)]
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    hr(f"Kaggriculture pipeline   run {args.timestamp}")
    print(f"  base agent : {os.path.relpath(args.base, ROOT)}")
    print(f"  project    : {ROOT}")

    top_paths, mine_paths = stage_fetch(args)
    args.config = resolve_config(args, top_paths)

    top_agg, _mine_agg, _ours, report = stage_analyze(
        args, top_paths, mine_paths, args.base)

    agent = stage_refine(args, top_agg, args.base)

    opponents = [args.base, os.path.join(ROOT, "agents", "v1_heuristic.py"), "starter"]
    opponents = [o for o in opponents
                 if o in BUILTINS or (os.path.exists(o) and
                                      os.path.abspath(o) != os.path.abspath(agent))]
    stats = stage_evaluate(args, agent, opponents)

    rounds = 0
    while True:
        hr("RESULTS")
        print(f"  candidate : {os.path.relpath(agent, ROOT)}")
        print(f"  analysis  : {os.path.relpath(report, ROOT)}")
        print(f"  improve rounds so far: {rounds}\n")
        print_stats(stats)

        if args.non_interactive:
            print("\n  --non-interactive: stopping before submission.")
            print(f"  To submit: python -m kaggriculture.pipeline.pipeline --offline --skip-fetch "
                  f"# then choose [s]\n"
                  f"  or: kaggle competitions submit kaggriculture -f build/main.py "
                  f"-m \"...\"")
            return

        print("\n  [i] improve further   [r] re-evaluate   "
              "[s] submit to Kaggle   [q] quit")
        choice = (ask("  choice: ", "q").lower() or "q")[0]
        if choice == "i":
            agent = stage_improve(args, agent, rounds)
            rounds += 1
            stats = stage_evaluate(args, agent, opponents,
                                   label=f"re-evaluate after round {rounds}")
        elif choice == "r":
            stats = stage_evaluate(args, agent, opponents, label="re-evaluate")
        elif choice == "s":
            if stage_submit(args, agent, stats):
                return
        else:
            print("\n  stopped. The candidate agent is kept at "
                  f"{os.path.relpath(agent, ROOT)}")
            return


if __name__ == "__main__":
    main()
