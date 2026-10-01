#!/usr/bin/env python3
"""Download Kaggriculture episode data into data/.

RUN THIS FROM YOUR OWN WINDOWS TERMINAL. It needs network and your Kaggle
login, neither of which an assistant session can reach.

    cd D:\\codebase\\kaggriculture
    python scripts/download_data.py                    # sensible defaults
    python scripts/download_data.py --per-day 80 --own 60
    python scripts/download_data.py --check            # just verify the setup

It checks prerequisites first and tells you exactly what to fix, then streams
episodes: fetch one (~27 MB), reduce it to ~60 numbers, delete it, next. Disk
never grows. Re-running only fetches what is new.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in scripts/)
sys.path.insert(0, os.path.join(ROOT, "src"))


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg, fix):
    print(f"  [FAIL] {msg}")
    for line in fix.splitlines():
        print(f"         {line}")


def preflight():
    print("Checking prerequisites\n")
    good = True

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import episodes as ep
    try:
        cmd = ep.kaggle_cmd()
        shown = cmd[0] if len(cmd) == 1 else " ".join(cmd)
        ok(f"Kaggle CLI usable: {shown}")
    except ep.KaggleError as exc:
        bad("the Kaggle CLI is not usable", str(exc))
        return False

    try:
        r = subprocess.run(cmd + ["competitions", "list", "-s", "kaggriculture"],
                           capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            err = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
            bad(f"kaggle CLI is not authenticated ({err[0][:90]})",
                "kaggle auth login\n"
                "  or put your token in %USERPROFILE%\\.kaggle\\access_token\n"
                "  get one at https://www.kaggle.com/settings/api")
            good = False
        else:
            ok("kaggle CLI is authenticated")
    except Exception as exc:                                       # noqa: BLE001
        bad(f"could not run the kaggle CLI: {exc}", "pip install kaggle")
        good = False

    if good:
        try:
            r = subprocess.run(cmd + ["competitions", "list", "--group", "entered"],
                               capture_output=True, text=True, timeout=90)
            if "kaggriculture" in (r.stdout or "").lower():
                ok("you have joined the competition")
            else:
                bad("you have not joined the competition",
                    "open https://www.kaggle.com/competitions/kaggriculture\n"
                    "and click 'Join Competition' (the CLI cannot accept rules)")
                good = False
        except Exception:                                          # noqa: BLE001
            print("  [warn] could not verify competition entry; continuing")

    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 2:
        bad(f"only {free:.1f} GB free", "free up ~2 GB; episodes stream but need headroom")
        good = False
    else:
        ok(f"{free:.0f} GB free")
    return good


def diagnose_leaderboard(top_n):
    """Show exactly what the leaderboard call returns and what we parse from it.

    Run this when the download reports 0 team names resolved -- it prints the
    first lines of raw CLI output for each invocation tried, which is the one
    thing needed to fix the parser.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import episodes as ep                                          # noqa: E402

    print(f"\nLeaderboard diagnosis (want top {top_n})\n" + "-" * 66)
    for cmd in ep._leaderboard_invocations():
        label = " ".join(cmd[1:])
        print(f"\n$ kaggle {label}")
        try:
            raw = ep._run(cmd, timeout=120)
        except ep.KaggleError as exc:
            print(f"  FAILED: {str(exc).splitlines()[0]}")
            continue
        lines = (raw or "").splitlines()
        if not lines:
            print("  (no output)")
            continue
        print(f"  {len(lines)} line(s); first 5:")
        for ln in lines[:5]:
            print(f"    | {ln[:150]}")
        names, header = ep._names_from_csv(raw, top_n)
        print(f"  csv header  : {header or 'not CSV'}")
        print(f"  csv names   : {names[:5]}{' ...' if len(names) > 5 else ''}")
        if not names:
            print(f"  table names : {ep._names_from_table(raw, top_n)[:5]}")

    names = ep.leaderboard_top(top_n)
    print("\n" + "-" * 66)
    print(f"resolved {len(names)} name(s): {names}")
    if not names:
        print("\nNo names resolved. This is NOT fatal -- the daily dataset is "
              "already ranked by rating,\nso `python scripts/download_data.py --any-top` "
              "gets you the same top-player games without the filter.")
    return 0


def diagnose_submissions():
    """Show how your own submissions and episodes are being resolved.

    Run this when the download reports "no submissions found". kaggle 1.8.3
    ships a broken `competitions submissions` CLI -- it passes page_number= to
    an API method of its own that does not accept it -- so the project falls
    back to calling the API in-process. This prints which path actually worked.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import episodes as ep                                          # noqa: E402

    print("\nOwn-submission diagnosis\n" + "-" * 66)
    print("\n1. via the CLI")
    try:
        out = ep._run(["kaggle", "competitions", "submissions",
                       "kaggriculture", "--csv"])
        lines = (out or "").splitlines()
        print(f"   ok, {len(lines)} line(s); first 3:")
        for ln in lines[:3]:
            print(f"     | {ln[:150]}")
    except ep.KaggleError as exc:
        parts = str(exc).splitlines()
        print(f"   FAILED: {parts[0]}")
        for ln in parts[1:6]:
            print(f"     | {ln[:150]}")

    print("\n2. via the in-process API")
    api = ep.kaggle_api()
    print(f"   authenticated: {api is not None}")

    print("\n3. resolved rows")
    rows = ep._own_submission_rows(verbose=True)
    print(f"   {len(rows)} submission row(s)")
    if not rows:
        print("\n   Nothing resolved. If you HAVE submitted to this competition,")
        print("   this is the 1.8.3 CLI bug and the API fallback also failed.")
        print('   Workaround:  python -m pip install "kaggle==1.7.4.5"')
        return 0

    print(f"   keys: {sorted(rows[0])[:14]}")
    ident = None
    for r in rows[:5]:
        got = next((str(r[k]) for k in ("ref", "id", "submissionId")
                    if r.get(k) is not None and str(r[k]).isdigit()), None)
        ident = ident or got
        print(f"     submission {got}  {r.get('fileName', '')} "
              f"{r.get('date', r.get('submissionDate', ''))}")
    if ident:
        print(f"\n4. episodes for submission {ident}")
        ids = ep._own_episode_ids(ident, verbose=True)
        print(f"   {len(ids)} episode id(s): {ids[:8]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=3, help="how many daily datasets")
    ap.add_argument("--per-day", type=int, default=40,
                    help="top episodes per day (x --days). 40x3=120 by default; "
                         "raise it freely, replays stream and are deleted after "
                         "featurising so disk does not grow")
    ap.add_argument("--own", type=int, default=0,
                    help="your own episodes (0 = ALL of them)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="restrict ladder episodes to the top-N leaderboard teams")
    ap.add_argument("--any-top", action="store_true",
                    help="do not filter by team: take the highest-rated episodes")
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel replay downloads (default 6; each replay is "
                         "~27 MB and every one costs a fresh kaggle-CLI startup, "
                         "so this is the single biggest speed lever)")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--verbose", action="store_true",
                    help="echo every Kaggle CLI call and how long it took")
    ap.add_argument("--leaderboard", action="store_true",
                    help="diagnose top-N resolution only: show the raw CLI "
                         "output and the names parsed out of it, then exit")
    ap.add_argument("--submissions", action="store_true",
                    help="diagnose your-own-games only: list your submissions "
                         "and their episodes, then exit")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the EDA report afterwards")
    ap.add_argument("--notebooks", type=int, nargs="?", const=25, default=25,
                    help="also harvest the top-N public notebooks and decode "
                         "the agents embedded in them (0 to skip). This is "
                         "where the leaders publish their actual submissions")
    ap.add_argument("--notebooks-only", action="store_true",
                    help="harvest notebooks and exit -- no replay download")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import progress as pr                                          # noqa: E402
    if args.verbose:
        pr.set_verbose(True)
    pr.reset()

    print("=" * 66)
    print("Kaggriculture — episode download")
    print("=" * 66)
    if not preflight():
        print("\nFix the items above and re-run.")
        return 1
    if args.check:
        print("\nAll good. Run without --check to download.")
        return 0

    if args.leaderboard:
        return diagnose_leaderboard(args.top_n)

    if args.submissions:
        return diagnose_submissions()

    if args.notebooks_only:
        import notebooks as NB                                      # noqa: E402
        NB.harvest(top=args.notebooks or 25)
        return 0

    print("\nDownloading (streaming: fetch -> featurise -> delete)\n")
    import scheduled_fetch                                          # noqa: E402
    argv = ["scheduled_fetch", "--per-day", str(args.per_day),
            "--own", str(args.own or 10 ** 6), "--days", str(args.days),
            "--jobs", str(args.jobs)]
    if not args.any_top:
        argv += ["--top-n", str(args.top_n)]
    sys.argv = argv
    rc = scheduled_fetch.main()

    if args.notebooks:
        print("\nHarvesting public notebooks (the leaders publish their agents)\n")
        try:
            import notebooks as NB                                  # noqa: E402
            NB.harvest(top=args.notebooks)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  ! notebook harvest failed: {exc}")

    csv_path = os.path.join(ROOT, "data", "episodes.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"\ndata/episodes.csv now holds {n:,} player-rows "
              f"({n//2:,} episodes)")
    if args.report:
        print("\nRebuilding the EDA report ...")
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "eda_report.py")])

    print("\nNext:")
    print("  python tools\\eda_report.py                 # docs\\eda-report.html")
    print("  python tools\\learn_params.py --top 25      # what predicts winning")
    print("  python tools\\train_supervised.py           # train the value model")
    print("  powershell -File tools\\install_scheduler.ps1   # keep it fresh hourly")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
