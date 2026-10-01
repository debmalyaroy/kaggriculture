"""Daily DELTA pipeline for the RL corpus: new Kaggle daily datasets + GM re-extracts -> sanitized corpus.

    python python/delta.py run              # plan -> push -> wait -> download -> file -> prune -> index -> report
    python python/delta.py plan             # show what a run would extract (no side effects)
    python python/delta.py status           # state + both kernels' Kaggle status
    python python/delta.py bootstrap        # ONE-TIME: record the current corpus as the baseline
    python python/delta.py manifest         # laptop: write data/slim/s1/corpus_manifest.tsv (what the corpus IS)
    python python/delta.py pull-corpus      # AWS box: rebuild data/slim from our notebooks' outputs by that manifest

Everything is resumable: data/kaggle_out/delta_state.json records each kernel's phase
(planned -> pushed -> downloaded -> filed). Re-running `run` continues where it stopped.
Full description: docs/delta.md.

What "sanitize" means here:
  * engine: only 1.32.7 episodes (the extractor's --engine default; the ladder cut over 2026-08-15 01:38 UTC)
  * dedup: python/corpus_index.py keeps one row per episode_id (newest run wins)
  * prune: a day re-extracted (GM from a newer version, or a daily re-run) has its older parts deleted
  * rerank: the index recomputes per-seat rating_pre/post, band and daily leaderboard rank
  * validation self-play episodes stay in the corpus but are flagged (episode_type)
"""
import argparse
import csv
import datetime as dt
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kjob  # noqa: E402  shared private-notebook machinery (push / status / fetch / download)
OUT_ROOT = os.path.join(RL, "data", "kaggle_out")
SLIM = os.path.join(RL, "data", "slim", "s1")
STATE = os.path.join(OUT_ROOT, "delta_state.json")
REPORT = os.path.join(OUT_ROOT, "delta_report.md")
KERNELS = os.path.join(RL, "kaggle", "kernels")
TEMPLATE = os.path.join(RL, "kaggle", "delta", "notebook_template.py")
OWNER = "debmalya84"
BIN_DS = f"{OWNER}/kaggriculture-rl-bin"
GM_DS = "georgymamarin/kaggriculture-episodes"
FIRST_DATE = "2026-08-15"          # engine 1.32.7 cut-over day (the extractor drops earlier episodes)
GM_MIN_GAIN = 0.02                 # re-extract a GM day when its replay coverage rose by >= 2 points
DAILY_SETTLE_MIN = 30              # ignore a daily dataset updated less than 30 min ago (still uploading)
POLL_SEC = 120
WAIT_HOURS = 8
KINDS = ("daily", "gm")


log = kjob.log
kaggle = kjob.kaggle


def kname(kind):
    return f"krl-delta-{kind}"


# ---------------------------------------------------------------- state
def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8-sig"))
    return None


def save_state(st):
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1, sort_keys=True)
    os.replace(tmp, STATE)


# ---------------------------------------------------------------- sources
def gm_version():
    rc, out = kaggle(["datasets", "list", "--user", GM_DS.split("/")[0], "--csv"])
    for r in csv.DictReader(io.StringIO(out[out.find("ref,"):])):
        if r.get("ref") == GM_DS:
            return r["lastUpdated"]
    raise RuntimeError(f"GM dataset not listed: {out[:200]}")


def gm_coverage():
    """{date: replay_coverage} from the CURRENT GM version's daily_stats.csv."""
    d = os.path.join(OUT_ROOT, "delta")
    os.makedirs(d, exist_ok=True)
    rc, out = kaggle(["datasets", "download", GM_DS, "-f", "daily_stats.csv", "-p", d, "--force"])
    path = os.path.join(d, "daily_stats.csv")
    if not os.path.exists(path) and os.path.exists(path + ".zip"):
        import zipfile
        zipfile.ZipFile(path + ".zip").extractall(d)
    if not os.path.exists(path):
        raise RuntimeError(f"daily_stats.csv download failed: {out[:200]}")
    return {r["date"]: float(r["replay_coverage"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def daily_available():
    """{date: (ref, lastUpdated)} for every kaggle/kaggriculture-episodes-YYYY-MM-DD dataset with files."""
    found = {}
    for page in range(1, 20):
        rc, out = kaggle(["datasets", "list", "--user", "kaggle", "-s", "kaggriculture-episodes-2026",
                          "--sort-by", "updated", "--page", str(page), "--csv"])
        rows = [r for r in csv.DictReader(io.StringIO(out[out.find("ref,"):])) if r.get("ref")] if "ref," in out else []
        if not rows:
            break
        for r in rows:
            m = re.fullmatch(r"kaggle/kaggriculture-episodes-(\d{4}-\d{2}-\d{2})", r["ref"])
            if m:
                found[m.group(1)] = (r["ref"], r["lastUpdated"])
    return found


def has_files(ref):
    rc, out = kaggle(["datasets", "files", ref, "--page-size", "1"])
    return rc == 0 and ".json" in out


def local_dates(src):
    return sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(SLIM, f"source={src}", "date=*")))


# ---------------------------------------------------------------- plan
def plan(st):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = {}
    # daily: every settled dataset from FIRST_DATE on that we have not filed
    avail = daily_available()
    todo = []
    for date, (ref, upd) in sorted(avail.items()):
        if date < FIRST_DATE or date in st["daily_done"]:
            continue
        age_min = (now - dt.datetime.fromisoformat(upd[:19])).total_seconds() / 60
        if age_min < DAILY_SETTLE_MIN or not has_files(ref):
            continue
        todo.append((date, ref))
    if todo:
        out["daily"] = {"dates": [d for d, _ in todo], "datasets": [r for _, r in todo], "extra": []}
    # gm: only when the GM dataset has a newer version; redo days whose coverage improved (or are new)
    ver = gm_version()
    if ver[:19] > st["gm_version"][:19]:
        cov = gm_coverage()
        redo = sorted(d for d, c in cov.items()
                      if d >= FIRST_DATE and c > 0 and c - st["gm_coverage"].get(d, 0.0) >= GM_MIN_GAIN)
        if redo:
            out["gm"] = {"dates": redo, "since": redo[0], "until": redo[-1], "gm_version": ver,
                         "coverage": {d: cov[d] for d in redo},
                         "extra": ["--since", redo[0], "--until", redo[-1]]}
        else:
            st["gm_version"] = ver       # new version, nothing worth re-extracting
    return out


# ---------------------------------------------------------------- kernels
def make_kernel(kind, p):
    code = open(TEMPLATE, encoding="utf-8").read()
    code = (code.replace("__KIND__", repr(kind)).replace("__EXTRA__", repr(p["extra"]))
            .replace("__PLAN__", repr({k: v for k, v in p.items() if k != "extra"})))
    sources = [BIN_DS] + (p["datasets"] if kind == "daily" else [GM_DS])
    return kjob.make_kernel(kname(kind), sources, code)


def kernel_status(kind):
    return kjob.status(kname(kind))


def push(kind, st):
    k = st["kernels"][kind]
    make_kernel(kind, k["plan"])
    k["pushed_utc"] = kjob.push(kname(kind))
    k["phase"] = "pushed"
    shutil.rmtree(os.path.join(OUT_ROOT, kname(kind)), ignore_errors=True)   # never mix versions
    log(f"PUSHED {kname(kind)} {k['plan'].get('dates')}")
    return True


# ---------------------------------------------------------------- download
def fetch(kind, pattern):
    kjob.fetch(kname(kind), os.path.join(OUT_ROOT, kname(kind)), pattern)


def download(kind, st):
    """run.json + manifest first (stale / failed outputs refused); then exactly the missing files."""
    k = st["kernels"][kind]
    kjob.download(kname(kind), os.path.join(OUT_ROOT, kname(kind)), "slim/manifest.txt", k["pushed_utc"])
    k["phase"] = "downloaded"
    return True


# ---------------------------------------------------------------- file + prune
def file_run(kind, st):
    k = st["kernels"][kind]
    d = os.path.join(OUT_ROOT, kname(kind))
    run = json.load(open(os.path.join(d, "run.json")))
    src = "gm" if kind == "gm" else "official"
    sha7 = run["binary_sha"].replace("sha256", "").strip()[:7]
    runid = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ") + f"-{kname(kind)}-{sha7}"
    n, dates = 0, set()
    # resume an interrupted filing (28 Sep: killed mid-move): KRL_DELTA_RUNID = the id the first half was filed
    # under, so both halves share it and the prune keeps them; the dates/count include what was already filed
    if os.environ.get("KRL_DELTA_RUNID"):
        runid = os.environ["KRL_DELTA_RUNID"]
        for f in glob.glob(os.path.join(SLIM, f"source={src}", "date=*", f"part-{runid}-*.parquet")):
            n += 1
            dates.add(os.path.basename(os.path.dirname(f))[5:])
    for f in glob.glob(os.path.join(d, "slim", "slim", "date=*", "*.parquet")):
        date_dir = os.path.basename(os.path.dirname(f))
        dst = os.path.join(SLIM, f"source={src}", date_dir)
        os.makedirs(dst, exist_ok=True)
        tag = re.sub(r"^part-", "", os.path.splitext(os.path.basename(f))[0])
        shutil.move(f, os.path.join(dst, f"part-{runid}-{tag}.parquet"))
        n += 1
        dates.add(date_dir[5:])
    ldst = os.path.join(SLIM, "ledger")
    os.makedirs(ldst, exist_ok=True)
    for f in glob.glob(os.path.join(d, "slim", "ledger", "*.parquet")):
        tag = re.sub(r"^part-", "", os.path.splitext(os.path.basename(f))[0])
        shutil.move(f, os.path.join(ldst, f"ledger-{runid}-{tag}.parquet"))
    runs = os.path.join(SLIM, "runs")
    os.makedirs(runs, exist_ok=True)
    run.update(runid=runid, kernel=kname(kind), source=src, filed_dates=sorted(dates))
    json.dump(run, open(os.path.join(runs, f"{runid}.json"), "w"), indent=1)
    for lg in glob.glob(os.path.join(d, "*.log")):
        shutil.copy(lg, os.path.join(runs, f"{runid}.log"))
    k.update(phase="filed", runid=runid, filed_files=n, filed_dates=sorted(dates))
    log(f"FILED {kname(kind)} {n} parquet files as {runid} ({len(dates)} dates)")
    return runid, sorted(dates)


def prune(src, runid, dates):
    """Delete older parts of `src` for the days this run re-extracted. A GM re-extract comes from a
    newer, fuller GM version; a daily re-run re-reads the same dataset. Either way the new run's
    files for that day supersede every older part (the index would pick them anyway)."""
    removed = 0
    for date in dates:
        for f in glob.glob(os.path.join(SLIM, f"source={src}", f"date={date}", "part-*.parquet")):
            if f"-{runid}-" not in os.path.basename(f):
                os.remove(f)
                removed += 1
    return removed


# ---------------------------------------------------------------- commands
def cmd_bootstrap(a):
    if load_state() and not a.force:
        sys.exit("state exists; use --force to overwrite")
    ver, cov = gm_version(), gm_coverage()
    have = set(local_dates("gm"))
    st = {"gm_version": ver, "gm_coverage": {d: c for d, c in cov.items() if d in have},
          "daily_done": local_dates("official"), "kernels": {}, "history": []}
    save_state(st)
    log(f"BOOTSTRAP gm_version={ver}; gm days {len(st['gm_coverage'])}; daily days {len(st['daily_done'])}")


def cmd_plan(a):
    st = load_state() or sys.exit("no state: run `python python/delta.py bootstrap` once")
    p = plan(dict(st))
    print(json.dumps(p, indent=1) if p else "nothing new")


def cmd_status(a):
    st = load_state() or sys.exit("no state")
    print(f"gm_version {st['gm_version']}; daily_done {st['daily_done'][0]}..{st['daily_done'][-1]} ({len(st['daily_done'])})")
    for kind in KINDS:
        k = st["kernels"].get(kind)
        print(f"{kname(kind)}: kaggle={kernel_status(kind)} local={k and k.get('phase')} {k and k.get('plan', {}).get('dates')}")


def cmd_run(a):
    st = load_state() or sys.exit("no state: run `python python/delta.py bootstrap` once")
    # 1. plan (only for kinds not already in flight)
    busy = {kind for kind, k in st["kernels"].items() if k.get("phase") not in (None, "done")}
    if len(busy) < len(KINDS):
        p = plan(st)
        for kind, kp in p.items():
            if kind not in busy:
                st["kernels"][kind] = {"phase": "planned", "plan": kp}
                log(f"PLANNED {kname(kind)} {kp['dates']}")
        save_state(st)
    active = [kind for kind in KINDS if st["kernels"].get(kind, {}).get("phase") not in (None, "done")]
    if not active:
        log("nothing new (no unfiled daily dataset; GM has no newer version with better coverage)")
        _report(st, [])
        save_state(st)
        return
    # 2. push
    for kind in active:
        if st["kernels"][kind]["phase"] == "planned":
            push(kind, st)
            save_state(st)
    # 3. wait -> download -> file -> prune
    deadline = time.time() + WAIT_HOURS * 3600
    done_now = []
    while active and time.time() < deadline:
        for kind in list(active):
            k = st["kernels"][kind]
            if k["phase"] == "pushed":
                s = kernel_status(kind)
                if s in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                    k["phase"] = "failed"
                    log(f"FAILED {kname(kind)} status={s}")
                    active.remove(kind)
                    continue
                if s != "COMPLETE":
                    continue
                download(kind, st)
                save_state(st)
            if k["phase"] == "downloaded":
                runid, dates = file_run(kind, st)
                k["pruned"] = prune("gm" if kind == "gm" else "official", runid, dates)
                log(f"PRUNED {k['pruned']} superseded parts for {len(dates)} dates")
                if kind == "gm":
                    st["gm_version"] = k["plan"]["gm_version"]
                    st["gm_coverage"].update(k["plan"]["coverage"])
                else:
                    st["daily_done"] = sorted(set(st["daily_done"]) | set(k["plan"]["dates"]))
                shutil.rmtree(os.path.join(OUT_ROOT, kname(kind)), ignore_errors=True)
                save_state(st)
            if k["phase"] == "filed":
                k["phase"] = "done"
                st["history"].append({kk: k.get(kk) for kk in ("runid", "filed_files", "filed_dates", "pruned", "pushed_utc")}
                                     | {"kind": kind})
                done_now.append(kind)
                active.remove(kind)
                save_state(st)
        if active:
            time.sleep(POLL_SEC)
    if active:
        log(f"TIMEOUT waiting for {active}; re-run `python python/delta.py run` to resume")
    # 4. dedup + rerank
    if done_now:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import corpus_index
        corpus_index.main()
    _report(st, done_now)
    save_state(st)


def _report(st, done_now):
    lines = [f"## delta run {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')}"]
    if not done_now:
        lines.append("- nothing new filed")
    for kind in done_now:
        k = st["kernels"][kind]
        lines.append(f"- {kname(kind)}: filed {k.get('filed_files')} files as `{k.get('runid')}` for dates "
                     f"{k.get('filed_dates')}; pruned {k.get('pruned')} superseded parts")
    summ = os.path.join(SLIM, "index", "summary.txt")
    if done_now and os.path.exists(summ):
        lines.append("- index: " + open(summ, encoding="utf-8").readline().strip())
    lines.append(f"- state: gm_version {st['gm_version']}, daily through {st['daily_done'][-1] if st['daily_done'] else '-'}")
    with open(REPORT, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- corpus manifest + pull (AWS box)
MANIFEST = os.path.join(SLIM, "corpus_manifest.tsv")
PART_RE = re.compile(r"^part-(\d{8}T\d{4}Z)-(krl-[a-z0-9-]+?)-([0-9a-f]{7})-(.+)\.parquet$")


def cmd_manifest(a):
    """Every filed corpus part (source=official|gm): local path, producing notebook, output tag, bytes.
    A filed name is part-<RUNID>-<tag>.parquet with RUNID = <time>-<kernel>-<sha7>, and the notebook's
    output file is <tag>.parquet or part-<tag>.parquet, so the manifest is enough to fetch it again.
    League parts (source=league) are laptop-made and travel with aws/sync_up.sh instead."""
    rows = []
    for f in sorted(glob.glob(os.path.join(SLIM, "source=*", "date=*", "*.parquet"))):
        rel = os.path.relpath(f, SLIM).replace(os.sep, "/")
        if rel.startswith("source=league/"):
            continue
        m = PART_RE.match(os.path.basename(f))
        if not m:
            sys.exit(f"unrecognised corpus file name: {rel}")
        rows.append((rel, m.group(2), m.group(4), os.path.getsize(f)))
    with open(MANIFEST + ".part", "w", newline="\n") as fh:
        fh.write("path\tkernel\ttag\tbytes\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    os.replace(MANIFEST + ".part", MANIFEST)
    by = {}
    for r in rows:
        by[r[1]] = by.get(r[1], 0) + 1
    log(f"MANIFEST {len(rows)} files, {sum(r[3] for r in rows) / 1e9:.2f} GB, notebooks {by}")


def cmd_pull_corpus(a):
    """Rebuild the corpus from our private notebooks' outputs by the manifest: fetch every part we lack
    (40 per request, 8 rounds), move it to its filed path, and require the exact byte size."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(MANIFEST)][1:]
    tmp = os.path.join(OUT_ROOT, "pull")

    def missing():
        return [r for r in rows if not (os.path.exists(os.path.join(SLIM, r[0])) and os.path.getsize(os.path.join(SLIM, r[0])) == int(r[3]))]

    for rnd in range(8):
        todo = missing()
        if not todo:
            break
        log(f"pull-corpus round {rnd + 1}: {len(todo)} of {len(rows)} files missing")
        by = {}
        for r in todo:
            by.setdefault(r[1], []).append(r)
        # one unit = up to 400 missing files of one notebook in ONE pattern: every request walks the
        # notebook's whole output listing (~16 pages for the big GM runs) and Kaggle rate-limits those
        # page calls (429), so the number of walks is what matters, not the files per walk. Units run
        # in parallel, each in its own folder.
        units, k = [], 0
        for kern, rs in sorted(by.items()):
            for i in range(0, len(rs), 400):
                units.append((kern, k, rs[i:i + 400]))
                k += 1

        def pull_unit(u):
            kern, k, chunk = u
            dest = os.path.join(tmp, kern, f"c{k:04d}")
            # a tag repeats across date folders (and in ledger/), so anchor on the full output path
            # slim/slim/date=<D>/[part-]<tag>.parquet
            pats = [re.escape("slim/slim/" + r[0].split("/")[1] + "/") + "(part-)?" + re.escape(r[2]) + r"\.parquet" for r in chunk]
            kjob.fetch(kern, dest, "^(" + "|".join(pats) + ")$")
            n = 0
            for r in chunk:
                d = os.path.join(dest, "slim", "slim", r[0].split("/")[1])
                got = [p for p in (os.path.join(d, f"part-{r[2]}.parquet"), os.path.join(d, f"{r[2]}.parquet")) if os.path.exists(p)]
                if got and os.path.getsize(got[0]) == int(r[3]):
                    out = os.path.join(SLIM, r[0])
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    shutil.move(got[0], out)
                    n += 1
            shutil.rmtree(dest, ignore_errors=True)
            return kern, n, len(chunk)

        from concurrent.futures import ThreadPoolExecutor
        done = 0
        with ThreadPoolExecutor(a.jobs) as ex:
            for kern, n, want in ex.map(pull_unit, units):
                done += n
                log(f"pull-corpus {kern}: {n}/{want} filed ({done}/{len(todo)} this round)")
    todo = missing()
    shutil.rmtree(tmp, ignore_errors=True)
    if todo:
        sys.exit(f"pull-corpus: {len(todo)} files still missing, e.g. {[r[0] for r in todo[:3]]}")
    log(f"PULLED corpus: {len(rows)} files, {sum(int(r[3]) for r in rows) / 1e9:.2f} GB, all sizes exact")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--force", action="store_true")
    sub.add_parser("plan")
    sub.add_parser("status")
    sub.add_parser("run")
    sub.add_parser("manifest")
    pc = sub.add_parser("pull-corpus")
    pc.add_argument("--jobs", type=int, default=2, help="parallel Kaggle download requests (8 hits the 429 rate limit)")
    a = ap.parse_args()
    {"bootstrap": cmd_bootstrap, "plan": cmd_plan, "status": cmd_status, "run": cmd_run,
     "manifest": cmd_manifest, "pull-corpus": cmd_pull_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
