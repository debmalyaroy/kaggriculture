"""RL task queue: one list of every RL task, run one by one or in parallel lanes, with a status
view that reports what is ACTUALLY happening (live pids, exit codes, log tails, free RAM).

    python ops/queue.py status            # the real state of every task and lane
    python ops/queue.py run               # the runner: starts ready jobs, records exits (loops)
    python ops/queue.py mark ID running|done|failed|blocked [--note TEXT]   # dev (hand) tasks
    python ops/queue.py retry ID          # a failed job back to pending

Tasks live in ops/tasks.json (spec, edited by hand) and their state in ops/state.json (written
only by this tool, atomically). Two kinds:
  job  -- a command the runner launches when every dependency is done and its lane has a free
          slot and free RAM >= min_free_gb. stdout+stderr go to data/ops/logs/<id>.log.
  dev  -- code work done in the Claude session; the session marks it. It never auto-starts, but
          jobs depending on it wait for it.
A job's process is started detached, so it survives a runner restart; the runner re-attaches
by pid. A job whose pid is gone without a recorded exit is reported as `lost`, never as done.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import krlenv  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(RL, "ops", "tasks.json")
STATE = os.path.join(RL, "ops", "state.json")
LOGS = os.path.join(RL, "data", "ops", "logs")
RUNNER_PID = os.path.join(RL, "ops", "runner.pid")
POLL_S = 20
BASH = shutil.which("bash") or r"C:\Program Files\Git\usr\bin\bash.exe"
TERMINAL = ("done", "failed", "skipped")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_spec():
    return json.load(open(SPEC, encoding="utf-8"))


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"tasks": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATE)


def free_gb():
    return psutil.virtual_memory().available / 2**30


def alive(pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, TypeError):
        return False


def tree_rss_gb(pid):
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(x.memory_info().rss for x in procs if x.is_running()) / 2**30
    except psutil.Error:
        return 0.0


def log_tail(tid, n=1):
    path = os.path.join(LOGS, f"{tid}.log")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            lines = [l for l in fh.read().decode("utf-8", "replace").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def deps_done(task, st):
    return all(st["tasks"].get(d, {}).get("state") == "done" for d in task.get("deps", []))


GATE_SHARE = 16  # = python/learn/ppo.py GATE_SHARE


def ncpu():
    return max(2, (os.cpu_count() or 4) - 2)


def launch(task, running=None):
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f"{task['id']}.log"), "ab")
    log.write(f"\n=== {now()} START {task['id']}: {task['cmd']}\n".encode())
    log.flush()
    exit_file = os.path.join(LOGS, f"{task['id']}.exit")
    if os.path.exists(exit_file):
        os.remove(exit_file)
    # wrapper records the exit code even if the runner is not alive when the job ends
    cmd = task["cmd"].replace("{PY}", f'"{krlenv.PY}"').replace("{BIN}", krlenv.BIN_DIR.replace(os.sep, "/")).replace("{EXE}", krlenv.EXE)
    # {NCPU_SHARE}: every core, or GATE_SHARE of them while PPO holds the rest (PPO yields the same
    # number back, python/learn/ppo.py share)
    share = GATE_SHARE if (running or {}).get("ppo") else ncpu()
    cmd = cmd.replace("{NCPU_SHARE}", str(share)).replace("{NCPU}", str(ncpu())).replace("{PPO_GAMES}", os.environ.get("KRL_PPO_GAMES", "1024"))
    wrapped = f'{cmd}; echo $? > "{exit_file.replace(os.sep, "/")}"'
    flags = 0
    if os.name == "nt":
        # NOT DETACHED_PROCESS: MSYS bash dies silently without a console. And a bare "bash"
        # resolves to the WSL stub in System32 when CreateProcess searches, so use the full path.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    p = subprocess.Popen([BASH, "-c", wrapped], cwd=task.get("cwd", RL), stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=(os.name != "nt"),
                         env=dict(os.environ, **krlenv.env(), KRL_THREADS=str(share)))
    return p.pid


def reconcile(st):
    """Record exits of jobs that finished (from their .exit file) and flag lost ones."""
    changed = False
    for tid, t in st["tasks"].items():
        if t.get("state") != "running" or t.get("kind") != "job":
            continue
        exit_file = os.path.join(LOGS, f"{tid}.exit")
        if os.path.exists(exit_file):
            try:
                rc = int(open(exit_file).read().strip() or "1")
            except ValueError:
                rc = 1
            t.update(state="done" if rc == 0 else "failed", rc=rc, ended=now())
            if rc == 0:
                t.pop("retries", None)
            changed = True
        elif not alive(t.get("pid")):
            t.update(state="lost", ended=now(), note="process gone without an exit record (killed/reaped)")
            changed = True
    return changed


def _ts(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


RETRY_S = int(os.environ.get("KRL_RETRY_S", "300"))


def rearm(spec, st):
    """Recurring work. `every_h`: a finished task goes back to pending that many hours after it
    last STARTED (round += 1). `follow`: a task re-runs whenever a task it follows has completed
    a newer round than the one it last consumed -- so a delta pull re-runs features, then BC,
    each exactly once per new delta, and a crash mid-chain resumes at the failed link."""
    changed = False
    t_now = dt.datetime.now(dt.timezone.utc)
    for task in spec["tasks"]:
        rec = st["tasks"].get(task["id"])
        if not rec:
            continue
        # hands-off: a failed or lost job is retried (it resumes from what is on disk) up to `retries`
        # times (default 3), RETRY_S after it ended; after that it stays failed and blocks its chain
        if task["kind"] == "job" and rec.get("state") in ("failed", "lost") and rec.get("retries", 0) < task.get("retries", 3):
            t1 = _ts(rec.get("ended"))
            if t1 is None or (t_now - t1).total_seconds() >= RETRY_S:
                rec.update(state="pending", retries=rec.get("retries", 0) + 1, rc=None,
                           note=f"auto-retry {rec.get('retries', 0) + 1} after {rec.get('state')}")
                changed = True
                continue
        if task.get("every_h") and rec.get("state") in ("done", "failed", "lost"):
            t0 = _ts(rec.get("started"))
            if t0 and (t_now - t0).total_seconds() >= task["every_h"] * 3600:
                rec.update(state="pending", round=rec.get("round", 0) + 1, rc=None)
                changed = True
        for f in task.get("follow", []):
            fr = st["tasks"].get(f, {})
            if fr.get("state") == "done" and fr.get("round", 0) > rec.get("seen", {}).get(f, -1) \
                    and rec.get("state") in ("done",):
                rec.update(state="pending", round=rec.get("round", 0) + 1, rc=None)
                changed = True
    return changed


def step(spec, st):
    """One runner pass: reconcile, re-arm recurring work, then start whatever is ready."""
    changed = reconcile(st)
    changed = rearm(spec, st) or changed
    lanes = spec.get("lanes", {})
    running = {}
    for t in st["tasks"].values():
        if t.get("state") == "running":
            running[t.get("lane")] = running.get(t.get("lane"), 0) + 1
    for task in spec["tasks"]:
        tid = task["id"]
        rec = st["tasks"].setdefault(tid, {"state": "pending", "kind": task["kind"], "lane": task.get("lane")})
        rec["kind"], rec["lane"] = task["kind"], task.get("lane")
        if task["kind"] != "job" or rec["state"] != "pending":
            continue
        if not deps_done(task, st):
            continue
        lane = task.get("lane", "cpu")
        if running.get(lane, 0) >= lanes.get(lane, 1):
            rec["wait"] = f"lane {lane} busy"
            continue
        need = task.get("min_free_gb", 2.0)
        if free_gb() < need:
            rec["wait"] = f"free RAM {free_gb():.1f} GB < {need} GB"
            changed = True
            continue
        rec.pop("wait", None)
        rec.pop("note", None)
        # record which upstream rounds this run consumes (for `follow` re-arming)
        rec["seen"] = {f: st["tasks"].get(f, {}).get("round", 0) for f in task.get("follow", [])}
        rec.update(state="running", pid=launch(task, running), started=now(), ended=None, rc=None)
        running[lane] = running.get(lane, 0) + 1
        changed = True
    write_lanes(running)
    return changed


def write_lanes(running):
    """data/ops/lanes.json: running jobs per lane, read by ppo.py to yield cores to gates."""
    p = os.path.join(RL, "data", "ops", "lanes.json")
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(running, open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def fmt_dur(a, b=None):
    try:
        t0 = dt.datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        t1 = dt.datetime.strptime(b, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if b else \
            dt.datetime.now(dt.timezone.utc)
        s = int((t1 - t0).total_seconds())
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    except (TypeError, ValueError):
        return ""


def status(as_json=False):
    spec, st = load_spec(), load_state()
    reconcile(st)
    runner = None
    try:
        rp = int(open(RUNNER_PID).read())
        runner = rp if alive(rp) else None
    except (OSError, ValueError):
        pass
    rows = []
    for task in spec["tasks"]:
        rec = st["tasks"].get(task["id"], {"state": "pending"})
        state = rec.get("state", "pending")
        if state == "pending" and task["kind"] == "job":
            blockers = [d for d in task.get("deps", []) if st["tasks"].get(d, {}).get("state") != "done"]
            state = f"waiting on {','.join(blockers)}" if blockers else ("ready: " + rec["wait"] if rec.get("wait") else "ready")
        elif state == "pending":
            state = "todo"
        row = {"id": task["id"], "kind": task["kind"], "lane": task.get("lane", "-"), "board": task.get("board", ""),
               "title": task["title"], "state": state, "slot": task.get("slot", "")}
        if rec.get("started"):
            row["time"] = fmt_dur(rec["started"], rec.get("ended"))
        if rec.get("round"):
            row["round"] = rec["round"]
        if rec.get("state") == "running" and task["kind"] == "job":
            row["pid_alive"] = alive(rec.get("pid"))
            row["rss_gb"] = round(tree_rss_gb(rec.get("pid")), 2)
        if task["kind"] == "job" and rec.get("state") in ("running", "failed", "lost", "done"):
            row["log"] = (log_tail(task["id"]) or [""])[-1][:140]
        if rec.get("note"):
            row["note"] = rec["note"][:140]
        if rec.get("rc") is not None:
            row["rc"] = rec["rc"]
        rows.append(row)
    head = {"at": now(), "runner": f"alive (pid {runner})" if runner else "NOT RUNNING",
            "free_ram_gb": round(free_gb(), 1), "cpu_pct": psutil.cpu_percent(interval=0.5)}
    if as_json:
        print(json.dumps({"head": head, "tasks": rows}, indent=1))
        return
    print(f"RL QUEUE  {head['at']}  runner {head['runner']}  free RAM {head['free_ram_gb']} GB  CPU {head['cpu_pct']}%")
    for r in rows:
        extra = []
        if "time" in r:
            extra.append(r["time"])
        if "pid_alive" in r:
            extra.append(("pid alive" if r["pid_alive"] else "PID DEAD") + f", {r['rss_gb']} GB")
        if "rc" in r:
            extra.append(f"rc {r['rc']}")
        if "round" in r:
            extra.append(f"round {r['round']}")
        print(f"{r['id']:<5} {r['kind']:<3} {r['lane']:<5} {r['state']:<26} {r['title'][:58]:<58} "
              f"{' | '.join(extra)}")
        if r.get("log"):
            print(f"{'':>16}log: {r['log']}")
        if r.get("note"):
            print(f"{'':>16}note: {r['note']}")


def run():
    open(RUNNER_PID, "w").write(str(os.getpid()))
    print(f"{now()} runner up (pid {os.getpid()})", flush=True)
    while True:
        spec, st = load_spec(), load_state()
        before = {k: v.get("state") for k, v in st["tasks"].items()}
        if step(spec, st):
            save_state(st)
        for k, v in st["tasks"].items():
            if before.get(k) != v.get("state"):
                print(f"{now()} {k} -> {v.get('state')}" + (f" rc={v.get('rc')}" if v.get("rc") is not None else ""),
                      flush=True)
        time.sleep(POLL_S)


def mark(tid, state, note):
    spec, st = load_spec(), load_state()
    task = next((t for t in spec["tasks"] if t["id"] == tid), None)
    if task is None:
        sys.exit(f"unknown task {tid}")
    rec = st["tasks"].setdefault(tid, {"kind": task["kind"], "lane": task.get("lane")})
    if state == "running" and not rec.get("started"):
        rec["started"] = now()
    if state in TERMINAL + ("blocked",):
        rec["ended"] = now()
    rec["state"] = "pending" if state == "retry" else state
    if state == "retry":
        rec.pop("retries", None)  # a manual retry restores the automatic-retry budget
    if note:
        rec["note"] = note
    save_state(st)
    print(f"{tid} -> {rec['state']}")


def dryrun(release_holds=False):
    """Verify the spec without running anything: every job's command expands and names a script /
    binary that exists, then the runner is simulated (the real `step`, lanes respected, every job
    succeeding one pass after launch, RAM ignored) until nothing changes. Prints the launch order
    and exits 1 if any job is never reached."""
    global launch, free_gb, write_lanes
    spec, bad = load_spec(), []
    for t in spec["tasks"]:
        if t["kind"] != "job":
            continue
        cmd = t["cmd"].replace("{PY}", "PY").replace("{BIN}", krlenv.BIN_DIR.replace(os.sep, "/")).replace("{EXE}", krlenv.EXE)
        for tok in cmd.replace("&&", " ").split():
            if tok.endswith((".py", ".sh")) or tok.startswith(krlenv.BIN_DIR.replace(os.sep, "/")):
                if not os.path.exists(tok if os.path.isabs(tok) else os.path.join(RL, tok)):
                    bad.append(f"{t['id']}: missing {tok}")
    st = {"tasks": {}}
    if release_holds:  # simulate with every dev task (the build holds) marked done
        for t in spec["tasks"]:
            if t["kind"] == "dev":
                st["tasks"][t["id"]] = {"state": "done", "kind": "dev", "lane": None}
    real_launch, real_free, real_lanes = launch, free_gb, write_lanes
    order, pend = [], []
    launch = lambda task, running=None: (order.append(task["id"]), pend.append(task["id"]), 0)[2]  # noqa: E731
    free_gb = lambda: 1e9  # noqa: E731
    write_lanes = lambda running: None  # noqa: E731
    try:
        for p in range(200):
            for tid in pend:  # jobs launched last pass finish now
                st["tasks"][tid].update(state="done", rc=0, ended=now())
            pend.clear()
            if not step(spec, st) and not pend:
                break
            if pend:
                print(f"pass {p:2d}: launch {', '.join(pend)}")
    finally:
        launch, free_gb, write_lanes = real_launch, real_free, real_lanes
    never = [t["id"] for t in spec["tasks"] if t["kind"] == "job" and t["id"] not in order]
    for b in bad:
        print("MISSING", b)
    print(f"dry run: {len(order)} launches, never reached: {never or 'none'}; missing files: {len(bad)}")
    sys.exit(1 if never or bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "run", "mark", "retry", "dryrun"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("state", nargs="?")
    ap.add_argument("--note")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--release-holds", action="store_true", help="dryrun: treat dev tasks (holds) as done")
    a = ap.parse_args()
    if a.cmd == "status":
        status(a.json)
    elif a.cmd == "run":
        run()
    elif a.cmd == "mark":
        mark(a.id, a.state, a.note)
    elif a.cmd == "retry":
        mark(a.id, "retry", a.note)
    elif a.cmd == "dryrun":
        dryrun(a.release_holds)


if __name__ == "__main__":
    main()
