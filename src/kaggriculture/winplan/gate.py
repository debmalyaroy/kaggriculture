"""Per-game gate on the Rust serve engine, with paired comparison.

Every game is recorded (opponent, world, seed, seat, both banks) to a jsonl
file, so two candidates gated on the same roster and worlds pair up game by
game and `compare()` runs McNemar's exact test (win_metric.paired_test).

Seats: the game settles seat-symmetrically for these agents (verified on the
official engine 2026-09-24), so each (opponent, world) is played in seat 0
and a `symmetry_worlds` subset is ALSO played in seat 1 as an audit; any
asymmetry is reported rather than assumed away.

Fidelity: `loss_forensics.capture_game` applies serve_match's S3/S4/S5 fixes
and the engine keeps empty [] slots (2026-09-24 fix). A fresh agent module is
loaded for every game (no state carried between games).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time

from kaggriculture.winplan import paths as P


def _file_hash(path):
    """Content hash of an agent: the shim's target dir contents if it is a shim."""
    h = hashlib.sha256()
    src = open(path, "rb").read()
    h.update(src)
    text = src.decode("utf-8", "replace")
    if "_d = " in text and "spec_from_file_location" in text:   # a field shim
        try:
            target = text.split("_d = ", 1)[1].split("\n", 1)[0].strip().strip("'\"").replace("\\\\", "\\")
            for root, _, files in sorted(os.walk(target)):
                for f in sorted(files):
                    if f.endswith((".py", ".json")):
                        h.update(open(os.path.join(root, f), "rb").read())
        except Exception:                                            # noqa: BLE001
            pass
    return h.hexdigest()[:12]


def _play(task):
    """One game in a worker. Returns a result row (errors recorded, not raised)."""
    import contextlib
    import io
    from kaggriculture.bandit.gate import loss_forensics as LF
    ours_path, opp_name, opp_path, world, seed, seat = task
    row = {"opp": opp_name, "world": world, "seed": seed, "seat": seat}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ours = LF.load_pyagent(ours_path)
            _, _, us, them = LF.capture_game(ours, opp_path, seed, seat)
        row.update(us=us, them=them, gap=us - them)
    except Exception as exc:                                         # noqa: BLE001
        row.update(error=f"{type(exc).__name__}: {str(exc)[:160]}")
    row["secs"] = round(time.time() - t0, 1)
    return row


def roster(field_dir=None, only=None, exclude=None):
    """(name, path) for every shim in the field dir."""
    field_dir = field_dir or P.FIELD
    out = []
    for f in sorted(os.listdir(field_dir)):
        if not f.endswith(".py"):
            continue
        name = f[:-3]
        if only and name not in only:
            continue
        if exclude and name in exclude:
            continue
        out.append((name, os.path.join(field_dir, f)))
    return out


def run_gate(ours_path, label, opponents, worlds=24, symmetry_worlds=2,
             workers=None, progress=None):
    """Gate `ours_path` against `opponents` [(name, path)].

    Results stream to data/winplan/gates/<label>__<hash>.jsonl and a run is
    RESUMABLE: games already recorded for the same agent hash are skipped.
    `progress(done, total)` is called as games finish. Returns the summary.
    """
    from kaggriculture.bandit.gate import harness as H
    workers = workers or int(os.environ.get("NN_WORKERS", "8"))
    ours_hash = _file_hash(ours_path)
    out = os.path.join(P.GATES, f"{label}__{ours_hash}.jsonl")
    done = {}
    if os.path.exists(out):
        for ln in open(out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "error" not in r:
                done[(r["opp"], r["world"], r["seat"])] = r
    ws = H.world_seeds(worlds)
    tasks = []
    for name, path in opponents:
        for i, (wn, seed) in enumerate(ws):
            seats = (0, 1) if i < symmetry_worlds else (0,)
            for seat in seats:
                if (name, wn, seat) not in done:
                    tasks.append((ours_path, name, path, wn, seed, seat))
    total = len(tasks) + len(done)
    n_done = len(done)
    if progress:
        progress(n_done, total)
    # ProcessPoolExecutor, not mp.Pool: when a worker dies abruptly (OOM on a
    # shared box, 2026-09-24) mp.Pool loses the task and imap waits forever;
    # the executor raises BrokenProcessPool, so we rebuild it and carry on
    # with the games not yet recorded.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    attempts = 0
    while tasks and attempts < 5:
        attempts += 1
        pending = {}
        try:
            # no max_tasks_per_child: the executor can hang when a worker
            # retires (seen 2026-09-24: parents alive, zero workers, 40 min idle)
            with ProcessPoolExecutor(workers) as ex, \
                    open(out, "a", encoding="utf-8") as fh:
                pending = {ex.submit(_play, t): t for t in tasks}
                for fut in as_completed(pending):
                    row = fut.result()
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    t = pending[fut]
                    if "error" not in row:
                        done[(row["opp"], row["world"], row["seat"])] = row
                    n_done += 1
                    if progress and (n_done % 25 == 0 or n_done == total):
                        progress(n_done, total)
            tasks = []
        except BrokenProcessPool:
            tasks = [t for t in tasks if (t[1], t[3], t[5]) not in done]
            time.sleep(10)
    if tasks:
        raise RuntimeError(f"gate aborted: worker pool kept breaking; {len(tasks)} games unplayed (resumable)")
    return summarize(out, ours_path=ours_path, label=label, worlds=worlds)


def load(path):
    rows = {}
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "error" not in r:
            rows[(r["opp"], r["world"], r["seat"])] = r
    return rows


def summarize(path, ours_path="", label="", worlds=None):
    rows = load(path)
    errors = sum(1 for ln in open(path, encoding="utf-8") if '"error"' in ln)
    per = {}
    asym = []
    for (opp, world, seat), r in rows.items():
        if seat == 1:
            s0 = rows.get((opp, world, 0))
            if s0 and (s0["us"], s0["them"]) != (r["us"], r["them"]):
                asym.append((opp, world))
            continue
        d = per.setdefault(opp, {"w": 0, "l": 0, "t": 0, "gaps": []})
        d["w" if r["gap"] > 0 else "l" if r["gap"] < 0 else "t"] += 1
        d["gaps"].append(r["gap"])
    w = sum(d["w"] for d in per.values()); l = sum(d["l"] for d in per.values())
    t = sum(d["t"] for d in per.values()); n = w + l + t
    fams = {opp: {"w": d["w"], "l": d["l"], "t": d["t"],
                  "mean_gap": round(sum(d["gaps"]) / len(d["gaps"]), 1)}
            for opp, d in per.items()}
    losing = sorted([o for o, d in fams.items() if d["l"] > d["w"]],
                    key=lambda o: fams[o]["w"] - fams[o]["l"])
    return {"label": label, "file": P.rel(path), "ours": P.rel(ours_path) if ours_path else "",
            "worlds": worlds, "games": n, "wins": w, "losses": l, "ties": t,
            "pct": round(100 * (w + 0.5 * t) / n, 1) if n else None,
            "errors": errors, "asymmetric": asym[:20], "n_asymmetric": len(asym),
            "losing_families": losing, "families": fams}


def compare(path_a, path_b):
    """Paired McNemar A vs B over the seat-0 games both files share."""
    from kaggriculture.measure import win_metric as WM
    a, b = load(path_a), load(path_b)
    keys = sorted(k for k in a if k in b and k[2] == 0)
    sa = [WM.score(a[k]["us"], a[k]["them"]) for k in keys]
    sb = [WM.score(b[k]["us"], b[k]["them"]) for k in keys]
    res = WM.paired_test(sa, sb)
    res["n"] = len(keys)
    per = {}
    for k, x, y in zip(keys, sa, sb):
        d = per.setdefault(k[0], [0, 0])
        if x > y: d[0] += 1
        elif y > x: d[1] += 1
    res["per_family"] = {o: {"a_better": v[0], "b_better": v[1]} for o, v in per.items() if v != [0, 0]}
    return res


def spot_check(ours_path, opp_path, seeds):
    """Rust vs the vendored official engine, exact banks or bust."""
    import contextlib
    import importlib.util
    import io
    import sys
    if sys.path[:1] != [P.VENDOR]:
        sys.path.insert(0, P.VENDOR)
    from kaggle_environments import make
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    assert os.path.normcase(P.VENDOR) in os.path.normcase(K.__file__), K.__file__
    from kaggriculture.bandit.gate import loss_forensics as LF

    def load_official(p):
        spec = importlib.util.spec_from_file_location(f"sc{os.urandom(4).hex()}", p)
        m = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(m)
        f = m.agent
        n = f.__code__.co_argcount if hasattr(f, "__code__") else 2
        return (lambda o, c: f(o, c)) if n > 1 else (lambda o, c: f(o))

    out = []
    for seed in seeds:
        env = make("kaggriculture", configuration={"seed": seed, "actTimeout": 1, "runTimeout": 1200}, debug=False)
        with contextlib.redirect_stdout(io.StringIO()):
            # file paths, so the official loader picks the entry point exactly
            # as the ladder does (last callable, not `agent`)
            env.run([os.path.abspath(ours_path), os.path.abspath(opp_path)])
            official = tuple(float(s["reward"]) for s in env.steps[-1])
            _, _, us, them = LF.capture_game(LF.load_pyagent(ours_path), opp_path, seed, 0)
        out.append({"seed": seed, "official": official, "rust": (us, them), "exact": official == (us, them)})
    return out
