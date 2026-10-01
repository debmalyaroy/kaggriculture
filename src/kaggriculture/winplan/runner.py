"""Win-plan runner: executes the automated tasks of the plan, in dependency
order, and records everything.

State:  .local/winplan/state.json   (tasks, gates, feed -- the board's data)
Events: .local/winplan/events.log   one `WINPLAN {json}` line per change
        (kind: task | gate | feed). The session watching this file mirrors
        each event onto the War Room board.

    python -m kaggriculture.winplan.runner status
    python -m kaggriculture.winplan.runner run                 # all ready automated tasks
    python -m kaggriculture.winplan.runner run S1.1 S1.2       # specific tasks (deps first)
    python -m kaggriculture.winplan.runner run --force S0.5    # rerun a done task
    python -m kaggriculture.winplan.runner candidate LABEL PATH   # queue a build for S1.10

NEVER submits. Submission tasks (P.2-P.4) belong to the operator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback

from kaggriculture.paths import ROOT
from kaggriculture.winplan import paths as P


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ state --
import contextlib
import time

LOCK = P.STATE + ".lock"


@contextlib.contextmanager
def _locked():
    """Cross-process lock: the gate chain and the ladder chain run as separate
    processes and both update state.json."""
    for _ in range(600):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 60:      # stale lock from a crash
                    os.remove(LOCK)
            except OSError:
                pass
            time.sleep(0.1)
    else:
        raise RuntimeError("state lock timeout")
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(LOCK)
        except OSError:
            pass


def load_state():
    try:
        return json.load(open(P.STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"tasks": {}, "gates": {}, "feed": []}


def save_state(st):
    tmp = P.STATE + f".{os.getpid()}.tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, P.STATE)


@contextlib.contextmanager
def edit_state():
    with _locked():
        st = load_state()
        yield st
        save_state(st)


def emit(kind, payload):
    with open(P.EVENTS, "a", encoding="utf-8") as fh:
        fh.write("WINPLAN " + json.dumps({"kind": kind, "t": now(), **payload}) + "\n")


def set_task(tid, **fields):
    with edit_state() as st:
        rec = st["tasks"].setdefault(tid, {})
        rec.update(fields); rec["updated"] = now()
        snap = {k: rec.get(k) for k in ("status", "result", "progress")}
    emit("task", {"id": tid, **snap})


def add_gate(gid, summary, note=""):
    g = {"label": summary.get("label", gid), "wins": summary.get("wins"), "losses": summary.get("losses"),
         "pct": summary.get("pct"), "worlds": summary.get("worlds"), "note": note, "updated": now(),
         "file": summary.get("file")}
    with edit_state() as st:
        st["gates"][gid] = g
    emit("gate", {"id": gid, **g})


def feed(msg):
    with edit_state() as st:
        st["feed"] = (st["feed"] + [{"t": now(), "msg": msg}])[-60:]
    emit("feed", {"msg": msg})


def _progress(tid, what):
    last = [-1]

    def cb(done, total):
        pct = int(100 * done / total) if total else 100
        if pct // 10 != last[0] // 10 or done == total:          # every 10%
            last[0] = pct
            set_task(tid, status="running", progress=f"{what}: {done}/{total} games ({pct}%)")
    return cb


# ------------------------------------------------------------------ tasks --
def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _agent_path(name, cfg):
    if name == cfg["baseline"]["label"]:
        return _abs(cfg["baseline"]["path"])
    for c in cfg.get("candidates", []):
        if c["label"] == name:
            return _abs(c["path"])
    return os.path.join(P.FIELD, name + ".py")


def _fmt_losers(summary, k=6):
    f = summary["families"]
    return ", ".join(f"{o} {f[o]['w']}-{f[o]['l']}" for o in summary["losing_families"][:k]) or "none"


def t_harvest(cfg):
    from kaggriculture.winplan import harvest as HV
    HV.migrate_legacy()
    res = HV.harvest(cfg["harvest_since"], log=lambda m: set_task("P.5", status="running", progress=m))
    return (f"{res['notebooks']} notebooks since {cfg['harvest_since']}, {res['packaged']} packaged, "
            f"{len(res['new_agents'])} new agents; field = {res['field_size']} agents"
            + (f". New: {', '.join(res['new_agents'][:8])}" if res["new_agents"] else ""))


def t_spot(cfg):
    from kaggriculture.winplan import gate as G
    from kaggriculture.winplan import harvest as HV
    HV.migrate_legacy()                       # fast: makes sure the field exists
    sc = cfg["spot_check"]
    res = G.spot_check(_abs(cfg["baseline"]["path"]), os.path.join(P.FIELD, sc["opponent"] + ".py"), sc["seeds"])
    ok = all(r["exact"] for r in res)
    detail = "; ".join(f"seed {r['seed']}: rust {r['rust'][0]:.0f}/{r['rust'][1]:.0f} vs official "
                       f"{r['official'][0]:.0f}/{r['official'][1]:.0f}" for r in res)
    if not ok:
        raise RuntimeError("ENGINE DRIFT -- pipeline stopped. " + detail)
    return "Exact on all seeds. " + detail


def t_baseline(cfg):
    from kaggriculture.winplan import gate as G
    b = cfg["baseline"]
    s = G.run_gate(_abs(b["path"]), f"base_{b['label']}", G.roster(), worlds=cfg["worlds"],
                   symmetry_worlds=cfg["symmetry_worlds"], progress=_progress("S0.3", f"{b['label']} vs field"))
    add_gate(f"g-{b['label']}-{cfg['worlds']}w", s, note=f"Loses to: {_fmt_losers(s, 4)}")
    return (f"{s['wins']}W-{s['losses']}L-{s['ties']}T of {s['games']} ({s['pct']}%), {len(G.roster())} opponents x "
            f"{cfg['worlds']} worlds. Losing families: {_fmt_losers(s)}. Asymmetric seat pairs: {s['n_asymmetric']}. "
            f"Errors: {s['errors']}.")


def t_roundrobin(cfg):
    from kaggriculture.winplan import gate as G
    names = cfg["roundrobin"]
    score = {n: [0, 0, 0] for n in names}                          # w, l, t over all pairings
    matrix = {}
    for i, a in enumerate(names):
        opps = [(b, _agent_path(b, cfg)) for b in names[i + 1:]]
        if not opps:
            continue
        s = G.run_gate(_agent_path(a, cfg), f"rr_{a}", opps, worlds=cfg["worlds"], symmetry_worlds=0,
                       progress=_progress("S1.1", f"{a} vs {len(opps)} ({i + 1}/{len(names) - 1})"))
        for b, fam in s["families"].items():
            matrix[f"{a}|{b}"] = fam
            score[a][0] += fam["w"]; score[a][1] += fam["l"]; score[a][2] += fam["t"]
            score[b][0] += fam["l"]; score[b][1] += fam["w"]; score[b][2] += fam["t"]
    rank = sorted(names, key=lambda n: -(score[n][0] + 0.5 * score[n][2]) / max(1, sum(score[n])))
    json.dump({"rank": rank, "score": score, "matrix": matrix},
              open(os.path.join(P.DATA, "roundrobin.json"), "w"), indent=1)
    with edit_state() as st:
        st["roundrobin_winner"] = rank[0]
    lines = [f"{n}: {score[n][0]}-{score[n][1]}" + (f"-{score[n][2]}T" if score[n][2] else "") for n in rank]
    return f"Winner: {rank[0]}. Standings (W-L over {cfg['worlds']} worlds per pairing): " + "; ".join(lines)


def t_winner_gate(cfg):
    from kaggriculture.winplan import gate as G
    win = load_state().get("roundrobin_winner")
    if not win:
        raise RuntimeError("no round-robin winner yet (run S1.1)")
    s = G.run_gate(_agent_path(win, cfg), f"base_{win}", G.roster(exclude={win}), worlds=cfg["worlds"],
                   symmetry_worlds=cfg["symmetry_worlds"], progress=_progress("S1.2", f"{win} vs field"))
    add_gate(f"g-{win}-{cfg['worlds']}w", s, note=f"Round-robin winner. Loses to: {_fmt_losers(s, 4)}")
    return (f"{win}: {s['wins']}W-{s['losses']}L-{s['ties']}T ({s['pct']}%) vs {len(G.roster(exclude={win}))} "
            f"opponents. Losing families: {_fmt_losers(s)}.")


def t_candidates(cfg):
    """Gate every queued candidate build and compare it, paired, to its reference."""
    from kaggriculture.winplan import gate as G
    st = load_state(); gated = st.setdefault("candidates_done", {})
    todo = [c for c in cfg.get("candidates", []) if c["label"] not in gated]
    if not todo:
        return None                                                   # nothing to do: stay waiting
    lines = []
    for c in todo:
        s = G.run_gate(_abs(c["path"]), f"cand_{c['label']}", G.roster(), worlds=cfg["worlds"],
                       symmetry_worlds=cfg["symmetry_worlds"], progress=_progress("S1.10", c["label"]))
        ref_label = c.get("vs", cfg["baseline"]["label"])
        ref = [f for f in os.listdir(P.GATES) if f.startswith(f"base_{ref_label}__") or f.startswith(f"cand_{ref_label}__")]
        cmp = G.compare(os.path.join(ROOT, s["file"]),
                        os.path.join(P.GATES, sorted(ref)[-1])) if ref else None
        verdict = ""
        if cmp:
            verdict = (f" vs {ref_label}: better in {cmp['better_a']}, worse in {cmp['better_b']} paired games, "
                       f"p={cmp['p_value']:.3f}, {cmp['score_diff']:+.3f} score/game"
                       + (" (significant)" if cmp["significant"] else ""))
        add_gate(f"g-cand-{c['label']}", s, note=(c.get("note", "") + verdict).strip())
        with edit_state() as st:
            st.setdefault("candidates_done", {})[c["label"]] = {"summary": s, "compare": cmp}
        lines.append(f"{c['label']}: {s['wins']}-{s['losses']} ({s['pct']}%){verdict}. Loses to: {_fmt_losers(s, 4)}")
    return " | ".join(lines)


def t_crawl(cfg):
    from kaggriculture.winplan import ladder as L
    L.adopt_legacy()
    try:
        c = L.crawl(cfg["crawl_seed_subs"], min_rating=cfg["crawl_min_rating"],
                    log=lambda m: set_task("S2.1", status="running", progress=m))
    except RuntimeError as exc:
        if not os.path.exists(os.path.join(P.CRAWL, "episodes.json")):
            raise
        feed(f"S2.1: fresh crawl failed ({exc}); using the existing crawl")
        eps = json.load(open(os.path.join(P.CRAWL, "episodes.json")))
        c = {"submissions": len(json.load(open(os.path.join(P.CRAWL, "subs.json")))), "episodes": len(eps),
             "top_vs_top": sum(1 for e in eps.values() if len(e["agents"]) == 2 and all(
                 (a.get("initialScore") or 0) >= cfg["crawl_min_rating"] and a.get("reward") is not None
                 for a in e["agents"]))}
    d = L.download_top(floor=cfg["top_floor"], budget=cfg["replay_budget"],
                       log=lambda m: set_task("S2.1", status="running", progress=m))
    return (f"Crawl: {c['submissions']} submissions, {c['episodes']} episodes, {c['top_vs_top']} top-vs-top. "
            f"Replays: {d['downloaded']} new ({d['failed']} failed), {d['on_disk']} on disk, {d['teams']} teams.")


def t_mine(cfg):
    from kaggriculture.winplan import ladder as L
    e = L.economy(log=lambda m: set_task("S2.2", status="running", progress=m))
    if not e.get("games"):
        raise RuntimeError("no replays on disk to mine (run S2.1)")
    r = L.extract_routes()
    top_rev = sorted(e["winners_rev_by_product"].items(), key=lambda kv: -kv[1])[:4]
    return (f"{e['games']} games, {e['teams']} teams. Winners bank ${e['winners_bank']:,}; revenue leaders "
            + ", ".join(f"{k} ${v:,}" for k, v in top_rev)
            + f"; first land @ step {e['winners_first_land_step']}, {e['winners_land_buys']} land buys. "
            f"Routes: {r['routes']} ({r['winning_routes']} winning) over {r['worlds_covered']} shop pairs.")


TASKS = {
    # id: (fn, deps, "what the board calls it")
    "P.5": (t_harvest, [], "harvest"),
    "S0.5": (t_spot, [], "engine spot check"),
    "S0.3": (t_baseline, ["S0.5"], "baseline gate"),
    "S1.1": (t_roundrobin, ["S0.5"], "round-robin"),
    "S1.2": (t_winner_gate, ["S1.1"], "winner gate"),
    "S1.10": (t_candidates, ["S0.3"], "candidate gates"),
    "S2.1": (t_crawl, [], "crawl + replays"),
    "S2.2": (t_mine, ["S2.1"], "economy + routes"),
}
ORDER = ["S0.5", "S0.3", "S1.1", "S1.2", "S1.10", "S2.1", "S2.2", "P.5"]


def run(ids=None, force=False):
    cfg = P.config()
    want = ids or ORDER
    # pull in dependencies (they are skipped if already done)
    closure, stack = [], list(want)
    while stack:
        t = stack.pop()
        if t not in TASKS:
            print(f"unknown or manual task: {t}"); continue
        if t not in closure:
            closure.append(t); stack += TASKS[t][1]
    plan = [t for t in ORDER if t in closure]
    for tid in plan:
        st = load_state()
        status = st["tasks"].get(tid, {}).get("status")
        if status == "done" and not (force and tid in want) and tid != "S1.10":
            continue
        deps_bad = [d for d in TASKS[tid][1] if load_state()["tasks"].get(d, {}).get("status") != "done"]
        if deps_bad:
            set_task(tid, status="blocked", progress=f"waiting on {', '.join(deps_bad)}")
            continue
        # clear the previous run's result so a stale error doesn't ride along
        set_task(tid, status="running", result="", progress=f"started {TASKS[tid][2]}")
        try:
            result = TASKS[tid][0](cfg)
        except Exception as exc:                                     # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            set_task(tid, status="failed", result=msg[:600], progress="")
            feed(f"{tid} failed: {msg[:200]}")
            traceback.print_exc()
            if tid == "S0.5":
                print("engine drift -- stopping"); return 2
            continue
        if result is None:
            set_task(tid, status="todo", progress="nothing queued")
            continue
        set_task(tid, status="done", result=result[:1500], progress="")
        feed(f"{tid} done: {result[:220]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="win-plan harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("ids", nargs="*"); r.add_argument("--force", action="store_true")
    sub.add_parser("status")
    c = sub.add_parser("candidate"); c.add_argument("label"); c.add_argument("path")
    c.add_argument("--vs", default=None); c.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        st = load_state()
        for tid in ORDER:
            t = st["tasks"].get(tid, {})
            print(f"{tid:6s} {t.get('status', 'todo'):8s} {t.get('progress') or ''} {(t.get('result') or '')[:140]}")
        return 0
    if a.cmd == "candidate":
        cfg = json.load(open(P.CONFIG, encoding="utf-8"))
        cfg["candidates"] = [x for x in cfg.get("candidates", []) if x["label"] != a.label] + [
            {"label": a.label, "path": a.path, "vs": a.vs or cfg["baseline"]["label"], "note": a.note}]
        json.dump(cfg, open(P.CONFIG, "w", encoding="utf-8"), indent=2)
        feed(f"Queued candidate {a.label} for gating")
        return 0
    return run(a.ids, a.force)


if __name__ == "__main__":
    sys.exit(main())
