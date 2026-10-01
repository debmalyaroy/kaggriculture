"""Python bridge to the Rust engine: mass open-loop pre-ranking of routes.

The Rust port measures 103,260 steps/s = 143 episodes/s on ONE core against
the official engine's 0.36 ep/s at 3 workers -- ~400x. That converts "which
of 12,000 routes deserve official-engine games" from impossible to ~90
seconds. PRE-RANKER ONLY: every release decision stays on the official
engine; this narrows candidates, it never crowns them.

Two distinct validity questions, guarded separately:

* ENGINE FIDELITY -- does Rust compute the same game as Python? Proven
  bit-exact per step by tests/test_rust_engine.py (chaos + real replays), and
  re-proven continuously by the pipeline's daily parity stage.
* RANKING RECALL -- open-loop schedules approximate closed-loop agents (the
  repair/guard layers react to the game; a tape cannot). Does the open-loop
  ranking still put the official engine's winners on top? Measured by
  `--recall`, recorded to models/lab/preranker_recall.json, and consumed by
  train_gates.preranker_allowed(). Speed is NOT the acceptance criterion --
  a fast pre-ranker that ranks badly makes the crown funnel wider AND worse.

    python src/rust_prerank.py --rank --top 40          # rank fresh routes
    python src/rust_prerank.py --recall 8               # measure recall@N
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

KAGG = os.path.join(ROOT, "rustengine", "target", "release",
                    "kagg.exe" if os.name == "nt" else "kagg")
TAPE_DIR = os.path.join(ROOT, ".local", "rustdiff", "prerank")
RECALL_OUT = os.path.join(ROOT, "models", "lab", "preranker_recall.json")


def _norm_tokens(a):
    if not isinstance(a, list) or not a:
        return ["PASS"]
    out = []
    for v in a:
        if isinstance(v, bool):
            return ["PASS"]
        if isinstance(v, (int, float)):
            out.append(str(int(v)))
        elif isinstance(v, str) and " " not in v and "\t" not in v:
            out.append(v)
        else:
            return ["PASS"]
    return out


def _line(turn):
    turn = turn if isinstance(turn, dict) else {}
    farmer = " ".join(_norm_tokens(turn.get("farmer")))
    hands = ";".join(" ".join(_norm_tokens(h))
                     for h in (turn.get("hands") or []))
    orders = []
    for o in (turn.get("market") or []):
        t = _norm_tokens(o)
        if t != ["PASS"]:
            orders.append(" ".join(t))
    return f"{farmer}\t{hands}\t{';'.join(orders)}"


def build_tape(actions_a, actions_b, seed, name):
    os.makedirs(TAPE_DIR, exist_ok=True)
    path = os.path.join(TAPE_DIR, f"{name}.tape")
    n = min(719, max(len(actions_a), len(actions_b)))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {seed}\n")
        for t in range(n):
            fh.write(_line(actions_a[t] if t < len(actions_a) else {}) + "\n")
            fh.write(_line(actions_b[t] if t < len(actions_b) else {}) + "\n")
    return path


def play_tape(path):
    """(bank_a, bank_b) from one Rust episode. ~7 ms."""
    out = subprocess.run([KAGG, "episode", path], capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:300])
    import struct
    for line in out.stdout.splitlines():
        if line.startswith("FINAL "):
            _, b0, b1 = line.split()
            return (struct.unpack("<d", struct.pack("<Q", int(b0)))[0],
                    struct.unpack("<d", struct.pack("<Q", int(b1)))[0])
    raise RuntimeError("no FINAL line")


def rank(candidate_ids, referee_ids, seeds=(60000, 150000), log=print):
    """Candidates ordered by expected SCORE over the referee schedules.

    Open-loop both sides: candidate tape vs referee tape, both seats' games
    priced by win/draw/loss exactly like win_metric.score(). Margin is a
    tie-break only.
    """
    import kaggriculture.data.routes as R
    import kaggriculture.measure.win_metric as WM
    loaded = {}
    for rid in set(candidate_ids) | set(referee_ids):
        try:
            loaded[rid] = R.load_route(rid)
        except Exception:                                       # noqa: BLE001
            continue
    rows = []
    for cid in candidate_ids:
        if cid not in loaded:
            continue
        pts = games = 0.0
        margin = 0.0
        for ref in referee_ids:
            if ref not in loaded or ref == cid:
                continue
            for seed in seeds:
                tape = build_tape(loaded[cid], loaded[ref], seed,
                                  f"{cid}_vs_{ref}_{seed}"[:120])
                a, b = play_tape(tape)
                pts += WM.score(a, b)
                games += 1
                margin += a - b
                os.remove(tape)
        if games:
            rows.append((pts / games, margin / games, cid))
    rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
    for sc, mg, cid in rows[:20]:
        log(f"  {cid:<28} score {sc:.3f}  (margin diag {mg:+,.0f})")
    return [cid for _, _, cid in rows]


def measure_recall(n_candidates=8, n_official_refs=3, ref_rank=(80, 400),
                   log=print):
    """Recall@N: does the Rust open-loop TOP-N contain the official engine's
    closed-loop top-N? The expensive side runs the real routes.py-built agents
    through evaluate.py, so this measures the whole approximation gap --
    open-loop tape vs repaired closed-loop agent -- not just engine fidelity.

    Referees come from a DEEPER rank band than the candidates on purpose: the
    first measurement used near-top referees and the official side saturated
    (six candidates at 0/6, two at 6/6), which makes ranking inside the tie
    arbitrary and the recall estimate coarse. A referee only informs when
    some candidates beat it and some do not.
    """
    import kaggriculture.data.routes as R
    idx = R.load_index()
    fresh = sorted((r for r in idx["routes"].values()
                    if (r.get("rank") or 999) <= 60),
                   key=lambda r: r.get("date") or "", reverse=True)
    cands = [r["id"] for r in fresh[:n_candidates]]
    mid = sorted((r for r in idx["routes"].values()
                  if ref_rank[0] <= (r.get("rank") or 999) <= ref_rank[1]),
                 key=lambda r: r.get("date") or "", reverse=True)
    refs = [r["id"] for r in mid[:n_official_refs]]
    if len(refs) < n_official_refs:      # sparse mid-ladder: fall back
        refs += [r["id"] for r in fresh[n_candidates:n_candidates
                                        + n_official_refs - len(refs)]]
    if len(cands) < 4 or not refs:
        log("not enough fresh routes for a recall measurement")
        return None
    log(f"recall: {len(cands)} candidates x {len(refs)} referees")

    rust_order = rank(cands, refs, log=lambda *_: None)

    # Official side: build each candidate as a real agent, play the same
    # referees as tapes through evaluate.py (closed-loop, repair layers live).
    import kaggriculture.measure.win_metric as WM
    cdir = os.path.join(ROOT, ".local", "rustdiff", "recall_agents")
    os.makedirs(cdir, exist_ok=True)
    ref_agents = []
    for ref in refs:
        out = os.path.join(cdir, f"ref_{ref}.py")
        subprocess.run([sys.executable, "src/kaggriculture/data/routes.py", "--build", ref,
                        "--out", os.path.relpath(out, ROOT)],
                       capture_output=True, cwd=ROOT, timeout=600)
        if os.path.exists(out):
            ref_agents.append(out)
    official = []
    for cid in cands:
        agent = os.path.join(cdir, f"c_{cid}.py")
        subprocess.run([sys.executable, "src/kaggriculture/data/routes.py", "--build", cid,
                        "--out", os.path.relpath(agent, ROOT)],
                       capture_output=True, cwd=ROOT, timeout=600)
        if not os.path.exists(agent):
            continue
        p = subprocess.run([sys.executable, "src/kaggriculture/measure/evaluate.py", agent,
                            "--vs", *ref_agents, "-n", "1",
                            "--seed0", "60000", "--no-record"],
                           capture_output=True, text=True, cwd=ROOT,
                           timeout=3600)
        try:
            rows = WM.parse_eval(p.stdout + p.stderr)
        except ValueError as exc:
            log(f"  ! {cid}: {exc}")
            continue
        w = sum(r["wins"] for r in rows)
        g = sum(r["games"] for r in rows)
        official.append((-(w / g if g else 0), cid))
        log(f"  official {cid}: {w}/{g}")
    official.sort()
    off_order = [cid for _, cid in official]

    n = max(2, len(off_order) // 2)
    top_rust = set(rust_order[:n])
    top_off = set(off_order[:n])
    recall = len(top_rust & top_off) / max(1, len(top_off))
    payload = {"recall_at_n": recall, "n": n,
               "candidates": len(off_order),
               "rust_order": rust_order, "official_order": off_order}
    os.makedirs(os.path.dirname(RECALL_OUT), exist_ok=True)
    json.dump(payload, open(RECALL_OUT, "w", encoding="utf-8"), indent=1)
    log(f"recall@{n}: {recall:.2f} -> {RECALL_OUT}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--refs", type=int, default=6)
    ap.add_argument("--recall", type=int, metavar="N",
                    help="measure recall@ with N candidates")
    args = ap.parse_args()
    if not os.path.exists(KAGG):
        print("rust binary missing: cd rustengine && cargo build --release")
        return 1
    if args.recall:
        measure_recall(args.recall)
        return 0
    if args.rank:
        import kaggriculture.data.routes as R
        idx = R.load_index()
        fresh = sorted((r for r in idx["routes"].values()
                        if (r.get("rank") or 999) <= 60),
                       key=lambda r: r.get("date") or "", reverse=True)
        cands = [r["id"] for r in fresh[:args.top]]
        refs = [r["id"] for r in fresh[args.top:args.top + args.refs]]
        import time
        t0 = time.time()
        order = rank(cands, refs)
        print(f"{len(order)} candidates ranked in {time.time() - t0:.1f}s")
        return 0
    print(__doc__.splitlines()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
