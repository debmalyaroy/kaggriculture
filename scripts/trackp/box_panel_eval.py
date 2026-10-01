#!/usr/bin/env python
"""(13) BOX PANEL EVALUATOR -- runs ON THE BOX, CPU-only, non-disruptive.

The RL loop trains by fast NN self-play and never checks the policy against the
real panel. This daemon closes that gap WITHOUT touching the training process:

  every EVERY steps (checked each POLL):
    1. snapshot the live policy_rl.pt (retry until a clean, non-mid-write copy)
    2. play it vs the top-agents panel on `kagg serve` (CPU -> ZERO GPU contention)
    3. append {gstep, panel_score, per-opp} to ckpts/panel_eval.jsonl (the curve)
    4. if panel_score improved -> save ckpts/policy_rl_best.pt (+ .panel_best.json)
       and upload it + the curve to gdrive (the best-by-REAL-metric checkpoint,
       which is the build-gate signal -- RL is NOT monotonic, so latest != best)

Runs on CPU so it never steals the training GPU. Reactive opponents are real
Python kernels driven through serve (exact-bank vs official). Launch detached,
like the archiver:

  cd /root/kaggriculture
  nohup setsid /venv/main/bin/python scripts/trackp/box_panel_eval.py \
      >ckpts/panel_eval.log 2>&1 </dev/null &

Stop:  pkill -f box_panel_eval.py
NEVER submits. NEVER writes policy_rl.pt (the trainer owns it); only reads it.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))

CKPTS = os.environ.get("CKPTS", os.path.join(ROOT, "ckpts"))
GDRIVE = os.environ.get("GDRIVE", "gdrive:kaggriculture")
ARCH = GDRIVE + "/ckpt_archive"
RAILS_PATH = os.path.join(ROOT, "configs", "trackp_rails.json")


def _log(msg):
    print(f"[{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}] {msg}", flush=True)


def _rclone(src, dst, timeout):
    """Bounded rclone copyto -- NEVER wedge the eval loop on slow/stalled gdrive
    egress. Returns True on success; False on failure/timeout (caller keeps the
    authoritative local copy and re-uploads on the next best)."""
    try:
        r = subprocess.run(["rclone", "copyto", src, dst],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def gstep(pt):
    import torch
    try:
        return int(torch.load(pt, map_location="cpu", weights_only=False).get("global_step", 0))
    except Exception:
        return -1                      # mid-write / truncated -> caller retries


ROSTER_PATH = os.path.join(ROOT, "configs", "panel_roster.json")


def _panel(n):
    """Prefer the PINNED roster (configs/panel_roster.json) so the panel is
    IDENTICAL on the box and on the laptop; fall back to top_reactive_agents.
    Roster paths are repo-relative -> resolved against ROOT (works cross-OS)."""
    out = []
    if os.path.exists(ROSTER_PATH):
        for r in json.load(open(ROSTER_PATH)):
            p = os.path.join(ROOT, r["path"])
            if os.path.exists(p):
                out.append((str(r["name"])[:34], p, float(r.get("rating") or 0)))
        if out:
            return out
        _log(f"roster {ROSTER_PATH} present but 0 files resolved -- falling back")
    from kaggriculture.data import selfplay_corpus as SC
    for name, path, rating in SC.top_reactive_agents(n):
        if path and os.path.exists(path):
            out.append((str(name)[:34], path, float(rating or 0)))
    return out


def _snapshot(src, dst, tries=6, wait=25):
    """Copy a clean (fully-written) policy_rl.pt. RL rewrites it every ~150s, so
    a copy can catch a partial write -- verify by loading global_step, retry."""
    import torch
    for _ in range(tries):
        try:
            shutil.copy2(src, dst)
            torch.load(dst, map_location="cpu", weights_only=False)   # integrity check
            return True
        except Exception:
            time.sleep(wait)
    return False


def eval_ckpt(ckpt, panel, rails, seeds, seed0):
    """Play `ckpt` (loaded ONCE, CPU) vs every panel opponent, seat-alternated.
    Returns (panel_score, detail). Opponents reloaded per game (fresh state)."""
    from kaggriculture.trackp.policy_agent import make_agent
    from kaggriculture.engine import serve_match as SM
    cand = make_agent(ckpt, rail_config=rails, sample=False, device="cpu")
    srv = SM.Serve()
    W = D = L = 0
    bank = opp = 0.0
    per = {}
    try:
        for (onm, opath, ort) in panel:
            w = d = lo = 0
            for i in range(seeds):
                seed = seed0 + i
                op = SM.load_agent(opath)
                try:
                    if i % 2 == 0:
                        b0, b1 = SM.run_match(cand, op, seed, srv)
                    else:
                        b1, b0 = SM.run_match(op, cand, seed, srv)
                except Exception as e:
                    _log(f"  vs {onm} seed{seed} FAILED: {e}"); continue
                bank += b0; opp += b1
                w += b0 > b1; d += b0 == b1; lo += b0 < b1
            g = max(1, w + d + lo)
            per[onm] = {"rating": ort, "score": round((w + 0.5 * d) / g, 4), "w": w, "d": d, "l": lo}
            W += w; D += d; L += lo
    finally:
        srv.close()
    G = max(1, W + D + L)
    score = round((W + 0.5 * D) / G, 4)
    return score, {"score": score, "w": W, "d": D, "l": L, "games": G,
                   "mean_bank": round(bank / G, 1), "mean_opp": round(opp / G, 1), "per_opp": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=30_000_000, help="RL steps between evals")
    ap.add_argument("--seeds", type=int, default=6, help="worlds/seeds per opponent")
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--panel-n", type=int, default=24, help="top_reactive_agents(N) request")
    ap.add_argument("--poll-min", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (keep low so RL's CPU-side isn't starved)")
    ap.add_argument("--once", action="store_true", help="evaluate the current ckpt once and exit")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, a.threads))
    rails = json.load(open(RAILS_PATH)) if os.path.exists(RAILS_PATH) else None
    panel = _panel(a.panel_n)
    if not panel:
        raise SystemExit("empty panel -- check crown_panel.json / data/kernels on this box")

    PT = os.path.join(CKPTS, "policy_rl.pt")
    SNAP = os.path.join(CKPTS, "policy_rl.evalsnap.pt")
    BEST = os.path.join(CKPTS, "policy_rl_best.pt")
    BEST_JSON = os.path.join(CKPTS, ".panel_best.json")
    CURVE = os.path.join(CKPTS, "panel_eval.jsonl")
    STATE = os.path.join(CKPTS, ".panel_eval_state")   # last gstep evaluated

    best = json.load(open(BEST_JSON)) if os.path.exists(BEST_JSON) else {"score": -1.0, "gstep": 0}
    last = int(open(STATE).read().strip()) if os.path.exists(STATE) else -1
    _log(f"panel evaluator start: {len(panel)} opponents, seeds={a.seeds}, every={a.every} steps, "
         f"CPU threads={a.threads}, best so far score={best.get('score')} @ {best.get('gstep')}")

    def run_eval():
        nonlocal best
        if not os.path.exists(PT):
            _log("policy_rl.pt missing -- skip"); return
        if not _snapshot(PT, SNAP):
            _log("could not get a clean snapshot -- will retry next poll"); return
        gs = gstep(SNAP)
        t0 = time.time()
        score, detail = eval_ckpt(SNAP, panel, rails, a.seeds, a.seed0)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rec = {"stamp": stamp, "gstep": gs, "eval_sec": round(time.time() - t0, 1), **detail}
        with open(CURVE, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _log(f"gstep={gs:,} panel_score={score} ({detail['w']}-{detail['d']}-{detail['l']}) "
             f"bank {detail['mean_bank']:.0f} vs {detail['mean_opp']:.0f}  ({rec['eval_sec']:.0f}s)")
        # upload the curve every eval (tiny, keeps a durable record off-box)
        _rclone(CURVE, f"{ARCH}/panel_eval.jsonl", 120)
        # --- FEEDBACK: keep the best-by-panel-score checkpoint ---
        if score > best.get("score", -1.0):
            shutil.copy2(SNAP, BEST)                       # authoritative copy is LOCAL
            best = {"score": score, "gstep": gs, "stamp": stamp,
                    "w": detail["w"], "d": detail["d"], "l": detail["l"],
                    "mean_bank": detail["mean_bank"], "mean_opp": detail["mean_opp"]}
            json.dump(best, open(BEST_JSON, "w"), indent=2)
            # gdrive-mirror the 70 MB best ONLY when it is worth the egress -- a
            # 0.0 baseline is not; mirror once the policy actually beats someone.
            if score > 0.0:
                ok = ("uploaded" if _rclone(BEST, f"{ARCH}/policy_rl_best.pt", 900)
                      else "UPLOAD slow/failed (local kept; retries on next best)")
                _rclone(BEST_JSON, f"{ARCH}/policy_rl_best.json", 60)
            else:
                ok = "kept LOCAL only (score 0 -> skip 70MB egress)"
            _log(f"  *** NEW BEST panel_score={score} @ {gs:,} -> policy_rl_best.pt ({ok})")
        try:
            os.remove(SNAP)
        except OSError:
            pass
        with open(STATE, "w") as fh:
            fh.write(str(gs))

    if a.once:
        run_eval(); return
    while True:
        gs = gstep(PT)
        if gs >= 0 and gs >= last + a.every:
            run_eval()
            last = gs
        time.sleep(a.poll_min * 60)


if __name__ == "__main__":
    main()
