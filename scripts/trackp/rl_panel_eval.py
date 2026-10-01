#!/usr/bin/env python
"""(12) RL PANEL EVAL -- neural trackp candidates vs the top-agents panel on the
serve FAST PATH.

For each candidate (an agent .py that exports `agent`), play every panel
opponent across N worlds/seeds on the Rust `kagg serve` engine (exact-bank vs
official for reactive agents). Seat is ALTERNATED across seeds so each candidate
sees both seats on a comparable (opponent, seed, seat) grid.

Two efficiency rules that matter for a NEURAL candidate:
  * the candidate model is loaded ONCE and reused (it is stateless per game --
    argmax over a frozen policy), avoiding a 70 MB reload per game;
  * each OPPONENT is reloaded per game (reactive agents carry per-episode module
    state; reusing a stale instance would corrupt it).

MUST be run with the torch env (the neural policy needs torch):
    C:/ProgramData/anaconda3/envs/llm/python.exe scripts/trackp/rl_panel_eval.py \
        --candidates .local/checkpoints/self_play/agent_rl_30M.py \
                     .local/checkpoints/self_play/agent_rl_60M.py \
                     .local/checkpoints/self_play/agent_rl_90M.py \
        --seeds 24 --panel-n 24 --out .local/checkpoints/self_play/panel_eval.json

Note: each neural game is ~15-20s (per-turn batch-1 inference dominates, NOT the
engine), so 3 candidates x 11 opponents x 24 seeds ~= 3.5-4h. Reduce --seeds or
--panel-n for a quick pass. NEVER submits.
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))


def _panel(n):
    """[(name, path, rating)] top reactive agents (the top-agents panel)."""
    from kaggriculture.data import selfplay_corpus as SC
    out = []
    for name, path, rating in SC.top_reactive_agents(n):
        if path and os.path.exists(path):
            out.append((str(name)[:34], path, float(rating or 0)))
    return out


def _verify(cand_agent, name, srv, SM):
    """Play one short game vs PASS to confirm the agent loads and emits legal
    actions (serve raises on an illegal/desynced action)."""
    def pass_agent(obs, config=None):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    try:
        b0, b1 = SM.run_match(cand_agent, pass_agent, 3, srv)
        return True, f"OK (bank {b0:.0f} vs PASS {b1:.0f})"
    except Exception as e:
        return False, f"FAIL: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", nargs="+", required=True, help="agent .py files")
    ap.add_argument("--seeds", type=int, default=24, help="worlds/seeds per opponent")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--panel-n", type=int, default=24, help="top_reactive_agents(N) request")
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "checkpoints", "self_play", "panel_eval.json"))
    a = ap.parse_args()

    from kaggriculture.engine import serve_match as SM

    panel = _panel(a.panel_n)
    if not panel:
        raise SystemExit("empty panel -- check models/crown_panel.json / data/kernels")
    print(f"[panel] {len(panel)} top-agents opponents, {a.seeds} seeds each, seat-alternated")
    for nm, _, rt in panel:
        print(f"    {nm:<34} rating={rt:.0f}")

    srv = SM.Serve()
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": a.seeds,
              "panel": [{"name": n, "rating": r} for n, _, r in panel], "candidates": {}}
    try:
        # --- load each candidate ONCE (stateless neural policy) + verify ---
        cands = []
        print("\n=== verify candidates ===")
        for cf in a.candidates:
            cf_abs = cf if os.path.isabs(cf) else os.path.join(ROOT, cf)
            label = os.path.splitext(os.path.basename(cf_abs))[0]
            t0 = time.time()
            ag = SM.load_agent(cf_abs)                 # execs wrapper -> make_agent (once)
            ok, msg = _verify(ag, label, srv, SM)
            print(f"  {label:<16} loaded {time.time()-t0:.1f}s  {msg}")
            report["candidates"][label] = {"file": cf_abs, "verify_ok": ok, "verify": msg,
                                           "vs": {}, "totals": {}}
            if ok:
                cands.append((label, ag))
        if not cands:
            raise SystemExit("no candidate passed verification")

        # --- panel loop ---
        print("\n=== panel eval (fast path) ===")
        t_start = time.time()
        for label, ag in cands:
            agg_w = agg_d = agg_l = 0
            agg_bank = agg_opp = 0.0
            for (onm, opath, ort) in panel:
                w = d = lo = 0
                bank = opp_bank = 0.0
                for i in range(a.seeds):
                    seed = a.seed0 + i
                    opp = SM.load_agent(opath)         # fresh per game (stateful reactive)
                    try:
                        if i % 2 == 0:                 # alternate seats
                            b0, b1 = SM.run_match(ag, opp, seed, srv)
                        else:
                            b1, b0 = SM.run_match(opp, ag, seed, srv)
                    except Exception as e:
                        print(f"    {label} vs {onm} seed{seed} FAILED: {e}")
                        continue
                    bank += b0; opp_bank += b1
                    w += b0 > b1; d += b0 == b1; lo += b0 < b1
                g = max(1, w + d + lo)
                score = (w + 0.5 * d) / g
                report["candidates"][label]["vs"][onm] = {
                    "rating": ort, "score": round(score, 4), "w": w, "d": d, "l": lo,
                    "games": g, "mean_bank": round(bank / g, 1), "mean_opp": round(opp_bank / g, 1)}
                agg_w += w; agg_d += d; agg_l += lo; agg_bank += bank; agg_opp += opp_bank
                print(f"  {label:<12} vs {onm:<34} r{ort:.0f}: "
                      f"{score:.3f} ({w}-{d}-{lo})  bank {bank/g:.0f} / {opp_bank/g:.0f}")
            G = max(1, agg_w + agg_d + agg_l)
            tot = {"score": round((agg_w + 0.5 * agg_d) / G, 4), "w": agg_w, "d": agg_d,
                   "l": agg_l, "games": G, "mean_bank": round(agg_bank / G, 1),
                   "mean_opp": round(agg_opp / G, 1)}
            report["candidates"][label]["totals"] = tot
            print(f"  --> {label}: PANEL SCORE {tot['score']:.4f}  "
                  f"({agg_w}-{agg_d}-{agg_l})  mean bank {tot['mean_bank']:.0f} "
                  f"vs {tot['mean_opp']:.0f}\n")
        report["wall_seconds"] = round(time.time() - t_start, 1)
    finally:
        srv.close()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ranked summary
    print("=" * 64)
    print("PANEL SCOREBOARD (higher = better; 0.5 = even):")
    ranked = sorted(((lab, c["totals"].get("score", 0.0))
                     for lab, c in report["candidates"].items() if c.get("verify_ok")),
                    key=lambda kv: -kv[1])
    for lab, sc in ranked:
        t = report["candidates"][lab]["totals"]
        print(f"  {lab:<16} {sc:.4f}  ({t['w']}-{t['d']}-{t['l']})  "
              f"bank {t['mean_bank']:.0f} vs {t['mean_opp']:.0f}")
    print(f"report -> {a.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
