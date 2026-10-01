"""Verify the config LEVERS actually move the agent -- switch dispatch / each
rail / the sweep threshold on and off in combination and measure win+margin vs
the killer family. After the 2026-09-20 fixes (dispatch branch keys re-generated
at the real dispatch step; write_stage propagates `tables`; KAGG_LOG logging in
the isolated binary) each lever should now FIRE; this quantifies whether it also
helps.

Emits ONE table. Also, per combo, reports how many checkpoints dispatch HIT vs
tschinkel (from KAGG_LOG=dispatch) -- a lever that never fires is flagged.

Run (one heavy job at a time):
  NN_WORLDS=8 python -m kaggriculture.bandit.gate.verify_levers
"""
import os, json, subprocess
import multiprocessing as mp
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
import kaggriculture.measure.eval_harness as EH

ALL_CKPTS = [[146, 6], [290, 12], [362, 15], [506, 21], [650, 27]]
RAILS = ["weed_repair", "cash_guard", "escalation", "endgame", "anti_dump", "front_run"]


def cfg(checkpoints=None, off=(), tables=None, d6_optional=False):
    """Base config with a chosen checkpoint set, rails turned OFF by name, and an
    optional `tables` override (e.g. {'sweep':15} to un-starve the sweep)."""
    c = H.base_config()
    c["checkpoints"] = [list(x) for x in (ALL_CKPTS if checkpoints is None else checkpoints)]
    if d6_optional:
        c["d6_optional"] = True
    for g in c["guardrails"]:
        if g.get("name") in off:
            g["on"] = False
    if tables:
        c["tables"] = tables
    return c


# name -> (config, one-line description)
COMBOS = {
    "baseline_all":      (cfg(),                                   "all rails ON, dispatch ON (5 ckpts)"),
    "no_dispatch":       (cfg(checkpoints=[], d6_optional=True),   "pure base tape (no ckpts)"),
    "dispatch_only":     (cfg(off=("anti_dump", "front_run", "escalation", "cash_guard", "weed_repair")),
                                                                   "dispatch ON, all optional rails OFF"),
    "no_front_run":      (cfg(off=("front_run",)),                 "all but front_run"),
    "no_anti_dump":      (cfg(off=("anti_dump",)),                 "all but anti_dump"),
    "no_escalation":     (cfg(off=("escalation",)),                "all but escalation"),
    "sweep15":           (cfg(tables={"sweep": 15, "esc_sweep": 10}),
                                                                   "all rails + sweep un-starved (thr 15)"),
    "sweep15_no_antidump": (cfg(off=("anti_dump",), tables={"sweep": 15, "esc_sweep": 10}),
                                                                   "sweep un-starved, anti_dump OFF"),
    "minimal":           (cfg(checkpoints=[], d6_optional=True,
                              off=("anti_dump", "front_run", "escalation", "cash_guard", "weed_repair")),
                                                                   "pure tape + only mandatory endgame"),
}


def public_panel():
    """Banded public-agent panel (reactive .py agents from the crown panel +
    the killer family). NN_FULLPANEL=1 adds the extra mids. Non-loading entries
    are skipped so a bad payload never aborts the sweep."""
    R = os.path.join(ROOT, ".local", "crown_panel", "refs")
    cand = [
        ("tschinkel_2945",  os.path.join(ROOT, "agents", "pub_tschinkel_2945.py")),
        ("tetsutani_2500",  os.path.join(R, "2500-2700", "pub_tetsutani_mirror.py")),
        ("nathanjacob_2500", os.path.join(R, "2500-2700", "pub_nathanjacob_anticlone.py")),
        ("metacounter_2500", os.path.join(R, "2500-2700", "kaggriculture-metacounter-r1-scored-agent.py")),
        ("frontier_2905",   os.path.join(R, "2700plus", "tape27_2905_108026392_s1.py")),
        ("moon_2700",       os.path.join(R, "2700plus", "kaggriculture-adaptive-public-state-multi-route___MOON_PAYLOAD.py")),
        ("k0006_2494",      os.path.join(R, "2300-2500", "ref_k0006_2494.py")),
        ("mid_2490",        os.path.join(R, "2300-2500", "mid_2490_106296373.py")),
        ("prvsiyan_13tape", os.path.join(R, "2100-2300", "pub_prvsiyan_yhay81_13tape.py")),
    ]
    if os.environ.get("NN_FULLPANEL"):
        cand += [
            ("alperen_rhythm", os.path.join(ROOT, "agents", "pub_alperen_rhythm.py")),
            ("mid_2414",       os.path.join(R, "2300-2500", "mid_2414_106901648.py")),
            ("adaptiveshop_2500", os.path.join(R, "2500-2700", "kaggriculture-adaptive-shop-guard__BLOB.py")),
        ]
    return [(n, p) for n, p in cand if os.path.exists(p)]


def _cell(task):
    """One (combo, opponent) cell -> (combo, opp, win, margin). Runs in a worker
    process; builds the staged bandit agent + the opponent inside the worker and
    plays both seats over the seed set on the Rust serve engine."""
    combo, stage, an, ap, ws = task
    try:
        res = H.eval_vs(EH.bandit_binary_agent(stage), ap, ws)
        return (combo, an, res["win"], res["margin"], None)
    except Exception as e:
        return (combo, an, None, None, f"{type(e).__name__}: {str(e)[:60]}")


def main():
    n = int(os.environ.get("NN_WORLDS", "6"))
    workers = int(os.environ.get("NN_WORKERS", "5"))  # conservative: box is memory-constrained
    ws = H.world_seeds(n)
    panel = public_panel()
    print(f"VERIFY LEVERS vs PUBLIC PANEL  agents={len(panel)}  worlds={len(ws)} (x2 seats)  workers={workers}", flush=True)
    print(f"panel: {[nm for nm,_ in panel]}", flush=True)
    print("(Rust serve engine; parallel across workers)\n", flush=True)
    # build all combo stages once (fast: stage only, no recompile), then fan out
    stage_of, desc_of = {}, {}
    for name, (c, desc) in COMBOS.items():
        H.build_agent(f"vl_{name}", c)
        stage_of[name] = os.path.join(H.SCRATCH, f"vl_{name}_wingate")
        desc_of[name] = desc
    tasks = [(name, stage_of[name], an, ap, ws) for name in COMBOS for an, ap in panel]
    print(f"{len(tasks)} cells ({len(COMBOS)} combos x {len(panel)} agents), "
          f"{len(tasks)*len(ws)*2} games total\n", flush=True)
    per = {name: {} for name in COMBOS}
    done = 0
    with mp.Pool(workers, maxtasksperchild=4) as pool:
        for combo, an, win, margin, err in pool.imap_unordered(_cell, tasks):
            done += 1
            if err:
                print(f"  [{done}/{len(tasks)}] ! {combo} vs {an}: {err}", flush=True)
                continue
            per[combo][an] = {"win": win, "margin": margin}
            print(f"  [{done}/{len(tasks)}] {combo:20s} vs {an:18s} win={win:.2f} margin={margin:+8.0f}", flush=True)

    rows = []
    for name in COMBOS:
        p = per[name]
        if not p:
            continue
        mw = float(np.mean([v["win"] for v in p.values()]))
        mm = float(np.mean([v["margin"] for v in p.values()]))
        beat = sum(1 for v in p.values() if v["win"] > 0.5)
        rows.append((name, mw, mm, beat, len(p), p, desc_of[name]))
    print("\n=== RANKED by win, then margin (vs public panel) ===", flush=True)
    rows.sort(key=lambda r: (-r[1], -r[2]))
    for name, mw, mm, beat, npan, p, desc in rows:
        print(f"  {name:20s} win={mw:.2f} margin={mm:+8.0f} beat={beat}/{npan}  [{desc}]", flush=True)
    if rows:
        best = rows[0]
        print(f"\n=== OPTIMAL = {best[0]}  (per-agent win/margin) ===", flush=True)
        for an, v in sorted(best[5].items(), key=lambda kv: -kv[1]["win"]):
            print(f"  {an:20s} win={v['win']:.2f} margin={v['margin']:+8.0f}", flush=True)


if __name__ == "__main__":
    main()
