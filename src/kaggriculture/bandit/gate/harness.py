"""Shared gate/sweep harness: config injection, world-diverse seeds, build + eval.

Rail parameters are config knobs read at RUNTIME, so tuning is CONFIG-ONLY (no
recompile) once a rail exists in the isolated binary. This module centralises the
machinery both rail_sweep and world_gate use. Build artifacts (stage dirs, config
JSONs) are written under .local/scratch (regenerable, gitignored); this code is
tracked source.
"""
import os, json, shutil, copy
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.measure.eval_harness as EH
import kaggriculture.bandit.build.build_rust_bandit as B
import kaggriculture.pipeline.daily_slot2 as D

SCRATCH = os.path.join(ROOT, ".local", "scratch", "bandit_gate")
os.makedirs(SCRATCH, exist_ok=True)
# KAGG_BIN overrides (Linux / ARM boxes build kagg natively: no .exe). Same env var as serve_match.
FRESH_KAGG = os.environ.get("KAGG_BIN") or os.path.join(
    ROOT, "rustengine", ".local", "scratch", "bandit_target", "release", "kagg.exe" if os.name == "nt" else "kagg")


def base_config():
    return json.load(open(os.path.join(ROOT, "configs", "bandit_config.json")))


def joint_net():
    return json.load(open(os.path.join(ROOT, ".local", "nn", "joint_nn.json")))


def killers():
    cand = [
        ("tschinkel",  os.path.join(ROOT, "agents", "pub_tschinkel_2945.py")),
        ("tetsutani",  os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_tetsutani_mirror.py")),
        ("nathanjacob", os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_nathanjacob_anticlone.py")),
        # public high-scorers added to the shared gate 2026-09-20 (distinct from
        # the above by content hash; each validated to load + play a full 719-step
        # serve game). See [[trackp-public-agents-2026-09-20]].
        ("anhad_metav4v13",   os.path.join(ROOT, "agents", "pub_anhadmahajan_metav4v13.py")),
        ("dmitrii_mktshock",  os.path.join(ROOT, "agents", "pub_dmitriigluzdov_market_shock.py")),
        ("nathanjacob_pipe16", os.path.join(ROOT, "agents", "pub_nathanjacob_pipe16.py")),
        ("haodou_db6965",     os.path.join(ROOT, "agents", "pub_haodou_db6965.py")),
    ]
    return [(n, p) for n, p in cand if os.path.exists(p)]


# ---- rail builders: knob values are the tunable magic numbers -------------------
def price_sell(frm=145, look=12, floor=0.95):
    return {"name": "price_sell", "on": True, "ps_from": frm, "ps_look": look, "ps_floor": floor}

def supply_cap(floor=0.9, carry_max=0, bound_future=0):
    return {"name": "supply_cap", "on": True, "floor": floor,
            "carry_max": carry_max, "bound_future": bound_future}

def budget(block=72):
    return {"name": "budget_guard", "on": True, "block": block}

def opp_fr(thresh=0.6, floor=0.85, net=None, premium=None):
    d = {"name": "opp_front_run", "on": True, "opp_thresh": thresh, "opp_floor": floor,
         "nn": net if net is not None else joint_net()}
    if premium is not None:
        d["premium"] = premium
    return d

def endgame_liq(frm=648, floor=0.6, hard_day=29):
    return {"name": "endgame_liquidate", "on": True, "el_from": frm, "el_floor": floor, "el_hard_day": hard_day}

def mlp_sell(frm=145, until=640, floor=0.2, net=None):
    return {"name": "mlp_sell", "on": True, "nn_from": frm, "nn_until": until, "nn_floor": floor,
            "nn": net if net is not None else joint_net()}


def cfg_with(extra):
    c = base_config(); c["guardrails"] = copy.deepcopy(c["guardrails"]) + copy.deepcopy(extra)
    return c


def build_agent(name, cfg):
    """Build an isolated-binary bandit agent for a config (scratch under .local)."""
    assert os.path.exists(FRESH_KAGG), f"build the isolated binary first: {FRESH_KAGG}"
    cp = os.path.join(SCRATCH, f"{name}.json"); json.dump(cfg, open(cp, "w"))
    out = os.path.join(SCRATCH, name); shutil.rmtree(out, ignore_errors=True)
    # also clear the _wingate stage: _windows_bandit_gate_stage copytree's into it
    # and raises FileExistsError if a prior build (or a crashed run) left it behind.
    shutil.rmtree(out + "_wingate", ignore_errors=True)
    B.write_stage(out, cp); D._windows_bandit_gate_stage(out)
    stage = out + "_wingate"; shutil.copy2(FRESH_KAGG, os.path.join(stage, "kagg.exe"))
    return EH.bandit_binary_agent(stage)


def world_seeds(n=12):
    """One representative seed per DISTINCT shop-world, labeled via serve GENGAME
    (unlocked_shops at day 0). Returns [(world_name, seed)]. The label is stable
    across configs that share the opening (rails act past the shop-split)."""
    import kaggriculture.engine.serve_match as SM
    P = chr(31).join(SM.action_to_line(None) for _ in range(719))
    bank = json.load(open(os.path.join(ROOT, "data", "worlds", "seed_bank.json"), encoding="utf-8"))
    srv = SM.Serve(); out = []; seen = set()
    try:
        for s in bank.get("seeds", []):
            js = srv.cmd("GENGAME " + str(int(s)) + chr(30) + P + chr(30) + P)
            days = js.get("days") or []
            sh = ((days[0].get("town") or {}).get("unlocked_shops") or []) if days else []
            w = f"{sh[0]}|{sh[1]}" if len(sh) >= 2 else None
            if w and w not in seen:
                seen.add(w); out.append((w, int(s)))
            if len(out) >= n:
                break
    finally:
        srv.close()
    return out


def eval_vs(agent, ref, seeds):
    """Both-seats win/margin per world. `seeds` = [(world, seed)] or [seed]."""
    A = EH._as_agent(agent); R = EH._as_agent(ref)
    per_world = {}
    for item in seeds:
        wname, s = item if isinstance(item, tuple) else (str(item), item)
        b0, b1 = EH.play_two(A, R, s)
        w0 = 1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0)
        b0b, b1b = EH.play_two(R, A, s)
        w1 = 1.0 if b1b > b0b else (0.5 if b1b == b0b else 0.0)
        per_world[wname] = (0.5 * (w0 + w1), 0.5 * ((b0 - b1) + (b1b - b0b)))
    win = float(np.mean([w for w, _ in per_world.values()]))
    margin = float(np.mean([m for _, m in per_world.values()]))
    return dict(win=win, margin=margin, per_world=per_world)
