"""House rule: every agent ships with a knowledge graph in agents/.

Track P's graph emitter (fresh code): PARAMS grouped and annotated, the
L2/L1/L0 decision path, and the current measurement record pulled from the
models/trackp reports. Self-contained HTML, no external assets.

Usage: python src/trackp/graph.py [--agent agents/planner_v0.py]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import html
import json
import os
import re

try:
    from . import common
except ImportError:
    import sys
    from kaggriculture.trackp import common

GROUPS = {
    "labour": ["labour_base", "labour_max", "labour_ramp_day",
               "labour_per_tiles"],
    "planting": ["plants_per_unit", "plant_budget_turn",
                 "plant_value_margin", "wheat_floor",
                 "last_plant_margin_days"],
    "investment": ["idle_cash_target", "invest_land_day_ne",
                   "invest_land_day_sw", "invest_land_day_se", "max_coops",
                   "max_pastures", "animal_start_day", "animal_last_day"],
    "selling": ["sell_horizon_days", "sell_hold_gain", "sell_chunk",
                "shed_pressure", "terminal_day", "fert_sell_min"],
    "discipline": ["water_growth_bonus", "care_value"],
    "projection": ["proj_blend_prior", "macro_shop_react"],
}


def _report(name):
    p = os.path.join(common.MODELS, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def emit(agent_path: str) -> str:
    src = open(agent_path, encoding="utf-8").read()
    m = re.search(r"# --- PARAMS BEGIN ---\n(.*?)# --- PARAMS END ---",
                  src, re.S)
    ns: dict = {}
    exec(m.group(1), {}, ns)  # noqa: S102 -- our own build
    params = ns["PARAMS"]
    l1 = "MLP (exported, equivalence-asserted)" \
        if "L1_WEIGHTS = None" not in src else "hand rules + elite priors"
    prior_rows = src.count("(", src.index("FAMILY_PRIOR")) \
        if "FAMILY_PRIOR = []" not in src else 0

    reports = {
        "projector": _report("projector_report.json"),
        "search": _report("params_searched.json"),
        "iql": _report("iql_report.json"),
        "sim2real": _report("sim2real.json"),
        "panel": _report("panel_report.json"),
        "validity": _report("validity_report.json"),
        "graduation": _report("graduation_report.json"),
    }

    rows = []
    for group, keys in GROUPS.items():
        for k in keys:
            if k in params:
                rows.append(f"<tr><td>{group}</td><td><code>{k}</code></td>"
                            f"<td>{html.escape(str(params[k]))}</td></tr>")

    def _j(x):
        return html.escape(json.dumps(x, indent=1)) if x else "&mdash;"

    name = os.path.splitext(os.path.basename(agent_path))[0]
    doc = f"""<!doctype html><meta charset="utf-8">
<title>{name} — Track P knowledge graph</title>
<style>
 body{{font:15px/1.5 Georgia,serif;max-width:880px;margin:24px auto;
      padding:0 16px;background:#f7f6f2;color:#26241f}}
 h1{{font-size:1.5rem;border-bottom:3px solid #1a6e50;padding-bottom:6px}}
 h2{{font-size:1.1rem;margin-top:28px}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem}}
 td,th{{border-bottom:1px solid #ddd;padding:5px 9px;text-align:left;
       vertical-align:top}}
 code{{background:#edece6;padding:1px 4px;border-radius:3px}}
 pre{{background:#edece6;padding:10px;overflow-x:auto;font-size:.8rem}}
</style>
<h1>{name}</h1>
<p><strong>Track P planner</strong> — L2 price projection (quote curve +
observed shop-draw drain + opponent dump residual, {prior_rows} prior rows)
→ L1 macro policy ({l1}) → L0 dollar-priced job executor.
Fail-soft: a wrong macro decision only re-weights job pricing.</p>
<h2>PARAMS (searched by CMA-ES on the Rust league)</h2>
<table><tr><th>group</th><th>param</th><th>value</th></tr>
{''.join(rows)}</table>
<h2>Decision path</h2>
<ol>
<li><strong>Every turn</strong>: recover opponent dumps from public market
inventory deltas (exact arithmetic, no identifier needed).</li>
<li><strong>Day boundary / shop unlock</strong>: L1 re-plans — production
mix over 8 products, labour target, investment depth, sell pacing.</li>
<li><strong>Every turn</strong>: L0 prices each candidate job in projected
dollars (urgent watering +$10k — a plant unwatered twice is a weed),
assigns units greedily, emits ≤10 priority-ordered market orders.</li>
<li><strong>Day ≥ terminal_day</strong>: harvest-drop-sell sweep — unsold
shed at end correlates −0.14 with winning, field-wide.</li>
</ol>
<h2>Measurement record</h2>
<h3>Price projector (must beat the "current price" naive)</h3>
<pre>{_j({k: v for k, v in (reports['projector'] or {}).items()
           if k != 'per_product'})}</pre>
<h3>CMA-ES best</h3>
<pre>{_j((reports['search'] or {}).get('best'))}</pre>
<h3>IQL warm start (held-out day accuracy)</h3>
<pre>{_j(reports['iql'])}</pre>
<h3>Sim-to-real</h3>
<pre>{_j(reports['sim2real'])}</pre>
<h3>P1.6 regime panel</h3>
<pre>{_j({k: v for k, v in (reports['panel'] or {}).items()
           if k in ('strata', 'pooled_score', 'sign_p', 'games')})}</pre>
<h3>P4.1 open↔closed validity</h3>
<pre>{_j(reports['validity'])}</pre>
<h3>Graduation gate (report-only)</h3>
<pre>{_j({k: v for k, v in (reports['graduation'] or {}).items()
           if k not in ('planner', 'incumbent')})}</pre>
"""
    out = os.path.splitext(agent_path)[0] + ".html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("graph:", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(
        common.ROOT, "agents", "planner_v0.py"))
    a = ap.parse_args()
    emit(a.agent)
