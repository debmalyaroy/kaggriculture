"""Per-model knowledge graph: agents/<model>.html next to agents/<model>.py

Every agent gets a graph carrying three things:

1. **Configuration** — the model's PARAMS, grouped and annotated.
2. **Decision path** — which branch this model took at each strategic fork, and
   what the alternatives measured when they were tried.
3. **Game-state graph** — a real episode, sampled, showing at each state the
   action taken *and why*: the agent prices every candidate job in dollars, so
   the runners-up and their valuations are recoverable. That is the "why", not
   a narrative reconstruction.

    python -m kaggriculture.agentbuild.model_graph agents/agent_v3_20260804_160019.py
    python -m kaggriculture.agentbuild.model_graph agents/v2_tuned.py --every 72 --opponent starter
"""
from kaggriculture.paths import ROOT
import argparse
import html
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402

GROUPS = [
    ("Labour", ["hands_max", "hands_min", "hands_late_day", "hands_max_late",
                "hire_cash_frac", "hire_cash_floor"]),
    ("Capacity model", ["capacity_util", "cost_per_crop_day", "cost_per_animal_day"]),
    ("Spatial efficiency", ["travel_weight", "poach_penalty"]),
    ("Cash & feed", ["reserve_base", "reserve_per_tile", "feed_runway_days",
                     "wheat_tile_yield", "animal_cash_buffer", "animals_per_turn",
                     "feed_buffer_days", "feed_buy_price_cap", "feed_self_frac"]),
    ("Land", ["land_reserve", "land_last_day", "land_min_day", "land_min_used"]),
    ("Portfolio", ["target_wheat", "target_carrot", "target_tomato",
                   "target_strawberry", "target_melon", "target_goose",
                   "target_cow", "target_sheep", "max_herd", "filler_crop",
                   "animal_min_extra_days"]),
    ("Market", ["sell_floor", "sell_chunk", "dump_day", "shed_pressure",
                "seed_lookahead", "liquid_target", "liquid_weight"]),
    ("Task scoring", ["care_weight", "fert_collect_weight", "feed_safe_discount",
                      "fert_weight"]),
]

WHY = {
    "WATER": "keeps the plant alive (2 missed days = weed) or adds a bonus unit inside the yield window",
    "HARVEST": "collects units now worth market price; frees a one-time crop's tile",
    "PLANT": "commits an empty tile to the best $/tile-day crop that still has time to mature",
    "FEED": "unfed twice and the animal is gone permanently — the single most valuable job on the farm",
    "CARE": "banks +1 unit onto the animal's next scheduled production",
    "FERTILIZE": "doubles the per-day bonus for 3 days; on ongoing crops it doubles every production",
    "COLLECT_FERTILIZER": "picks up the free fertilizer each animal makes daily",
    "BUILD_COOP": "prepares a tile for a goose already owned or affordable",
    "BUILD_PASTURE": "prepares a tile for a cow or sheep",
    "DIG": "clears a weed or a dead pen so the tile can earn again",
    "PICKUP": "fetches wheat for feeding, or an animal to place",
    "PLACE": "puts an owned animal onto its matching structure",
    "DROP": "moves produce into the shed — SELL only draws from the shed",
    "PASS": "no job cleared the value of walking to it",
    "NORTH": "walking toward the highest value-per-turn job",
    "SOUTH": "walking toward the highest value-per-turn job",
    "EAST": "walking toward the highest value-per-turn job",
    "WEST": "walking toward the highest value-per-turn job",
}


def load_mod(path):
    spec = importlib.util.spec_from_file_location("mdl", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(mod, obs, cfg):
    """Recompute the agent's own task valuations for this state."""
    P = mod.PARAMS
    p = obs["player"]
    farm = obs["farms"][p]
    tiles = farm["tiles"]
    n = len(tiles)
    day, hour = int(obs["day"]), int(obs["hour"])
    total_days = 30
    try:
        total_days = max(1, int(cfg["episodeSteps"]) // int(cfg["turnsPerDay"]))
    except Exception:                                              # noqa: BLE001
        pass
    days_left = max(0, total_days - day)
    shed = dict(obs["private"]["shed"])
    seeds = dict(obs["private"]["seeds"])
    c, empty = mod.census(tiles, n)
    money = float(farm["money"])
    plan = mod.plan_tiles(obs, P, c, empty, days_left, money, shed, seeds, n)
    capital = money - mod.working_reserve(P, c, shed, mod._price(obs, "WHEAT")) \
        - P["animal_cash_buffer"]
    surplus = {}
    for st in ("COOP", "PASTURE"):
        inc = sum(shed.get(a, 0) for a in mod.ANIMALS if mod.ANIMALS[a]["structure"] == st)
        cheap = min(mod.ANIMALS[a]["cost"] for a in mod.ANIMALS
                    if mod.ANIMALS[a]["structure"] == st)
        aff = int(max(0.0, capital) // cheap)
        planned = sum(1 for v in plan.values() if v[0] == st)
        surplus[st] = max(0, c[st] - inc - aff - planned)
    tasks = mod.build_tasks(obs, P, tiles, n, day, days_left, c, plan, shed, surplus)
    tasks = sorted(tasks, key=lambda t: -t["value"])
    return {
        "day": day, "hour": hour, "money": money,
        "herd": c["GOOSE"] + c["COW"] + c["SHEEP"],
        "crops": sum(c[k] for k in mod.CROPS),
        "pens": c["COOP"] + c["PASTURE"], "weeds": c["WEED"],
        "quads": len(farm["unlocked_quadrants"]), "hands": len(farm["hands"]),
        "shed": {k: v for k, v in shed.items() if v},
        "prices": dict(obs["market"]["prices"]),
        "tasks": [{"op": " ".join(str(x) for x in t["op"]),
                   "pos": list(t["pos"]), "value": round(t["value"], 1),
                   "need": t["need"]} for t in tasks[:7]],
    }


def route_snapshot(mod, obs, act):
    """Frame for a tape/bandit agent, which has no PARAMS valuation.

    The honest equivalent of "its own arithmetic": the schedule the route
    carries for this step, what the repair stack changed (relay pulls, weed
    repairs, clamps), and each SELL quoted in dollars against the live market
    -- the same quote the impact ranking and the relay use.
    """
    p = obs["player"]
    farm = obs["farms"][p]
    step = int(obs.get("step", 0))
    route = getattr(mod, "_ROUTE", None) or []
    sched = route[min(step, len(route) - 1)] if route else {}
    inv = dict((obs.get("market") or {}).get("inventory", {}) or {})
    quote = getattr(mod, "_quote", None)
    known = getattr(mod, "_MARKET_PARAMS", {})
    tasks = []
    for o in (act.get("market") or []):
        if not (isinstance(o, list) and len(o) >= 2):
            continue
        qty = int(o[2]) if len(o) > 2 and str(o[2]).lstrip("-").isdigit() else 1
        val = 0.0
        if o[0] == "SELL" and quote and o[1] in known:
            val = round(qty * quote(o[1], int(inv.get(o[1], 0) or 0)), 1)
        tasks.append({"op": " ".join(str(x) for x in o),
                      "pos": [], "value": val, "need": ""})
    s_mk = {tuple(x) for x in (sched.get("market") or []) if isinstance(x, list)}
    a_mk = {tuple(x) for x in (act.get("market") or []) if isinstance(x, list)}
    delta = []
    for x in sorted(a_mk - s_mk):
        delta.append({"op": "STACK+ " + " ".join(str(v) for v in x),
                      "pos": [], "value": 0.0, "need": "added by repair/relay"})
    for x in sorted(s_mk - a_mk):
        delta.append({"op": "STACK- " + " ".join(str(v) for v in x),
                      "pos": [], "value": 0.0, "need": "held/repaid/clamped"})
    crops = herd = pens = weeds = 0
    for row in (farm.get("tiles") or []):
        for tile in row:
            if isinstance(tile, dict):
                if tile.get("crop"):
                    crops += 1
                if tile.get("animal"):
                    herd += 1
                kind = tile.get("kind")
                if kind in ("PASTURE", "COOP"):
                    pens += 1
                elif kind == "WEED":
                    weeds += 1
    shed = dict(((obs.get("private") or {}).get("shed", {})) or {})
    return {
        "day": int(obs.get("day", step // 24)),
        "hour": int(obs.get("hour", step % 24)),
        "money": float(farm.get("money", 0) or 0), "crops": crops,
        "herd": herd, "pens": pens, "weeds": weeds,
        "hands": len(farm.get("hands") or []),
        "quads": len(farm.get("unlocked_quadrants") or []),
        "shed": {k: v for k, v in shed.items() if v},
        "prices": dict((obs.get("market") or {}).get("prices", {}) or {}),
        "tasks": (tasks + delta)[:9],
    }


def run(agent_path, opponent, seed, every):
    from kaggle_environments import make
    mod = load_mod(agent_path)
    cfg = {"episodeSteps": 720, "seed": seed, "actTimeout": 60, "runTimeout": 100000}
    frames = []
    two = mod.agent.__code__.co_argcount > 1
    # Tape/bandit agents carry no PARAMS valuation -- use the schedule frame
    # instead of erroring "module 'mdl' has no attribute 'PARAMS'" per state.

    def wrapped(obs, config=None):
        act = mod.agent(obs, config) if two else mod.agent(obs)
        step = int(obs.get("step", 0))
        if step % every == 0:
            try:
                if hasattr(mod, "PARAMS"):
                    snap = snapshot(mod, obs, config or cfg)
                else:
                    snap = route_snapshot(mod, obs, act)
                snap["action"] = act
                frames.append(snap)
            except Exception as exc:                               # noqa: BLE001
                frames.append({"day": int(obs["day"]), "hour": int(obs["hour"]),
                               "error": str(exc), "action": act,
                               "money": obs["farms"][obs["player"]]["money"]})
        return act

    env = make("kaggriculture", configuration=cfg, debug=False)
    env.run([wrapped, opponent])
    final = env.steps[-1]
    return frames, float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def fmt_params(P):
    seen = set()
    out = []
    for title, keys in GROUPS:
        items = [(k, P[k]) for k in keys if k in P]
        seen |= {k for k, _ in items}
        if items:
            out.append((title, items))
    rest = [(k, v) for k, v in P.items() if k not in seen]
    if rest:
        out.append(("Other", rest))
    return out


def render(agent_path, P, frames, mine, theirs, opponent, seed, out_path):
    name = os.path.basename(agent_path)
    E = html.escape
    S = []
    A = S.append

    A(f"<h2>1 · Configuration</h2><div class='cols'>")
    for title, items in fmt_params(P):
        A(f"<div class='card'><h4>{E(title)}</h4><table class='kv'>")
        for k, v in items:
            vs = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            if len(vs) > 62:
                vs = vs[:59] + "…"
            A(f"<tr><td><code>{E(k)}</code></td><td class='n'>{E(vs)}</td></tr>")
        A("</table></div>")
    A("</div>")

    A("<h2>2 · Decision path</h2>")
    A("<p class='dim'>Each fork this model faced, the branch it took, and what "
      "the alternatives measured when they were tested in self-play.</p>")
    forks = [
        ("Task selection", f"value ÷ (1 + travel_weight×distance), "
         f"travel_weight = {P.get('travel_weight', 1.0)}",
         [("travel_weight 1.0 (no distance emphasis)", "baseline v2"),
          ("travel_weight 8.0", "0% win rate — over-corrects")]),
        ("Herd size", f"max_herd = {P.get('max_herd')}",
         [("max_herd 22 / 34", "score 42,864 vs 73,272 — bigger herds starve")]),
        ("Livestock mix", f"cow {P.get('target_cow')} · sheep {P.get('target_sheep')} "
         f"· goose {P.get('target_goose')}",
         [("cow-heavy (26)", "rejected — cows pay from day 8, geese from day 4"),
          ("no geese", "rejected — eggs are the only glut-proof premium good")]),
        ("Fertilizer", f"fert_weight = {P.get('fert_weight', 1.0)}",
         [("fert_weight 0 (ignore it)", "leaves free livestock fertilizer unused"),
          ("fert_weight 5.0", "+$1,233 vs +$5,784 at 2.5")]),
        ("Crop portfolio", f"melon {P.get('target_melon')} · "
         f"strawberry {P.get('target_strawberry')}",
         [("strawberry 32 / melon 6", "0% win rate, −$10,056 — strawberry is dear "
                                      "because it is slow, not because there is an opening")]),
        ("Endgame", f"dump from day {P.get('dump_day')}, haul outranks all on the last day",
         [("no endgame haul", "−$25k/match — SELL draws from the shed only")]),
    ]
    A("<table><tr><th>fork</th><th>chosen</th><th>alternatives and what they measured</th></tr>")
    for fork, chosen, alts in forks:
        alt = "<br>".join(f"<span class='alt'>{E(a)}</span> — {E(m)}" for a, m in alts)
        A(f"<tr><td><b>{E(fork)}</b></td><td class='ok'>{E(chosen)}</td><td>{alt}</td></tr>")
    A("</table>")

    A("<h2>3 · Game-state graph</h2>")
    A(f"<p class='dim'>One episode vs <code>{E(opponent)}</code>, seed {seed} — "
      f"final <b>${mine:,.0f}</b> vs ${theirs:,.0f}. Each state shows the board, "
      f"the action taken, and the agent's own dollar valuation of every candidate "
      f"job — including the ones it turned down.</p>")
    for fr in frames:
        if "error" in fr:
            A(f"<div class='state'><div class='sh'>Day {fr['day']} · hour {fr['hour']}"
              f"</div><p class='dim'>valuation unavailable: {E(fr['error'])}</p></div>")
            continue
        act = fr.get("action") or {}
        farmer = act.get("farmer") or ["PASS"]
        fop = farmer[0]
        chosen = " ".join(str(x) for x in farmer)
        mk = act.get("market") or []
        A("<div class='state'>")
        A(f"<div class='sh'>Day {fr['day']} · hour {fr['hour']}</div>")
        A("<div class='chips'>"
          f"<span class='chip'>bank ${fr['money']:,.0f}</span>"
          f"<span class='chip'>crops {fr['crops']}</span>"
          f"<span class='chip'>herd {fr['herd']}</span>"
          f"<span class='chip'>pens {fr['pens']}</span>"
          f"<span class='chip'>weeds {fr['weeds']}</span>"
          f"<span class='chip'>quadrants {fr['quads']}</span>"
          f"<span class='chip'>hands {fr['hands']}</span></div>")
        A(f"<div class='act'><b>farmer →</b> <code>{E(chosen)}</code>"
          f"<span class='why'>{E(WHY.get(fop, ''))}</span></div>")
        if mk:
            A("<div class='act'><b>market →</b> " +
              " ".join(f"<code>{E(' '.join(str(x) for x in o))}</code>" for o in mk[:6]) +
              "</div>")
        if fr["tasks"]:
            A("<table class='tasks'><tr><th>candidate job</th><th>tile</th>"
              "<th class='n'>value</th><th>needs</th></tr>")
            for i, t in enumerate(fr["tasks"]):
                cls = "top" if i == 0 else ""
                pos = (f"{t['pos'][0]},{t['pos'][1]}"
                       if len(t.get("pos") or []) >= 2 else "—")
                A(f"<tr class='{cls}'><td><code>{E(t['op'])}</code></td>"
                  f"<td class='n'>{pos}</td>"
                  f"<td class='n'>${t['value']:,.0f}</td>"
                  f"<td>{E(str(t['need']) if t['need'] else '—')}</td></tr>")
            A("</table>")
            A("<p class='dim sm'>Highest-value job is not always chosen: the agent "
              "divides by travel distance, so a nearer, cheaper job can win — that "
              "is the spatial-efficiency change.</p>")
        A("</div>")

    css = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#30363d;--ink:#e6edf3;
--dim:#9aa7b4;--faint:#6e7b8a;--ok:#5fbf8f;--alt:#d9a441}
@media(prefers-color-scheme:light){:root{--bg:#f7f9fc;--panel:#fff;--panel2:#eef2f7;
--line:#d3dae3;--ink:#111820;--dim:#4c5866;--faint:#7b8794}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:30px 20px 70px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 8px;padding-bottom:7px;
border-bottom:1px solid var(--line)}h4{margin:0 0 7px;font-size:12px;color:var(--dim);
text-transform:uppercase;letter-spacing:.08em}
.dim{color:var(--dim)}.sm{font-size:12px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.kv td{border:0;padding:3px 6px}
code{background:rgba(127,127,127,.16);padding:1px 5px;border-radius:4px;font-size:12px}
.ok{color:var(--ok)}.alt{color:var(--alt)}
.state{background:var(--panel);border:1px solid var(--line);border-left:3px solid #4c9be8;
border-radius:9px;padding:12px 14px;margin:14px 0}
.sh{font-weight:600;font-size:14px;margin-bottom:7px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:20px;
padding:2px 10px;font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
.act{margin:5px 0;font-size:13px}
.why{color:var(--faint);font-size:12px;margin-left:9px}
table.tasks tr.top td{background:rgba(95,191,143,.10)}
footer{margin-top:44px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{E(name)} — knowledge graph</title><style>{css}</style></head>"
           f"<body><div class='wrap'><h1>{E(name)}</h1>"
           f"<p class='dim'>Model knowledge graph — configuration, decision path, "
           f"and a state-by-state trace of why each action was taken.</p>"
           + "".join(S) +
           f"<footer>Generated by src/kaggriculture/agentbuild/model_graph.py · regenerate whenever the "
           f"model changes.</footer></div></body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def _route_params(agent):
    """Describe a route agent, which has no PARAMS block to read.

    Its configuration *is* the route, so summarise the route the way a
    parameter table summarises a policy: what it builds, when it first sells
    each product, and how much of each thing it moves across the season.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("routeagent", agent)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"could not load: {exc}"}
    route = getattr(mod, "_ROUTE", None) or getattr(mod, "_TRACE", None)
    if not route:
        return {"architecture": "unknown -- no PARAMS block and no route"}

    unit_ops, market, sell_days, first_sale = {}, {}, set(), {}
    hires = land = 0
    for t, turn in enumerate(route):
        if not isinstance(turn, dict):
            continue
        for op in [turn.get("farmer")] + list(turn.get("hands") or []):
            if isinstance(op, list) and op:
                unit_ops[op[0]] = unit_ops.get(op[0], 0) + 1
        for order in turn.get("market") or []:
            if not (isinstance(order, list) and order):
                continue
            if order[0] == "HIRE":
                hires += 1
            elif order[0] == "BUY_LAND":
                land += 1
            if len(order) >= 3:
                key = f"{order[0]} {order[1]}"
                market[key] = market.get(key, 0) + int(order[2] or 0)
                if order[0] == "SELL":
                    sell_days.add(t // 24)
                    first_sale.setdefault(order[1], t)

    out = {
        "architecture": "open-loop route + tape_runtime safety stack",
        "route turns": len(route),
        "HIRE orders": hires,
        "BUY_LAND orders": land,
        "days with a sale": len(sell_days),
        "max hands": max((len(t.get("hands") or []) for t in route
                          if isinstance(t, dict)), default=0),
    }
    for op, n in sorted(unit_ops.items(), key=lambda kv: -kv[1])[:10]:
        out[f"unit op {op}"] = n
    for key, n in sorted(market.items(), key=lambda kv: -kv[1])[:14]:
        out[f"market {key}"] = n
    for item, turn in sorted(first_sale.items(), key=lambda kv: kv[1]):
        out[f"first SELL {item}"] = f"turn {turn} (day {turn // 24})"
    return out


def generate_missing(pattern="agents/v*.py", opponent="starter", seed=11,
                     every=96, force=False, verbose=True):
    """Build a graph for every agent that lacks a current one.

    CLAUDE.md says an agent shipped without a graph is incomplete, and by the
    time anyone checked, 25 of 31 agents had none -- the rule was there, the
    enforcement was not. This is the enforcement, and `src/kaggriculture/pipeline/autopilot.py`'s
    build stage calls it so a new agent cannot reach the gate ungraphed.

    "Current" means newer than the agent file: a graph built before the last
    edit describes a model that no longer exists, which is worse than no graph.
    """
    import glob as _glob
    made, skipped, failed = [], [], []
    for path in sorted(_glob.glob(os.path.join(ROOT, pattern))):
        out = path[:-3] + ".html"
        if (not force and os.path.exists(out)
                and os.path.getmtime(out) >= os.path.getmtime(path)):
            skipped.append(os.path.basename(path))
            continue
        try:
            P = paramio.load(path)
        except ValueError:
            P = _route_params(path)
        except Exception as exc:                                   # noqa: BLE001
            failed.append((os.path.basename(path), str(exc)[:60]))
            continue
        try:
            frames, mine, theirs = run(path, opponent, seed, every)
            render(path, P, frames, mine, theirs, opponent, seed, out)
            made.append(os.path.basename(out))
            if verbose:
                print(f"  built {os.path.relpath(out, ROOT)}")
        except Exception as exc:                                   # noqa: BLE001
            failed.append((os.path.basename(path), str(exc)[:60]))
            if verbose:
                print(f"  ! {os.path.basename(path)}: {str(exc)[:60]}")
    if verbose:
        print(f"\n{len(made)} built, {len(skipped)} already current, "
              f"{len(failed)} failed")
        for name, why in failed:
            print(f"  ! {name}: {why}")
    return made, skipped, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", nargs="?")  # noqa: E501
    ap.add_argument("--missing", action="store_true",
                    help="build a graph for every agent that lacks a current one")
    ap.add_argument("--pattern", default="agents/v*.py")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--every", type=int, default=96, help="sample a state every N turns")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.missing or not args.agent:
        generate_missing(pattern=args.pattern, opponent=args.opponent,
                         seed=args.seed, every=args.every, force=args.force)
        return 0

    agent = os.path.abspath(os.path.join(ROOT, args.agent)) \
        if not os.path.isabs(args.agent) else args.agent
    if not os.path.exists(agent):
        sys.exit(f"no such agent: {agent}")
    out = args.out or agent[:-3] + ".html"
    # A route agent has no PARAMS block -- its configuration *is* the 719-turn
    # route. Rather than fail, describe what it actually carries, so
    # "every model needs a graph" stays true for both architectures.
    try:
        P = paramio.load(agent)
    except ValueError:
        P = _route_params(agent)
    frames, mine, theirs = run(agent, args.opponent, args.seed, args.every)
    p = render(agent, P, frames, mine, theirs, args.opponent, args.seed, out)
    print(f"wrote {p}  ({os.path.getsize(p):,} bytes, {len(frames)} states)")


if __name__ == "__main__":
    main()
