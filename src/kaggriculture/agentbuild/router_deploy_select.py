"""Deployed-form specialist selection for the router.

Fixes the 2026-08-31 selection flaw: specialists used to be screened as
standalone tapes played from step 0, but live they only ever run as a
suffix after the base tape's prefix (dispatch switches when the first
shop unlocks). Here every candidate is evaluated as the ACTUAL composite
the ladder would run: a mini-router {base, first={shop: cand}} played
paired against base-only on that shop's seed pool, vs real reactive
references. A specialist earns its seat ONLY on discordant wins with
zero discordant losses (win currency, not bank).

Usage:
    python src/router_deploy_select.py                 # select + report
    python src/router_deploy_select.py --write-config  # also update
                                                       # models/router/router_config.json
"""
import argparse
import json
import os
import sys


import kaggriculture.agentbuild.router_agent as RA  # noqa: E402
import kaggriculture.engine.serve_match as SM  # noqa: E402

MODELS = "models/router"
WORK = ".local/router/deploy_select"
REFS = ["agents/v40.0_bandit.py"]   # extended below from serve_gate OPPS
SEEDS_PER_POOL = 12
TOP_CANDIDATES = 10


def _refs():
    refs = list(REFS)
    try:
        import kaggriculture.engine.serve_gate as serve_gate
        for p in serve_gate.OPPS[:2]:
            if p not in refs and os.path.exists(p):
                refs.append(p)
    except Exception:                                      # noqa: BLE001
        pass
    return refs


def _pools():
    shops = json.load(open(os.path.join(MODELS, "seed_shops.json")))
    pools = {}
    for seed, sh in shops.items():
        f = sh[0] if isinstance(sh, (list, tuple)) else sh
        pools.setdefault(f, []).append(int(seed))
    return {f: sorted(s)[:SEEDS_PER_POOL] for f, s in pools.items()
            if len(s) >= 6}


def _play_pool(path, ref, seeds, srv):
    """Paired win cells for `path` vs `ref` over seeds x both seats."""
    cells = {}
    for seed in seeds:
        for seat in (0, 1):
            ca, cr = SM.load_agent(path), SM.load_agent(ref)
            a, b = (SM.run_match(ca, cr, seed, srv) if seat == 0
                    else SM.run_match(cr, ca, seed, srv)[::-1])
            cells[(seed, seat)] = a > b
    return cells


def select(write_config=False):
    os.makedirs(WORK, exist_ok=True)
    cfg = json.load(open(os.path.join(MODELS, "router_config.json")))
    sl = json.load(open(os.path.join(MODELS, "shortlist.json")))
    cand_ids = list(sl["shortlist"])[:TOP_CANDIDATES]
    base_only = os.path.join(WORK, "base_only.py")
    RA.build({"base": cfg["base"], "first": {}, "pair": {}}, base_only)

    srv = SM.Serve()
    refs = _refs()
    pools = _pools()
    chosen, report = {}, {}
    for shop, seeds in sorted(pools.items()):
        # base-only reference record on this pool, per ref
        base_cells = {r: _play_pool(base_only, r, seeds, srv) for r in refs}
        best = None
        for rid in cand_ids:
            if rid == cfg["base"]:
                continue
            comp = os.path.join(WORK, f"comp_{shop}_{rid}.py")
            try:
                RA.build({"base": cfg["base"], "first": {shop: rid},
                          "pair": {}}, comp)
            except SystemExit as e:      # builder invariant refusal
                report.setdefault(shop, []).append(
                    {"rid": rid, "refused": str(e)})
                continue
            dw = dl = 0
            for r in refs:
                cells = _play_pool(comp, r, seeds, srv)
                for k, won in cells.items():
                    if won and not base_cells[r][k]:
                        dw += 1
                    elif base_cells[r][k] and not won:
                        dl += 1
            report.setdefault(shop, []).append(
                {"rid": rid, "disc_wins": dw, "disc_losses": dl})
            if dl == 0 and dw > 0 and (best is None or dw > best[1]):
                best = (rid, dw)
        if best:
            chosen[shop] = best[0]
            print(f"{shop}: specialist {best[0]} "
                  f"(+{best[1]} discordant wins, 0 losses)")
        else:
            print(f"{shop}: no candidate beats base in deployed form")
    json.dump(report, open(os.path.join(WORK, "report.json"), "w"), indent=1)
    if write_config:
        cfg["first"] = chosen
        cfg["selected_by"] = "router_deploy_select (deployed-form, "
        cfg["selected_by"] += "discordant-wins-only)"
        json.dump(cfg, open(os.path.join(MODELS, "router_config.json"), "w"),
                  indent=1)
        print("router_config.json updated:", chosen or "base-only")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()
    select(write_config=args.write_config)
