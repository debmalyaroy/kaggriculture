"""Sibling-rank dataset: the searcher's OWN plan variants, rolled to terminal.

Why (2026-09-07): four same-recipe value-net retrains (v2/v3/v4/v5) each
raised validation sign-acc and none won a duel — leaf accuracy on random
tape states does not transfer to the deployed question. The searcher only
ever RANKS SIBLINGS: "from the state I am at, which of these plan
variants (sched/now/hold12/hold24/defer/invest) ends better?" This
generator produces exactly that distribution:

  per game: base route vs a pooled opponent tape (GENGAME);
  per decision point (sampled day boundaries, day 8..23): rebuild the
  searcher's plan variants from the route window + the observed shed and
  prices (same semantics as build_searcher's _plan_variants family);
  per variant: materialize the full 719-step tape (plan orders + the
  window's non-SELL keeps, identical to _ro_lines) and play it to
  terminal in ONE GENGAME — the day obs at decision+48 is the LEAF the
  runtime net would score, the final margin is the label.

Row: {"seed","world","gid","variant","f":[39 leaf feats],"y_margin"}
Groups (same gid) are siblings; the trainer pairs within groups.

    python src/trackp/harness/sibling_gen.py --games 3000 --shard 0
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import copy
import json
import os
import random
import re
import sys
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

H = 48                                   # searcher rollout horizon
BASE_PRICE = {"WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0,
              "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0,
              "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0}


def our_route():
    src = open(os.path.join(ROOT, "agents", "v45.0_bandit.py"),
               encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    return json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))


def window_sells(route, idx):
    win = []
    for j in range(idx, min(len(route), idx + H)):
        for o in (route[j].get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                win.append((j - idx, str(o[1]), int(o[2] or 0)))
    return win


def plan_variants(route, idx, shed, prices):
    """The searcher's variant set, replicated (build_searcher.py
    _plan_variants/_hold_variants/_prod_variants/_add_defer_variants)."""
    win = window_sells(route, idx)
    plans = {"sched": {}}
    for off, item, q in win:
        plans["sched"].setdefault(off, []).append(["SELL", item, q])
    now = {}
    have = dict(shed)
    for off, item, q in sorted(win):
        take = min(q, max(0, int(have.get(item, 0) or 0)))
        if take > 0:
            now.setdefault(0, []).append(["SELL", item, take])
            have[item] = have.get(item, 0) - take
        rem = q - take
        if rem > 0:
            now.setdefault(off, []).append(["SELL", item, rem])
    plans["now"] = now
    for delay in (12, 24):
        d = {}
        for off, item, q in win:
            d.setdefault(min(off + delay, H - 1), []).append(
                ["SELL", item, q])
        plans["hold%d" % delay] = d
    # invest: pull the window's first BUY 12 steps earlier
    first_buy = None
    for j in range(idx, min(len(route), idx + H)):
        for o in (route[j].get("market") or []):
            if (isinstance(o, list) and o
                    and o[0] in ("BUY_SEED", "BUY_ANIMAL")):
                first_buy = (j - idx, list(o))
                break
        if first_buy:
            break
    if first_buy and first_buy[0] >= 12:
        off, op = first_buy
        d = dict(plans["sched"])
        d[max(0, off - 12)] = list(d.get(max(0, off - 12), [])) + [op]
        plans["invest"] = d
    # defer_<item>: withhold a crashed item's sells entirely
    vol = {}
    for off, item, q in win:
        vol[item] = vol.get(item, 0) + q
    for item, v in vol.items():
        if v < 20:
            continue
        if float(prices.get(item, 0) or 0) >= 0.5 * BASE_PRICE.get(item, 25.0):
            continue
        d = {}
        for off, it2, q in win:
            if it2 == item:
                continue
            d.setdefault(off, []).append(["SELL", it2, q])
        plans["defer_" + item] = d
    return plans


def variant_route(route, idx, plan):
    """Full route with the window rewritten per the plan — identical
    semantics to the searcher's _ro_lines (plan orders + non-SELL keeps)."""
    var = route[:idx] + copy.deepcopy(route[idx:min(len(route), idx + H)]) \
        + route[idx + H:]
    for h in range(min(H, len(route) - idx)):
        r = var[idx + h]
        keep = [o for o in (r.get("market") or [])
                if isinstance(o, list) and o and o[0] != "SELL"]
        r["market"] = (list(plan.get(h, [])) + keep)[:10]
    return var


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--points", type=int, default=6,
                    help="decision points sampled per game")
    ap.add_argument("--kagg", default=None)
    a = ap.parse_args()
    import kaggriculture.engine.serve_match as SM
    from selfplay_gen import state_features
    if a.kagg:
        SM.KAGG = a.kagg
        os.environ["KAGG_BIN"] = a.kagg
    from kaggriculture.trackp import routes_io as R
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    pool = [h["id"] for h in hits]
    idx_all = R.load_index()
    seen_teams = set()
    for k, v in sorted(idx_all["routes"].items(),
                       key=lambda t: -(t[1].get("bank") or 0)):
        if (v.get("engine") == "1.32.7" and v.get("won")
                and (v.get("bank") or 0) >= 100000
                and str(v.get("date", "")) >= "2026-09-01"
                and v.get("team") not in seen_teams
                and k not in pool):
            pool.append(k)
            seen_teams.add(v.get("team"))
        if len(seen_teams) >= 40:
            break
    route = our_route()[:719]
    rng = random.Random(7000 + a.shard)
    seed_world = {s: w for w, ss in bank.items() for s in ss}
    seeds = list(seed_world)
    outd = os.path.join(ROOT, "data", "selfplay")
    os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, f"siblings_{a.shard}.jsonl")

    def tape_of(rt):
        return chr(31).join(SM.action_to_line(
            rt[i] if i < len(rt) else None) for i in range(719))

    def gengame(srv, seed, ta, tb):
        return srv.cmd("GENGAME " + str(seed) + chr(30) + ta + chr(30) + tb)

    # day-boundary decision points day 8..23 (leaf at +48 stays < 700)
    all_pts = [d * 24 for d in range(8, 24)]
    n_rows = n_groups = 0
    with open(outp, "a", encoding="utf-8") as out:
        srv = SM.Serve()
        try:
            for gi in range(a.games):
                seed = rng.choice(seeds)
                opp = rng.choice(pool)
                try:
                    B = R.load_route(opp)
                except Exception:                              # noqa: BLE001
                    continue
                tb = tape_of(B)
                base = gengame(srv, seed, tape_of(route), tb)
                if "error" in base:
                    print(f"  SERVE-ERROR g{gi}: "
                          f"{str(base['error'])[:120]}", flush=True)
                    continue
                day_obs = {int(ob.get("step") or 0): ob
                           for ob in base.get("days") or []}
                bf = (base.get("final") or {}).get("farms") or [{}, {}]
                base_margin = (float(bf[0].get("money") or 0)
                               - float(bf[1].get("money") or 0))
                pts = rng.sample(all_pts, min(a.points, len(all_pts)))
                for pt in sorted(pts):
                    ob = day_obs.get(pt)
                    if ob is None:
                        continue
                    priv = ob.get("private") or [{}, {}]
                    try:
                        shed = priv[0].get("shed") or {}
                    except Exception:                          # noqa: BLE001
                        shed = {}
                    prices = (ob.get("market") or {}).get("prices") or {}
                    plans = plan_variants(route, pt, shed, prices)
                    if len(plans) < 2:
                        continue
                    gid = f"{seed}:{opp}:{pt}"
                    grp = []
                    for name, plan in plans.items():
                        if name == "sched":
                            leaf = day_obs.get(pt + H)
                            marg = base_margin
                        else:
                            js = gengame(srv, seed,
                                         tape_of(variant_route(
                                             route, pt, plan)), tb)
                            if "error" in js:
                                continue
                            leaf = None
                            for o2 in js.get("days") or []:
                                if int(o2.get("step") or 0) == pt + H:
                                    leaf = o2
                                    break
                            f2 = (js.get("final") or {}).get("farms") \
                                or [{}, {}]
                            marg = (float(f2[0].get("money") or 0)
                                    - float(f2[1].get("money") or 0))
                        if leaf is None:
                            continue
                        grp.append((name, state_features(leaf, 0, pt + H),
                                    marg))
                    if len(grp) < 2:
                        continue
                    n_groups += 1
                    for name, f, marg in grp:
                        out.write(json.dumps(
                            {"seed": seed, "world": seed_world.get(seed),
                             "gid": gid, "variant": name, "f": f,
                             "y_margin": round(marg, 1)},
                            separators=(",", ":")) + "\n")
                        n_rows += 1
                if gi < 3 or (gi + 1) % 100 == 0:
                    print(f"  {gi+1}/{a.games} games, {n_groups} groups, "
                          f"{n_rows} rows", flush=True)
                if gi == 2 and n_rows == 0:
                    raise SystemExit("SANITY: three games, zero rows")
        finally:
            srv.close()
    print(f"{n_rows} sibling rows ({n_groups} groups) -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
