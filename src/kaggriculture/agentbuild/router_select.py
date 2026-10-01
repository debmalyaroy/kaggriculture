"""Staged candidate selection for the match-history router (ShopForge
discipline: every slot earned by PLAYED GAMES on serve, staged seeds,
both seats, non-regression guards; never vote popularity).

Stages:
  1. screen: every pool candidate, 1 seed x 2 seats vs the reference
  2. shortlist: top --short survivors, --mid seeds x 2 seats
  3. base final: top 5 on FRESH seeds -> the base history
  4. per-shop: on each first-shop's seed pool, top shortlist candidates
     paired vs the base; a specialist must beat the base on those seeds
     AND not regress against the outside-opponent set.

    python src/router_select.py --stage screen
    python src/router_select.py --stage shop --base <route_id>
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(ROOT, "models", "router")
POOL_DIR = os.path.join(ROOT, ".local", "router", "pool")
REFS = [os.path.join(ROOT, "data", "gauntlet", "kaito_v48.py"),
        os.path.join(ROOT, "data", "gauntlet", "mutoy.py"),
        os.path.join(ROOT, "data", "gauntlet", "pub_rayk_c94.py")]

_SRV = None


def _play(job):
    path_a, path_b, seed = job
    import sys as _s
    global _SRV
    import kaggriculture.engine.serve_match as SM
    if _SRV is None:
        _SRV = SM.Serve()
    a, b = SM.load_agent(path_a), SM.load_agent(path_b)
    try:
        return SM.run_match(a, b, seed, srv=_SRV)
    except Exception:
        try:
            _SRV.close()
        except Exception:
            pass
        _SRV = None
        return (0.0, 10 ** 9)          # a crashing candidate loses hard


def render_pool(max_candidates=250):
    """Pool: unique winning histories of the CURRENT engine's top teams,
    strongest banks first, rendered as standalone tape agents."""
    import kaggriculture.data.routes as R
    import kaggriculture.train.train_arms as train_arms
    idx = R.load_index()
    recs = [r for r in idx["routes"].values()
            if r.get("won") and r.get("engine") == "1.32.7"
            and r.get("source") not in ("majority-vote", "trackp-factory")
            and float(r.get("bank") or 0) >= 90000]
    recs.sort(key=lambda r: -(float(r.get("bank") or 0)))
    os.makedirs(POOL_DIR, exist_ok=True)
    seen_hash, made = set(), []
    for r in recs:
        if len(made) >= max_candidates:
            break
        try:
            t = R.load_route(r["id"])
        except OSError:
            continue
        h = hash(json.dumps(t, sort_keys=True, separators=(",", ":")))
        if h in seen_hash:
            continue
        seen_hash.add(h)
        p = os.path.join(POOL_DIR, f"{r['id']}.py")
        if not os.path.exists(p):
            train_arms.render(t, p, f"pool {r['id']} ({r.get('team')})")
        made.append((r["id"], p, r.get("team"),
                     float(r.get("bank") or 0)))
    json.dump([[m[0], m[2], m[3]] for m in made],
              open(os.path.join(OUT_DIR, "pool.json"), "w"))
    print(f"pool: {len(made)} unique winning histories "
          f"(banks {made[-1][3]:,.0f}..{made[0][3]:,.0f})")
    return made


def score_batch(cands, seeds, workers, ref=None):
    """Own-score of each candidate vs ref (default kaito) on seeds x2."""
    ref = ref or REFS[0]
    jobs, meta = [], []
    for rid, path in cands:
        for s in seeds:
            jobs.append((path, ref, s))
            meta.append((rid, s, 0))
            jobs.append((ref, path, s))
            meta.append((rid, s, 1))
    agg = {rid: [] for rid, _ in cands}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (rid, s, seat), banks in zip(meta, ex.map(_play, jobs,
                                                      chunksize=1)):
            own, opp = banks[seat], banks[1 - seat]
            agg[rid].append((1.0 if own > opp else
                             (0.5 if own == opp else 0.0), own))
    out = {}
    for rid, cells in agg.items():
        out[rid] = (sum(c[0] for c in cells) / len(cells),
                    sum(c[1] for c in cells) / len(cells))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("screen", "shop"), required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--short", type=int, default=24)
    ap.add_argument("--mid", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.stage == "screen":
        made = render_pool()
        cands = [(m[0], m[1]) for m in made]
        s1 = score_batch(cands, [300001], args.workers)
        keep = sorted(s1, key=lambda r: -(s1[r][0] * 1000 + s1[r][1] / 1000)
                      )[:args.short * 2]
        print(f"stage1: kept {len(keep)}")
        cands2 = [(rid, os.path.join(POOL_DIR, f"{rid}.py")) for rid in keep]
        s2 = score_batch(cands2, [300011, 300012, 300013], args.workers)
        keep2 = sorted(s2, key=lambda r: -(s2[r][0] * 1000 + s2[r][1] / 1000)
                       )[:args.short]
        cands3 = [(rid, os.path.join(POOL_DIR, f"{rid}.py")) for rid in keep2]
        s3 = score_batch(cands3, [300021, 300022, 300023, 300024],
                         args.workers)
        final = sorted(s3, key=lambda r: -(s3[r][0] * 1000 + s3[r][1] / 1000))
        json.dump({"shortlist": final,
                   "scores": {r: s3[r] for r in final}},
                  open(os.path.join(OUT_DIR, "shortlist.json"), "w"),
                  indent=1)
        for r in final[:10]:
            print(f"  {r:<20} score {s3[r][0]:.3f} bank {s3[r][1]:,.0f}")
        # INSTALL the winner as the config base (2026-09-01: the screen
        # stage printed a new base but never wrote it — deploy-select then
        # built every composite against the stale base).
        cfgp = os.path.join(OUT_DIR, "router_config.json")
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
        if final and cfg.get("base") != final[0]:
            cfg["base"] = final[0]
            cfg["first"] = {}          # stale specialists die with old base
            cfg["pair"] = {}
            json.dump(cfg, open(cfgp, "w"), indent=1)
            print(f"router_config.json base -> {final[0]} (specialists reset)")
        print(f"base candidate: {final[0]}")

    elif args.stage == "shop":
        short = json.load(open(os.path.join(OUT_DIR, "shortlist.json")))
        base = args.base or short["shortlist"][0]
        seed_map = json.load(open(os.path.join(OUT_DIR, "seed_shops.json")))
        by_shop = {}
        for sd, shops in seed_map.items():
            if shops:
                by_shop.setdefault(shops[0], []).append(int(sd))
        cfg = {"base": base, "first": {}, "pair": {}}
        base_path = os.path.join(POOL_DIR, f"{base}.py")
        top = short["shortlist"][:10]
        for shop, seeds in sorted(by_shop.items()):
            seeds = seeds[:6]
            cands = [(rid, os.path.join(POOL_DIR, f"{rid}.py"))
                     for rid in top]
            sc = score_batch(cands, seeds, args.workers)
            best = max(sc, key=lambda r: (sc[r][0], sc[r][1]))
            b_sc = sc.get(base, (0, 0))
            if best != base and sc[best][0] > b_sc[0] + 0.05:
                # outside-opponent non-regression on 2 refs
                ok = True
                for ref in REFS[1:]:
                    r_best = score_batch(
                        [(best, os.path.join(POOL_DIR, f"{best}.py"))],
                        seeds[:3], args.workers, ref=ref)[best]
                    r_base = score_batch(
                        [(base, base_path)], seeds[:3],
                        args.workers, ref=ref)[base]
                    if r_best[0] < r_base[0] - 0.1:
                        ok = False
                        break
                if ok:
                    cfg["first"][shop] = best
                    print(f"{shop}: specialist {best} "
                          f"({sc[best][0]:.3f} vs base {b_sc[0]:.3f})")
                    continue
            print(f"{shop}: base retained ({b_sc[0]:.3f})")
        json.dump(cfg, open(os.path.join(OUT_DIR, "router_config.json"),
                            "w"), indent=1)
        print("wrote models/router/router_config.json")


if __name__ == "__main__":
    main()
