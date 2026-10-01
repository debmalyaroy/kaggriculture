"""Census of bot families across every mined route and all of our own games.

Clusters all routes in the index by behavioral signature, then labels every
game our submissions ever played with the nearest family. Output: the family
map -- who runs what, how big each family is, how fresh, and how each one
does against us. Identity is behavioral only; team names are display labels.

    python -m kaggriculture.data.families            # sell-curve clustering (fast)
    python -m kaggriculture.data.families --broad    # full 283-dim signature (features.py)
                                        # + cross-episode consistency stats
"""
from kaggriculture.paths import ROOT
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402

TRACKED = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
           "WHEAT", "FERTILIZER")
SAMPLE = 30           # cumulative curve sampled every SAMPLE turns
LINK = 150            # units of L1 distance that still count as "same program"


def curve_of(actions):
    cum = {i: 0 for i in TRACKED}
    out = []
    for t in range(min(720, len(actions))):
        for o in (actions[t].get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] in TRACKED):
                try:
                    cum[o[1]] += max(0, int(o[2]))
                except (TypeError, ValueError):
                    pass
        if t % SAMPLE == 0:
            out.extend(cum[i] for i in TRACKED)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--broad", action="store_true",
                    help="cluster on the full behavioral signature "
                         "(features.py) instead of sell curves only, and "
                         "report per-family cross-episode consistency")
    ap.add_argument("--link", type=float, default=None,
                    help="linkage threshold override")
    args = ap.parse_args()

    if args.broad:
        return main_broad(args)
    idx = R.load_index()
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    print(f"{len(rows)} routes in the index")
    families = []        # each: {"rep": curve, "members": [rec], "final": cum}
    for rec in rows:
        try:
            acts = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        c = curve_of(acts)
        placed = False
        for fam in families:
            if sum(abs(a - b) for a, b in zip(c, fam["rep"])) < LINK * len(c) / 240:
                fam["members"].append(rec)
                placed = True
                break
        if not placed:
            final = {i: c[-len(TRACKED):][k] for k, i in enumerate(TRACKED)}
            families.append({"rep": c, "members": [rec], "final": final})
    families.sort(key=lambda f: -len(f["members"]))
    print(f"{len(families)} families\n")

    # Label all of our own games by nearest family final-sales signature.
    ours = {}
    og = os.path.join(ROOT, "data", "ourgames", "index.json")
    if os.path.exists(og):
        games = json.load(open(og, encoding="utf-8"))["games"]
        for g in games.values():
            opp = {i: int((g.get("opp_sold") or {}).get(i, 0) or 0) for i in TRACKED}
            total = sum(opp.values())
            if total < 50:
                continue
            best, who = None, None
            for fi, fam in enumerate(families):
                d = sum(abs(opp[i] - fam["final"][i]) for i in TRACKED)
                if best is None or d < best:
                    best, who = d, fi
            if best is not None and best <= 0.35 * total:
                rec = ours.setdefault(who, {"n": 0, "w": 0, "l": 0, "t": 0,
                                            "margin": 0.0})
                rec["n"] += 1
                key = "t" if g["tied"] else ("w" if g["won"] else "l")
                rec[key] += 1
                rec["margin"] += float(g.get("margin") or 0)

    print(f"{'#':>3} {'routes':>6} {'teams':>5} {'best rank':>9} "
          f"{'freshest':>10}  {'our games W-L-T (margin)':<28} example teams")
    report = []
    for fi, fam in enumerate(families[:20]):
        m = fam["members"]
        teams = {r.get("team") for r in m}
        rank = min((r.get("rank") or 999) for r in m)
        fresh = max(r.get("date", "") for r in m)
        o = ours.get(fi)
        ostr = (f"{o['w']}-{o['l']}-{o['t']} ({o['margin']:+,.0f})"
                if o else "-")
        names = ", ".join(sorted(str(t) for t in teams)[:3])
        line = (f"{fi:>3} {len(m):>6} {len(teams):>5} {rank:>9} "
                f"{fresh:>10}  {ostr:<28} {names[:44]}")
        print(line.encode("ascii", "replace").decode())
        report.append({"family": fi, "routes": len(m), "teams": len(teams),
                       "best_rank": rank, "freshest": fresh,
                       "our_games": o, "teams_sample": sorted(str(t) for t in teams)[:6]})
    out = os.path.join(ROOT, "models", "v22", "family_census.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


def main_broad(args):
    """Broad-signature clustering + cross-episode consistency.

    Consistency separates open-loop tapes (a team's episodes nearly
    identical) from reactive agents (episodes differ per opponent) -- which
    changes how the bandit should treat an identified family."""
    import kaggriculture.data.features as features
    idx = R.load_index()
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    link = args.link if args.link is not None else 0.055
    print(f"{len(rows)} routes, broad signature, link {link}")
    # Offline enrichment: append observation-features (reactivity, shop-mix,
    # microstructure) when the mine captured them. Valid HERE because
    # clustering is offline over full replays -- unlike the shipped
    # identifier, which must stay on runtime-reconstructable features.
    obs_dir = os.path.join(ROOT, "data", "obsfeat")

    def obs_of(rid):
        p = os.path.join(obs_dir, rid + ".json")
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                pass
        return [0.0] * 74

    feats = {}
    n_obs = 0
    for rec in rows:
        try:
            base, _ = features.route_features(R.load_route(rec["id"]))
            of = obs_of(rec["id"])
            if any(of):
                n_obs += 1
            feats[rec["id"]] = base + [0.3 * v for v in of]  # down-weighted
        except Exception:                                          # noqa: BLE001
            continue
    print(f"  {n_obs} routes carry observation-features")
    reps, fams = [], []
    for rec in rows:
        f = feats.get(rec["id"])
        if f is None:
            continue
        placed = False
        for fi, rep in enumerate(reps):
            if features.distance(f, rep) < link:
                fams[fi].append(rec)
                placed = True
                break
        if not placed:
            reps.append(f)
            fams.append([rec])
    order = sorted(range(len(fams)), key=lambda i: -len(fams[i]))
    print(f"{len(fams)} families\n")
    print(f"{'#':>3} {'routes':>6} {'teams':>5} {'best':>5} {'fresh':>10} "
          f"{'consist':>8}  teams")
    out = []
    for rank, fi in enumerate(order[:25]):
        members = fams[fi]
        teams = {}
        for r in members:
            teams.setdefault(r.get("team"), []).append(r["id"])
        # Cross-episode consistency: mean pairwise distance of one team's
        # routes inside this family (0 = pure tape, high = reactive).
        dists = []
        for t, ids in teams.items():
            for i in range(min(3, len(ids))):
                for j in range(i + 1, min(3, len(ids))):
                    dists.append(features.distance(feats[ids[i]], feats[ids[j]]))
        consist = (sum(dists) / len(dists)) if dists else -1.0
        best = min((r.get("rank") or 999) for r in members)
        fresh = max(r.get("date", "") for r in members)
        names = ", ".join(sorted(str(t) for t in teams)[:3])
        line = (f"{rank:>3} {len(members):>6} {len(teams):>5} {best:>5} "
                f"{fresh:>10} {consist:>8.3f}  {names[:40]}")
        print(line.encode("ascii", "replace").decode())
        out.append({"family": rank, "routes": len(members),
                    "teams": len(teams), "best_rank": best, "fresh": fresh,
                    "consistency": consist,
                    "rep_id": members[0]["id"],
                    "teams_sample": sorted(str(t) for t in teams)[:6]})
    dest = os.path.join(ROOT, "models", "v22", "family_census_broad.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\nwrote {os.path.relpath(dest, ROOT)}")
    return 0


if __name__ == "__main__":
    main()
