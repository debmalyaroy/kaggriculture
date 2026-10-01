"""Build OUR OWN broad opening-fingerprint dataset over the whole route corpus.

A public notebook (Revanth Tambisetty, 2026-08-09) fingerprints the top-5 by
extracting each team's first-48-turn opening signature from 15 replays and
clustering it into three families (v23_fork / sheep_first_hybrid /
counter_meta). We already harvest the same underlying episodes -- thousands of
them, not fifteen -- so rather than ingest their small hand-labelled CSV we
regenerate the analysis from our full index and get a far broader, self-owned
map of the field.

For each route we accumulate the opening BUILD signature over the first 48
turns (2 in-game days): hires, animals, seeds, land and buildings. Identity in
this game commits in the opening -- what you HIRE/BUY/PLANT -- and that is
exactly what a competitor's first-48-turn detector keys on. Sell curves are a
weaker, later channel. Routes are then greedily clustered on that signature and
each cluster is summarised (size, distinct teams, best ladder rank, freshest
date, final-bank distribution, win rate against the field).

Crucially it also locates OUR shipped base routes in this map: which cluster we
occupy and how saturated it is. That number is the input to the obfuscation
decision (src/obfuscate.py) -- a unique opening is uniquely identifiable; a
crowded one hides in the family.

    python src/fingerprint_dataset.py                 # build + report
    python src/fingerprint_dataset.py --link 4        # linkage threshold
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402

OPEN_TURNS = 48                       # 2 in-game days (24 turns/day)
OUT_DIR = os.path.join(ROOT, "data", "fingerprints")
ANIMALS = ("COW", "SHEEP", "GOOSE")
SEED_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
# Order of the flat signature vector used for clustering + CSV columns.
SIG_KEYS = (["hires", "land"] + [a.lower() for a in ANIMALS]
            + [c.lower() + "_seed" for c in SEED_CROPS]
            + ["pasture", "coop"])

# Revanth's three public clusters, by their published day-0 signatures, so we
# can label our own clusters against the taxonomy the field already reads.
PUBLIC_CLUSTERS = {
    "v23_fork":           {"hires": 5, "cow": 2, "sheep": 2, "goose": 0,
                           "wheat_seed": 7, "melon_seed": 12, "strawberry_seed": 0},
    "sheep_first_hybrid": {"hires": 3, "cow": 1, "sheep": 4, "goose": 0,
                           "wheat_seed": 5, "melon_seed": 5},
    "counter_meta":       {"hires": 14, "cow": 3, "sheep": 2, "goose": 0,
                           "wheat_seed": 14, "melon_seed": 3},
}


def opening_signature(actions):
    """Counts of every opening BUILD action over the first OPEN_TURNS turns."""
    sig = {k: 0 for k in SIG_KEYS}
    first_sell = None
    for t in range(min(OPEN_TURNS, len(actions))):
        for o in (actions[t].get("market") or []):
            if not (isinstance(o, list) and o):
                continue
            op = o[0]
            if op == "HIRE":
                sig["hires"] += 1
            elif op == "BUY_LAND":
                sig["land"] += 1
            elif op == "BUY_ANIMAL" and len(o) >= 2 and o[1] in ANIMALS:
                sig[o[1].lower()] += int(o[2]) if len(o) >= 3 else 1
            elif op == "BUY_SEED" and len(o) >= 3 and o[1] in SEED_CROPS:
                sig[o[1].lower() + "_seed"] += max(0, int(o[2] or 0))
            elif op == "SELL" and first_sell is None:
                first_sell = t
        for u in [actions[t].get("farmer")] + list(actions[t].get("hands") or []):
            if isinstance(u, list) and u:
                if u[0] == "BUILD_PASTURE":
                    sig["pasture"] += 1
                elif u[0] == "BUILD_COOP":
                    sig["coop"] += 1
    sig["first_sell"] = first_sell if first_sell is not None else OPEN_TURNS
    return sig


def vec(sig):
    return [float(sig[k]) for k in SIG_KEYS]


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def cluster(items, link):
    """Greedy single-pass linkage on the opening vector. `items` is a list of
    (key, sig, vec); returns list of clusters, each a list of items."""
    clusters = []          # each: {"rep": vec, "members": [item]}
    for it in sorted(items, key=lambda x: sum(x[2]), reverse=True):
        placed = False
        for c in clusters:
            if l1(it[2], c["rep"]) <= link:
                c["members"].append(it)
                placed = True
                break
        if not placed:
            clusters.append({"rep": it[2], "members": [it]})
    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters


def public_label(centroid_sig):
    """Nearest public cluster name (or 'other') for a cluster centroid."""
    best, name = None, "other"
    for nm, ref in PUBLIC_CLUSTERS.items():
        d = sum(abs(centroid_sig.get(k, 0) - v) for k, v in ref.items())
        if best is None or d < best:
            best, name = d, nm
    # only claim the label if it is genuinely close
    return name if best is not None and best <= 6 else "other"


def our_base_signatures():
    """Decode the opening signature of every shipped route agent we ship."""
    import importlib.util
    out = {}
    adir = os.path.join(ROOT, "agents")
    for fn in sorted(os.listdir(adir)):
        if not (fn.endswith("_route.py")):
            continue
        path = os.path.join(adir, fn)
        try:
            spec = importlib.util.spec_from_file_location("_r_" + fn, path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            if hasattr(m, "_ROUTE"):
                out[fn[:-3]] = opening_signature(m._ROUTE)
        except Exception:                                          # noqa: BLE001
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--link", type=float, default=4.0,
                    help="L1 linkage threshold on the opening vector "
                         "(smaller = tighter clusters; default 4)")
    args = ap.parse_args()

    idx = R.load_index()
    recs = list(idx["routes"].values())
    items = []
    per_route = []
    for rec in recs:
        try:
            acts = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        sig = opening_signature(acts)
        items.append((rec["id"], sig, vec(sig)))
        row = {"id": rec["id"], "team": rec.get("team", "?"),
               "rank": rec.get("rank") or "", "date": rec.get("date", ""),
               "bank": rec.get("bank", ""), "won": int(bool(rec.get("won"))),
               "first_sell": sig["first_sell"]}
        row.update({k: sig[k] for k in SIG_KEYS})
        per_route.append(row)
    print(f"{len(items)} routes fingerprinted over the first {OPEN_TURNS} turns")

    clusters = cluster(items, args.link)
    id2cluster = {}
    cluster_rows = []
    sig_by_id = {it[0]: it[1] for it in items}
    rec_by_id = {r["id"]: r for r in recs}
    for ci, c in enumerate(clusters):
        members = c["members"]
        # centroid signature (rounded mean over members)
        centroid = {}
        for k in SIG_KEYS:
            centroid[k] = round(sum(m[1][k] for m in members) / len(members), 1)
        teams = Counter()
        banks = []
        wins = ranks = ranked = 0
        fresh = ""
        for m in members:
            id2cluster[m[0]] = ci
            rec = rec_by_id[m[0]]
            teams[rec.get("team", "?")] += 1
            if isinstance(rec.get("bank"), (int, float)):
                banks.append(float(rec["bank"]))
            wins += int(bool(rec.get("won")))
            if rec.get("rank"):
                ranks += 1
            fresh = max(fresh, rec.get("date", ""))
        best_rank = min((rec_by_id[m[0]].get("rank") or 10 ** 9)
                        for m in members)
        best_rank = "" if best_rank >= 10 ** 9 else best_rank
        banks.sort()
        med = banks[len(banks) // 2] if banks else ""
        cluster_rows.append({
            "cluster": ci, "routes": len(members),
            "teams": len(teams), "best_rank": best_rank, "freshest": fresh,
            "median_bank": round(med) if med != "" else "",
            "win_rate": round(wins / len(members), 3),
            "public_label": public_label(centroid),
            "signature": " ".join(f"{k}={centroid[k]:g}" for k in SIG_KEYS
                                  if centroid[k]),
            "example_teams": ", ".join(
                str(t) for t, _ in teams.most_common(4)),
        })

    # Where do WE sit? Map each shipped base route to its nearest cluster.
    ours = our_base_signatures()
    where = {}
    for name, sig in ours.items():
        v = vec(sig)
        best, ci = None, None
        for i, c in enumerate(clusters):
            d = l1(v, c["rep"])
            if best is None or d < best:
                best, ci = d, i
        cr = cluster_rows[ci]
        where[name] = {
            "cluster": ci, "distance": round(best, 2),
            "cluster_routes": cr["routes"], "cluster_teams": cr["teams"],
            "public_label": cr["public_label"],
            "saturation_note": (
                "UNIQUE - only our own routes near this opening"
                if cr["teams"] <= 1 else
                f"shared with ~{cr['teams']} field teams"),
            "signature": {k: sig[k] for k in SIG_KEYS if sig[k]},
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "openings.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_route[0].keys()))
        w.writeheader()
        for r in per_route:
            r = dict(r); r["cluster"] = id2cluster.get(r["id"], "")
            w.writerow({k: r.get(k, "") for k in per_route[0].keys()})
    with open(os.path.join(OUT_DIR, "clusters.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cluster_rows[0].keys()))
        w.writeheader()
        w.writerows(cluster_rows)
    summary = {"routes": len(items), "clusters": len(clusters),
               "link": args.link, "top_clusters": cluster_rows[:15],
               "where_we_sit": where}
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w",
                            encoding="utf-8"), indent=1, default=str)

    print(f"\n{len(clusters)} opening clusters "
          f"(link {args.link}); top 12 by size:\n")
    print(f"{'#':>3} {'rts':>5} {'tms':>4} {'rank':>5} {'fresh':>10} "
          f"{'medbank':>8} {'win':>5} {'public':<18} signature")
    for cr in cluster_rows[:12]:
        line = (f"{cr['cluster']:>3} {cr['routes']:>5} {cr['teams']:>4} "
                f"{str(cr['best_rank']):>5} {cr['freshest']:>10} "
                f"{str(cr['median_bank']):>8} {cr['win_rate']:>5} "
                f"{cr['public_label']:<18} {cr['signature'][:46]}")
        print(line.encode("ascii", "replace").decode())
    print("\nWhere our shipped base routes sit:")
    for name, w in sorted(where.items()):
        print(f"  {name:<16} cluster {w['cluster']:>2} "
              f"({w['public_label']}, {w['saturation_note']})")
    print(f"\nwrote {os.path.relpath(OUT_DIR, ROOT)}/"
          "{openings,clusters}.csv + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
